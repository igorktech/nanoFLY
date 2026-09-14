"""Offline regression checks: python -m unittest discover -s tests -v."""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from data.text.prepare import add_arguments, collect, iter_texts, prepare
from nanofly.data import (TokenStream, _encode_documents, char_tokenizer, load_data,
                          train_bpe, write_prepared)
from nanofly.model import FlyConfig, FlyLM, save_checkpoint
from train import optimizer_updates

ROOT = Path(__file__).resolve().parents[1]


def tiny_graph():
    n = 128
    return {"nt": np.zeros(n, np.int64), "nt_names": np.array(["ach"]),
            "pre": np.arange(n), "post": np.roll(np.arange(n), 1),
            "count": np.ones(n), "grp_sensory": np.arange(n)}


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def prepared(self, name="data", val=()):
        path = self.root / name
        write_prepared(str(path), iter(["abcdef " * 8, "abcdef " * 9]), iter(val),
                       kind="char", alphabet="abcdef ", tok_chars=128)
        return path

    def args(self, path, tokenizer="", max_len=0):
        return SimpleNamespace(data=str(path), out=str(self.root / "run"), tokenizer=str(tokenizer),
                               max_len=max_len, limit=0, val_limit=0, val_frac=.02)

    def test_empty_validation_and_full_document(self):
        path = self.prepared()
        train, val, _, _ = load_data(self.args(path))
        self.assertEqual(len(val), 0)
        self.assertEqual(len(train[0]), len("abcdef " * 8) + 2)
        self.assertEqual(train.n_tokens, 56 + 63 + 2)

    def test_same_size_wrong_tokenizer_refused(self):
        path = self.prepared()
        wrong = self.root / "wrong.json"
        char_tokenizer("fedcba ").save(str(wrong))
        with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
            load_data(self.args(path, wrong))

    def test_invalid_special_ids_refused(self):
        path = self.prepared()
        tok = json.loads((path / "tokenizer.json").read_text())
        tok["model"]["vocab"]["<pad>"], tok["model"]["vocab"]["<s>"] = 1, 0
        wrong = self.root / "wrong.json"
        wrong.write_text(json.dumps(tok))
        with self.assertRaisesRegex(ValueError, "<pad>=0"):
            load_data(self.args(path, wrong))

    def test_explicit_truncation_warns(self):
        path = self.prepared()
        with self.assertWarnsRegex(UserWarning, "truncates"):
            train, _, _, _ = load_data(self.args(path, max_len=10))
        self.assertEqual(len(train[0]), 10)
        self.assertEqual(train[0][-1], 2)

    def test_raw_file_also_keeps_tail(self):
        path = self.root / "pairs.jsonl"
        path.write_text(json.dumps({"text": "abcdef " * 100, "news": "post"}) + "\n")
        tok = self.root / "char.json"
        char_tokenizer("abcdef ").save(str(tok))
        args = self.args(path, tok)
        args.text_field, args.news_field = "text", "news"
        args.vocab, args.vocab_type, args.alphabet, args.seed = 32, "char", "abcdef ", 0
        train, _, _, _ = load_data(args)
        self.assertGreater(len(train[0]), 320)
        self.assertEqual(train.news, ["post"])

    def test_chunk_covers_every_body_token(self):
        tok = train_bpe(["русские новости и длинная история"], 280)
        text = "русские новости и длинная история " * 100
        chunks = list(_encode_documents(tok, iter([text]), chunk=32))
        self.assertTrue(all(len(c) <= 32 for c in chunks))
        self.assertEqual([t for c in chunks for t in c[1:-1]], tok.encode(text).ids)

    def test_text_limit_does_not_read_whole_file(self):
        class BoundedRead(io.BytesIO):
            def read(self, size=-1):
                if size < 0:
                    raise AssertionError("unbounded read")
                return super().read(size)
        stream = BoundedRead(b"first\nline\n\n" + b"more text\n\n" * 10000)
        self.assertEqual(list(iter_texts("source.txt", stream, "text", 1)), ["first\nline"])

    def test_compressed_and_parquet_sources(self):
        try:
            import zstandard
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("install the data extra for zstd/parquet checks")
        compressed = zstandard.ZstdCompressor().compress(b'{"text":"first"}\n{"text":"second"}\n')
        self.assertEqual(list(iter_texts("x.jsonl.zst", io.BytesIO(compressed), "text", 1)), ["first"])
        path = self.root / "source.parquet"
        pq.write_table(pa.table({"text": ["first", "second", "third"]}), path)
        ap = add_arguments(argparse.ArgumentParser())
        args = ap.parse_args(["--out", str(self.root / "out"), "--local", str(path), "--limit", "2"])
        texts, _ = collect(args)
        self.assertEqual(list(texts), ["first", "second"])

    def test_streamed_mixture_sample_bound_and_split(self):
        a, b = self.root / "a.jsonl", self.root / "b.jsonl"
        for path, source in ((a, "news"), (b, "pikabu")):
            path.write_text("".join(json.dumps({"text": f"{source} document {i} " * 10}) + "\n"
                                    for i in range(100)))
        manifest = self.root / "mix.json"
        manifest.write_text(json.dumps([{"local": str(a), "weight": 2, "limit": 15},
                                        {"local": str(b), "weight": 1, "limit": 10}]))
        ap = add_arguments(argparse.ArgumentParser())
        args = ap.parse_args(["--out", str(self.root / "mix"), "--mix", str(manifest),
                              "--tok-chars", "1000", "--val-frac", ".3", "--vocab", "300"])
        texts, source = collect(args)
        self.assertIs(iter(texts), texts)
        meta = prepare(args, texts, source)
        self.assertEqual(meta["n_train"] + meta["n_val"], 25)
        self.assertLessEqual(meta["tokenizer_sample_chars"], 1000)
        self.assertEqual(meta["source"][0]["weight"], 2)
        self.assertFalse(list((self.root / "mix").glob(".prepare-*")))

    def test_repeated_sources_global_limit_and_stable_split(self):
        path = self.root / "docs.jsonl"
        path.write_text("".join(json.dumps({"text": f"document {i}"}) + "\n" for i in range(100)))
        ap = add_arguments(argparse.ArgumentParser())
        args = ap.parse_args(["--out", str(self.root / "mixed"), "--local", str(path),
                              "--local", str(path), "--limit", "50", "--val-frac", ".5"])
        texts, sources = collect(args)
        self.assertEqual(len(list(texts)), 50)
        texts, sources = collect(args)
        prepare(args, texts, sources)
        tr = TokenStream.from_dir(args.out, "train")
        va = TokenStream.from_dir(args.out, "val")
        self.assertFalse({tuple(tr[i]) for i in range(len(tr))} & {tuple(va[i]) for i in range(len(va))})


class TrainingTests(unittest.TestCase):
    def test_atomic_save_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "latest.pt"
            path.write_bytes(b"previous checkpoint")
            model = FlyLM(FlyConfig(vocab_size=10, d_emb=8, delay=2), tiny_graph())
            def fail(payload, f):
                f.write(b"partial")
                raise OSError("disk full")
            with patch("torch.save", side_effect=fail), self.assertRaises(OSError):
                save_checkpoint(str(path), model, {})
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(d).iterdir()), [path])

    def test_update_count_includes_padding_and_partial_batch(self):
        self.assertEqual(optimizer_updates(np.array([3, 10, 19]), 2, 4, 0, 1), 8)

    def test_resume_mid_document_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            graph = root / "graph.npz"
            np.savez(graph, **tiny_graph())
            data = root / "data"
            write_prepared(str(data), ["abcdef " * n for n in (4, 5, 6)], ["abcdef"],
                           kind="char", alphabet="abcdef ")
            base = [sys.executable, str(ROOT / "train.py"), "--graph", str(graph),
                    "--data", str(data), "--device", "cpu"]
            recipe = ["--arch", "decoder", "--d-emb", "8", "--delay", "2", "--ticks", "1",
                      "--batch", "2", "--seq", "4", "--epochs", "2", "--warmup", "2",
                      "--log-every", "1", "--eval-every", "2", "--save-every", "1"]
            env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            def run(out, flags):
                result = subprocess.run(base + ["--out", str(out)] + flags, cwd=ROOT,
                                        env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                self.assertEqual(result.returncode, 0, result.stdout)
                return result.stdout
            full, resumed = root / "full", root / "resumed"
            run(full, recipe)
            log = run(resumed, recipe + ["--max-steps", "3"])
            partial = torch.load(resumed / "latest.pt", weights_only=False)
            cursor = partial["training_state"]["progress"]
            self.assertEqual(cursor["optimizer_step"], 3)
            self.assertGreater(cursor["window_start"], 0)
            self.assertIsNotNone(cursor["state"])
            self.assertEqual(partial["training_state"]["scheduler"]["last_epoch"], 3)
            self.assertIn("update 2: val loss", log)
            run(resumed, ["--resume", str(resumed / "latest.pt")])
            expected = torch.load(full / "latest.pt", weights_only=False)
            actual = torch.load(resumed / "latest.pt", weights_only=False)
            for key in expected["state_dict"]:
                torch.testing.assert_close(actual["state_dict"][key], expected["state_dict"][key], rtol=0, atol=0)
            ea, aa = expected["training_state"], actual["training_state"]
            self.assertEqual(ea["progress"], aa["progress"])
            self.assertEqual(ea["scheduler"], aa["scheduler"])
            for key, state in ea["optimizer"]["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(aa["optimizer"]["state"][key][field], value, rtol=0, atol=0)
            self.assertEqual(aa["progress"]["optimizer_step"], aa["total_steps"])
            torch.testing.assert_close(ea["rng"]["torch"], aa["rng"]["torch"], rtol=0, atol=0)
            # SIGTERM is deferred to a safe optimizer boundary, leaving a usable latest.pt.
            interrupted = root / "signal"
            proc = subprocess.Popen(base + ["--out", str(interrupted)] + recipe, cwd=ROOT,
                                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                for line in proc.stdout:
                    if "update 1/" in line:
                        proc.send_signal(signal.SIGTERM)
                        break
                output = proc.communicate(timeout=30)[0]
                self.assertEqual(proc.returncode, 0, output)
                interrupted_ck = torch.load(interrupted / "latest.pt", weights_only=False)
                self.assertGreaterEqual(interrupted_ck["training_state"]["progress"]["optimizer_step"], 1)
                run(interrupted, ["--resume", str(interrupted / "latest.pt")])
                after_signal = torch.load(interrupted / "last.pt", weights_only=False)
                for key in expected["state_dict"]:
                    torch.testing.assert_close(after_signal["state_dict"][key], expected["state_dict"][key],
                                               rtol=0, atol=0)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                proc.stdout.close()
            # The same recovery path must work when a post projection and learned edges are present.
            pairs = root / "pairs.jsonl"
            pairs.write_text("".join(json.dumps({"text": "abcdef " * n, "news": "post"}) + "\n"
                                     for n in (4, 5, 6)))
            base[base.index("--data") + 1] = str(pairs)
            conditional = recipe.copy()
            conditional[conditional.index("--arch") + 1] = "encoder-decoder"
            conditional += ["--news-encoder", "hash", "--news-mode", "direct", "--mode", "edges",
                            "--tokenizer", str(data / "tokenizer.json")]
            cf, cr = root / "conditional-full", root / "conditional-resumed"
            run(cf, conditional)
            run(cr, conditional + ["--max-steps", "3"])
            run(cr, ["--resume", str(cr / "latest.pt")])
            ec = torch.load(cf / "last.pt", weights_only=False)
            ac = torch.load(cr / "last.pt", weights_only=False)
            for key in ec["state_dict"]:
                torch.testing.assert_close(ac["state_dict"][key], ec["state_dict"][key], rtol=0, atol=0)
            base[base.index("--data") + 1] = str(data)
            # Mutating even one prepared token must prevent an otherwise compatible resume.
            tokens = np.memmap(data / "train.bin", dtype=np.uint16, mode="r+")
            tokens[1] = 4 if tokens[1] != 4 else 5
            tokens.flush()
            del tokens
            result = subprocess.run(base + ["--out", str(resumed), "--resume", str(resumed / "latest.pt")],
                                    cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("dataset content changed", result.stdout)


if __name__ == "__main__":
    unittest.main()
