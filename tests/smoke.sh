#!/usr/bin/env bash
# The whole pipeline on a synthetic connectome, on CPU, in a couple of minutes: prepare data, train
# the decoder, export it, fine-tune the encoder-decoder from those weights, export that, load both
# folders back as a user would, and sample. Nothing here touches the network.
#
#   tests/smoke.sh            # uses python3
#   PY=venv/bin/python tests/smoke.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
T=${SMOKE_DIR:-/tmp/nanofly-smoke}
rm -rf "$T"; mkdir -p "$T"
say() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

say "synthetic graph"
[ -f graph_synth/graph.npz ] || $PY prepare_malecns.py --synthetic --out graph_synth >/dev/null
$PY - <<'EOF'
import json; m = json.load(open("graph_synth/summary.json"))
print(f"  {m['neurons']:,} neurons, {m['edges']:,} edges, {m['glomeruli']} glomeruli")
EOF

say "prepare data from the demo pairs"
$PY data/text/prepare.py --local data/pairs/pairs_demo.jsonl --field text --vocab 512 \
    --val-frac 0.1 --out "$T/demo"

say "A: decoder"
$PY train.py --graph graph_synth/graph.npz --data "$T/demo" --out "$T/A" --arch decoder \
    --d-emb 64 --delay 4 --max-steps 5 --batch 8 --seq 16 --device cpu 2>&1 | tee "$T/A.log"
grep -q "reserved 200" "$T/A.log" || { echo "FAIL: the post channel was not reserved"; exit 1; }

say "A: export"
$PY export_hf.py --ckpt "$T/A/ckpt.pt" --out "$T/hf_A" 2>&1 | tee "$T/exportA.log"
grep -q "FlyForCausalLM from configuration_fly.py, modeling_fly.py" "$T/exportA.log" \
    || { echo "FAIL: the decoder did not ship as Fly*"; exit 1; }

say "B: encoder-decoder, initialised from A"
$PY train.py --graph graph_synth/graph.npz --data data/pairs/pairs_demo.jsonl --out "$T/B" \
    --arch encoder-decoder --news-encoder hash --news-mode glomeruli --init-from "$T/A/ckpt.pt" \
    --max-steps 5 --batch 8 --seq 16 --device cpu 2>&1 | tee "$T/B.log"
grep -q "fresh: news_proj.bias, news_proj.weight" "$T/B.log" \
    || { echo "FAIL: expected only the post projection to be freshly initialised"; exit 1; }

say "B: a different token population must be refused"
if $PY train.py --graph graph_synth/graph.npz --data data/pairs/pairs_demo.jsonl --out "$T/bad" \
    --arch encoder-decoder --news-encoder hash --init-from "$T/A/ckpt.pt" --token-input sensory \
    --max-steps 1 --batch 8 --device cpu >/dev/null 2>&1; then
  echo "FAIL: --init-from accepted a different input layout"; exit 1
fi
echo "  refused, as it should be"

say "B: export"
$PY export_hf.py --ckpt "$T/B/ckpt.pt" --out "$T/hf_B" 2>&1 | tee "$T/exportB.log"
grep -q "ReplyflyForConditionalGeneration from configuration_replyfly.py" "$T/exportB.log" \
    || { echo "FAIL: the encoder-decoder did not ship as Replyfly*"; exit 1; }

say "load both folders the way a user does"
$PY - "$T" <<'EOF'
import sys, warnings
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM as M
T = sys.argv[1]
for folder, want in ((f"{T}/hf_A", "FlyForCausalLM"), (f"{T}/hf_B", "ReplyflyForConditionalGeneration")):
    got = type(M.from_pretrained(folder, trust_remote_code=True)).__name__
    assert got == want, f"{folder}: {got} != {want}"
    print(f"  {folder.split('/')[-1]}: {got}")
EOF

say "sample"
$PY sample.py --ckpt "$T/B/ckpt.pt" --news "hello" --max-new 8 --device cpu

printf '\n\033[1;32mnanoFLY smoke OK\033[0m  (%s)\n' "$T"
