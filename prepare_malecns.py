#!/usr/bin/env python3
"""
Prepare the full MaleCNS v1.0 connectome (brain + VNC) for fly-LM.

Source: https://male-cns.janelia.org/download/ (FlyEM / HHMI Janelia, University of Cambridge,
MRC LMB, Google Research), licensed CC BY 4.0.

Examples:
  python prepare_malecns.py --download --raw raw/ --out graph/
  python prepare_malecns.py --raw raw/ --inspect           # print columns and values, write nothing
  python prepare_malecns.py --synthetic --out graph_synth/  # a small fake graph for testing the pipeline

Output:
  graph/graph.npz     edges (pre, post, synapse count), transmitter, superclass, coordinates, neuron groups
  graph/summary.json  summary: neuron and edge counts, which columns were used, group sizes
  graph/neurons.tsv   neuron table for eyeballing
"""
import argparse
import json
import os
import re
import ssl
import sys
import time
import urllib.request

import numpy as np

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
FILES = {
    "ann": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "nt": "body-neurotransmitters-male-cns-v1.0.feather",
    "w": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
EXPECTED_NEURONS = 166_700
EXPECTED_EDGES = 25_582_938

NT_NAMES = ["ach", "gaba", "glu", "his", "da", "5ht", "oa", "ta", "unk"]
NT_ALIASES = {
    "acetylcholine": "ach", "ach": "ach", "cholinergic": "ach",
    "gaba": "gaba",
    "glutamate": "glu", "glut": "glu", "glu": "glu",
    "histamine": "his", "his": "his",
    "dopamine": "da", "da": "da",
    "serotonin": "5ht", "5ht": "5ht", "5-ht": "5ht", "5-hydroxytryptamine": "5ht",
    "octopamine": "oa", "oa": "oa",
    "tyramine": "ta", "ta": "ta",
}

ID_CANDIDATES = ["bodyId", "bodyid", "body_id", "body", "id", "segment_id", "segmentId"]
SUPERCLASS_CANDIDATES = ["superclass", "super_class", "superClass"]
CLASS_CANDIDATES = ["class", "cell_class", "cellClass", "neuron_class"]
TYPE_CANDIDATES = ["type", "cell_type", "cellType", "celltype"]
STATUS_CANDIDATES = ["status", "statusLabel", "status_label"]
POS_CANDIDATES = {
    "soma": ["somaLocation", "soma_position", "soma_location", "somaPosition", "soma_pos"],
    "root": ["rootLocation", "root_position", "root_location", "rootPosition"],
    "pos": ["position", "location", "pos", "point"],
}
PRE_CANDIDATES = ["body_pre", "bodyId_pre", "bodyid_pre", "pre", "source", "pre_id", "from"]
POST_CANDIDATES = ["body_post", "bodyId_post", "bodyid_post", "post", "target", "post_id", "to"]
WEIGHT_CANDIDATES = ["weight", "syn_count", "count", "n_syn", "synapses", "w"]

# Groups by superclass (case-insensitive substring) and by type name (regex)
SUPERCLASS_GROUPS = {
    "sensory": r"sensory",
    "descending": r"descending",
    "ascending": r"ascending",
    "motor": r"motor|efferent",
    "visual_projection": r"visual_projection",
}
TYPE_GROUPS = {
    "orn": r"^ORN[_ ]",
    "kc": r"^KC",
    "mbon": r"^MBON",
    "pam": r"^PAM",
    "ppl1": r"^PPL1",
    "photoreceptor": r"^R(1-6|[78])",
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def install_ca_bundle():
    """python.org builds on macOS ship without a CA store, so the default SSL context fails on the
    flyem bucket ("unable to get local issuer certificate"). Use certifi's bundle when it is there."""
    try:
        import certifi
    except ImportError:
        return False
    ctx = ssl.create_default_context(cafile=certifi.where())
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)))
    return True


def download(raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    install_ca_bundle()
    for key, name in FILES.items():
        dst = os.path.join(raw_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            log(f"already here: {name}")
            continue
        url = BASE_URL + name
        log(f"downloading {url}")
        tmp = dst + ".part"
        last = [0.0]

        def hook(blocks, bsize, total):
            done = blocks * bsize
            if time.time() - last[0] > 5:
                last[0] = time.time()
                pct = 100 * done / total if total > 0 else 0
                print(f"    {done / 1e6:8.0f} MB  {pct:5.1f}%", flush=True)

        urllib.request.urlretrieve(url, tmp, hook)
        os.replace(tmp, dst)


def is_str_col(series):
    import pandas as pd
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def read_feather(path, columns=None):
    import pyarrow.feather as feather
    return feather.read_table(path, columns=columns).to_pandas()


def parse_xyz(series):
    """Coordinate column: a list/array [x,y,z], a string 'x,y,z' or None. Returns (N,3) floats and a mask."""
    n = len(series)
    out = np.full((n, 3), np.nan, dtype=np.float64)
    for i, v in enumerate(series.values):
        if v is None:
            continue
        if isinstance(v, str):
            parts = re.findall(r"-?\d+(?:\.\d+)?", v)
            if len(parts) >= 3:
                out[i] = [float(p) for p in parts[:3]]
            continue
        try:
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.size >= 3:
                out[i] = arr[:3]
        except (TypeError, ValueError):
            pass
    ok = np.isfinite(out).all(1) & ~(out == 0).all(1)
    return out, ok


def synapse_positions(path, ids, batch_log=50):
    """Mean synapse position per neuron. Somata sit as a shell around the brain while synapses fill the
    neuropil, so it is the synapses that give the recognisable silhouette of the brain and the VNC."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    reader = ipc.open_file(pa.memory_map(path, "r"))
    cols = reader.schema.names
    c_body = pick(cols, ["body", "bodyId", "bodyid", "body_id", "segment_id"])
    c_x, c_y, c_z = (pick(cols, [a, a.upper(), f"{a}_pre", f"{a}_post"]) for a in "xyz")
    if not (c_body and c_x and c_y and c_z):
        raise SystemExit(f"no body/x/y/z found in {path}: {cols}")
    log(f"synapses: body='{c_body}', xyz='{c_x},{c_y},{c_z}', batches {reader.num_record_batches}")
    n = len(ids)
    acc = np.zeros((n, 3), dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    value_set = pa.array(ids)
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        body = b.column(c_body)
        mask = pc.is_in(body, value_set=value_set)
        body = pc.filter(body, mask).to_numpy(zero_copy_only=False)
        if not len(body):
            continue
        idx = np.searchsorted(ids, body)
        xyz = np.stack([pc.filter(b.column(c), mask).to_numpy(zero_copy_only=False).astype(np.float64)
                        for c in (c_x, c_y, c_z)], 1)
        np.add.at(acc, idx, xyz)
        np.add.at(cnt, idx, 1)
        if (i + 1) % batch_log == 0:
            log(f"  batch {i + 1}/{reader.num_record_batches}, synapses counted {cnt.sum():,}")
    ok = cnt > 0
    pos = np.full((n, 3), np.nan)
    pos[ok] = acc[ok] / cnt[ok, None]
    log(f"mean synapse position known for {ok.sum():,} of {n:,} neurons")
    return pos, ok


def find_positions(df):
    """Look for coordinates: soma, then root, then any point on the body. Triples of *_x/_y/_z columns work too."""
    n = len(df)
    pos = np.full((n, 3), np.nan)
    src = np.full(n, 255, dtype=np.uint8)
    used = {}
    for code, (kind, cands) in enumerate(POS_CANDIDATES.items()):
        col = pick(df.columns, cands)
        xyz, ok = None, None
        if col is not None:
            xyz, ok = parse_xyz(df[col])
            used[kind] = col
        else:
            prefixes = cands + [kind] + ([""] if kind == "pos" else [])
            for cand in prefixes:
                names = (lambda a: [a]) if cand == "" else (lambda a, c=cand: [f"{c}_{a}", f"{c}{a.upper()}"])
                cx, cy, cz = (pick(df.columns, names(a)) for a in "xyz")
                if cx and cy and cz:
                    xyz = df[[cx, cy, cz]].to_numpy(dtype=np.float64)
                    ok = np.isfinite(xyz).all(1) & ~(xyz == 0).all(1)
                    used[kind] = f"{cx},{cy},{cz}"
                    break
        if xyz is None:
            continue
        fill = ok & (src == 255)
        pos[fill] = xyz[fill]
        src[fill] = code
    return pos, src, used


def propagate_positions(pos, src, pre, post, iters=8, seed=0):
    """Neurons without coordinates get the mean of their graph neighbours (sensory somata are often outside the volume)."""
    import scipy.sparse as sp
    n = len(pos)
    missing = src == 255
    if not missing.any():
        return pos, src
    log(f"no coordinates for {missing.sum()} neurons, propagating through the graph")
    A = sp.coo_matrix((np.ones(len(pre), dtype=np.float32), (post, pre)), shape=(n, n)).tocsr()
    A = (A + A.T).tocsr()
    known = ~missing
    p = np.where(known[:, None], pos, 0.0)
    for _ in range(iters):
        k = known.astype(np.float32)
        num = A @ (p * k[:, None])
        den = A @ k
        newly = (~known) & (den > 0)
        p[newly] = num[newly] / den[newly, None]
        src[newly] = 3
        known = known | newly
        if known.all():
            break
    rest = ~known
    if rest.any():
        rng = np.random.default_rng(seed)
        c = p[known].mean(0) if known.any() else np.zeros(3)
        s = p[known].std(0) if known.any() else np.ones(3)
        p[rest] = c + rng.normal(size=(rest.sum(), 3)) * s * 0.3
        src[rest] = 4
    return p, src


def orient_positions(pos, superclass=None):
    """Centre at zero, VNC -> brain axis vertical (brain on top), brain width along x, radius ~1.
    Without a VNC label, fall back to principal axes: the widest one becomes x."""
    x = pos - pos.mean(0)
    vnc = np.array([("vnc" in str(s).lower()) for s in superclass]) if superclass is not None else np.zeros(len(x), bool)
    if 50 < vnc.sum() < 0.8 * len(x):
        up = x[~vnc].mean(0) - x[vnc].mean(0)
        up /= np.linalg.norm(up)
        rest = x[~vnc] - np.outer(x[~vnc] @ up, up)
        vals, vecs = np.linalg.eigh(np.cov(rest.T))
        side = vecs[:, np.argmax(vals)]
        side -= (side @ up) * up
        side /= np.linalg.norm(side)
    else:
        vals, vecs = np.linalg.eigh(np.cov(x.T))
        order = np.argsort(vals)[::-1]
        side, up = vecs[:, order[0]], vecs[:, order[1]]
    depth = np.cross(side, up)
    out = np.stack([x @ side, x @ up, x @ depth], 1)
    out -= (out.max(0) + out.min(0)) / 2
    r = np.percentile(np.abs(out).max(1), 99.5)
    return (out / max(r, 1e-9)).astype(np.float32)


def normalize_nt(values):
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("unk")
            continue
        s = str(v).strip().lower()
        out.append(NT_ALIASES.get(s, "unk"))
    return np.array([NT_NAMES.index(s) for s in out], dtype=np.uint8)


def load_nt(path, ids):
    df = read_feather(path)
    id_col = pick(df.columns, ID_CANDIDATES)
    if id_col is None:
        raise SystemExit(f"no id column in {path}, found: {list(df.columns)}")
    label_col = None
    str_cols = [c for c in df.columns if is_str_col(df[c])]
    for key in ["consensus", "celltype", "predicted", "nt"]:
        for c in str_cols:
            if key in c.lower():
                vals = df[c].dropna().astype(str).str.lower().unique()[:50]
                if any(v in NT_ALIASES for v in vals):
                    label_col = c
                    break
        if label_col:
            break
    df = df.drop_duplicates(id_col).set_index(id_col)
    if label_col is not None:
        log(f"transmitter: column '{label_col}'")
        nt = normalize_nt(df[label_col].reindex(ids).values)
        return nt, {"nt_id": id_col, "nt_label": label_col}
    prob_cols = {}
    long_names = {"acetylcholine": "ach", "gaba": "gaba", "glutamate": "glu", "histamine": "his",
                  "dopamine": "da", "serotonin": "5ht", "octopamine": "oa", "tyramine": "ta"}
    for name, code in long_names.items():
        for c in df.columns:
            if name in c.lower() and not is_str_col(df[c]) and code not in prob_cols:
                prob_cols[code] = c
    if not prob_cols:
        raise SystemExit(f"neither transmitter labels nor probabilities in {path}; columns: {list(df.columns)}")
    log(f"transmitter: argmax over {prob_cols}")
    codes = list(prob_cols)
    P = df[[prob_cols[c] for c in codes]].reindex(ids).to_numpy(dtype=np.float64)
    arg = np.nanargmax(np.nan_to_num(P, nan=-1.0), axis=1)
    has = np.isfinite(P).any(1)
    nt = np.array([NT_NAMES.index(codes[a]) if h else NT_NAMES.index("unk") for a, h in zip(arg, has)], dtype=np.uint8)
    return nt, {"nt_id": id_col, "nt_prob": prob_cols}


def load_edges(path, ids, min_syn):
    """Read the full segment-to-segment graph in batches and keep only edges between selected neurons."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc as ipc

    source = pa.memory_map(path, "r")
    reader = ipc.open_file(source)
    cols = reader.schema.names
    c_pre, c_post, c_w = pick(cols, PRE_CANDIDATES), pick(cols, POST_CANDIDATES), pick(cols, WEIGHT_CANDIDATES)
    if not (c_pre and c_post and c_w):
        raise SystemExit(f"no pre/post/weight in {path}: {cols}")
    log(f"edges: pre='{c_pre}', post='{c_post}', weight='{c_w}', batches {reader.num_record_batches}")
    value_set = pa.array(ids)
    pres, posts, ws = [], [], []
    total_rows = 0
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i)
        total_rows += b.num_rows
        pre_a, post_a, w_a = b.column(c_pre), b.column(c_post), b.column(c_w)
        mask = pc.and_(pc.is_in(pre_a, value_set=value_set), pc.is_in(post_a, value_set=value_set))
        if min_syn > 1:
            mask = pc.and_(mask, pc.greater_equal(w_a, min_syn))
        pres.append(pc.filter(pre_a, mask).to_numpy(zero_copy_only=False))
        posts.append(pc.filter(post_a, mask).to_numpy(zero_copy_only=False))
        ws.append(pc.filter(w_a, mask).to_numpy(zero_copy_only=False))
        if (i + 1) % 20 == 0:
            log(f"  batch {i + 1}/{reader.num_record_batches}, rows {total_rows:,}")
    pre = np.concatenate(pres).astype(np.int64)
    post = np.concatenate(posts).astype(np.int64)
    w = np.concatenate(ws).astype(np.int64)
    log(f"rows in the file {total_rows:,}, between selected neurons {len(pre):,}")
    return pre, post, w, {"pre": c_pre, "post": c_post, "weight": c_w}


def aggregate_edges(pre_i, post_i, w, n, keep_autapses):
    if not keep_autapses:
        m = pre_i != post_i
        pre_i, post_i, w = pre_i[m], post_i[m], w[m]
    key = post_i.astype(np.int64) * n + pre_i.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv, weights=w).astype(np.int64)
    post_u = (uniq // n).astype(np.int32)
    pre_u = (uniq % n).astype(np.int32)
    return pre_u, post_u, cnt.astype(np.int32)


GLOM_RE = re.compile(r"^ORN[_ ]?([A-Za-z0-9]+)")


def superclass_mask(superclass, spec):
    """Neurons whose superclass is in the comma-separated `spec` (case-insensitive)."""
    want = {s.strip().lower() for s in str(spec).split(",") if s.strip()}
    return np.array([str(s).strip().lower() in want for s in superclass], dtype=bool)


def subset_neurons(mask, pre, post, cnt):
    """Keep the neurons under `mask`: new indices, and the edges with both ends inside, renumbered."""
    idx = np.where(mask)[0]
    remap = np.full(len(mask), -1, dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    e = (remap[pre] >= 0) & (remap[post] >= 0)
    return idx, remap[pre[e]].astype(pre.dtype), remap[post[e]].astype(post.dtype), cnt[e]


def build_glomeruli(types):
    """Glomerulus label per neuron (-1 if not an ORN). A real smell is a pattern over glomeruli."""
    labels = []
    for t in types:
        m = GLOM_RE.match(str(t))
        labels.append(m.group(1) if m else None)
    names = sorted({l for l in labels if l})
    index = {nm: i for i, nm in enumerate(names)}
    code = np.array([index.get(l, -1) for l in labels], dtype=np.int32)
    return code, names


def hop_distance(pre, post, sources, n, max_hops=6):
    """Synapses from the sensory input to each neuron. 255 = unreachable."""
    import scipy.sparse as sp
    A = sp.csr_matrix((np.ones(len(pre), np.int8), (pre, post)), shape=(n, n))
    hop = np.full(n, 255, dtype=np.uint8)
    if not len(sources):
        return hop
    hop[sources] = 0
    frontier = np.asarray(sources)
    for d in range(1, max_hops + 1):
        nxt = np.unique(A.indices[np.concatenate([np.arange(A.indptr[i], A.indptr[i + 1]) for i in frontier])]) \
            if len(frontier) < 64 else np.unique(A[frontier].indices)
        nxt = nxt[hop[nxt] == 255]
        if not len(nxt):
            break
        hop[nxt] = d
        frontier = nxt
    return hop


def build_groups(superclass, types):
    groups = {}
    sc = np.array([str(s).lower() for s in superclass])
    for name, pat in SUPERCLASS_GROUPS.items():
        rx = re.compile(pat)
        groups[name] = np.where([bool(rx.search(s)) for s in sc])[0].astype(np.int32)
    ty = np.array(["" if t is None else str(t) for t in types])
    for name, pat in TYPE_GROUPS.items():
        rx = re.compile(pat)
        groups[name] = np.where([bool(rx.search(t)) for t in ty])[0].astype(np.int32)
    return groups


def synthetic(n=20000, deg=60, seed=0):
    """Fake CNS: a brain, two optic lobes and a VNC, connectivity decaying with distance. Debug only."""
    rng = np.random.default_rng(seed)
    parts = [("brain", 0.55, (0, 0.35, 0), (0.35, 0.22, 0.25)),
             ("ol_l", 0.14, (-0.7, 0.35, 0), (0.18, 0.2, 0.2)),
             ("ol_r", 0.14, (0.7, 0.35, 0), (0.18, 0.2, 0.2)),
             ("vnc", 0.17, (0, -0.55, 0.05), (0.15, 0.4, 0.12))]
    pos, region = [], []
    for k, (name, frac, c, s) in enumerate(parts):
        m = int(n * frac) if k < len(parts) - 1 else n - sum(len(p) for p in pos)
        pos.append(rng.normal(c, s, size=(m, 3)))
        region += [name] * m
    pos = np.concatenate(pos)
    region = np.array(region)
    sc = np.where(region == "vnc", "vnc_intrinsic", np.where(region == "brain", "cb_intrinsic", "ol_intrinsic")).astype(object)
    types = np.array([""] * n, dtype=object)

    def assign(mask_region, count, name, type_prefix=None):
        free = np.array([str(v).endswith("intrinsic") for v in sc]) & (types == "")
        idx = rng.choice(np.where(mask_region & free)[0], count, replace=False)
        sc[idx] = name
        if type_prefix:
            types[idx] = [f"{type_prefix}{i % 7}" for i in range(count)]
        return idx

    assign(region == "brain", int(0.03 * n), "cb_sensory")
    orn = assign(region == "brain", int(0.01 * n), "cb_sensory", "ORN_DA")
    assign(np.isin(region, ["ol_l", "ol_r"]), int(0.03 * n), "ol_sensory", "R1-6")
    assign(region == "vnc", int(0.03 * n), "vnc_sensory")
    assign(region == "brain", int(0.008 * n), "descending_neuron", "DNa")
    assign(region == "vnc", int(0.006 * n), "ascending_neuron", "AN")
    assign(region == "vnc", int(0.006 * n), "vnc_motor", "MN")
    assign(region == "brain", int(0.02 * n), "cb_intrinsic", "KCg")
    assign(region == "brain", int(0.002 * n), "cb_intrinsic", "MBON")
    assign(region == "brain", int(0.002 * n), "cb_intrinsic", "PAM")
    del orn
    from scipy.spatial import cKDTree
    tree = cKDTree(pos)
    k = deg * 2
    _, nb = tree.query(pos, k=k + 1)
    pre, post = [], []
    for i in range(n):
        cand = nb[i, 1:]
        take = rng.random(k) < 0.5
        pre.append(np.full(take.sum(), i))
        post.append(cand[take])
    long_pre = rng.integers(0, n, n * deg // 10)
    long_post = rng.integers(0, n, n * deg // 10)
    pre = np.concatenate(pre + [long_pre])
    post = np.concatenate(post + [long_post])
    w = rng.geometric(0.25, size=len(pre))
    nt = rng.choice([0, 1, 2, 4, 8], size=n, p=[0.7, 0.14, 0.1, 0.03, 0.03]).astype(np.uint8)
    return pos, sc.astype(str), types.astype(str), pre, post, w, nt


def save(out_dir, body_id, superclass, types, nt, pos, pos_src, pre, post, cnt, groups, info):
    glom, glom_names = build_glomeruli(types)
    hop = hop_distance(pre, post, groups.get("sensory", np.zeros(0, np.int64)), len(body_id))
    os.makedirs(out_dir, exist_ok=True)
    sc_names = sorted(set(superclass))
    sc_code = np.array([sc_names.index(s) for s in superclass], dtype=np.uint8)
    arrays = dict(
        body_id=body_id.astype(np.int64), pre=pre, post=post, count=cnt, nt=nt,
        nt_names=np.array(NT_NAMES), superclass=sc_code, superclass_names=np.array(sc_names),
        pos=pos.astype(np.float32), pos_source=pos_src,
        glom=glom, glom_names=np.array(glom_names if glom_names else [""]), hop=hop,
    )
    for g, idx in groups.items():
        arrays[f"grp_{g}"] = idx
    np.savez_compressed(os.path.join(out_dir, "graph.npz"), **arrays)
    n, e = len(body_id), len(pre)
    summary = dict(
        neurons=int(n), edges=int(e), synapses=int(cnt.sum()),
        expected_neurons=EXPECTED_NEURONS, expected_edges=EXPECTED_EDGES,
        superclass_counts={s: int((sc_code == i).sum()) for i, s in enumerate(sc_names)},
        nt_counts={NT_NAMES[i]: int((nt == i).sum()) for i in range(len(NT_NAMES))},
        group_counts={g: int(len(v)) for g, v in groups.items()},
        glomeruli=int(len(glom_names)), orn_neurons=int((glom >= 0).sum()),
        hops={str(d): int((hop == d).sum()) for d in range(7)} | {"unreachable": int((hop == 255).sum())},
        position_sources={k: int((pos_src == i).sum()) for i, k in enumerate(["soma", "root", "pos", "graph", "random", "synapse_mean"])},
        attribution="MaleCNS v1.0, FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research. CC BY 4.0.",
        **info,
    )
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "neurons.tsv"), "w") as f:
        f.write("idx\tbody_id\tsuperclass\ttype\tnt\n")
        for i in range(n):
            f.write(f"{i}\t{body_id[i]}\t{superclass[i]}\t{types[i]}\t{NT_NAMES[nt[i]]}\n")
    mb = os.path.getsize(os.path.join(out_dir, "graph.npz")) / 1e6
    log(f"done: {n:,} neurons, {e:,} edges, {int(cnt.sum()):,} synapses, {mb:.0f} MB -> {out_dir}")
    if info.get("source") == "malecns":
        if abs(n - EXPECTED_NEURONS) > 0.01 * EXPECTED_NEURONS:
            log(f"warning: {n:,} neurons, expected ~{EXPECTED_NEURONS:,}; check the --status / --require filter")
        if info.get("min_syn", 1) == 1 and abs(e - EXPECTED_EDGES) > 0.02 * EXPECTED_EDGES:
            log(f"warning: {e:,} edges, the release has {EXPECTED_EDGES:,}; the neuron filter can explain the gap")
    for g, v in summary["group_counts"].items():
        print(f"    {g:18s} {v:8,}")
    if glom_names:
        log(f"glomeruli: {len(glom_names)}, ORNs in them: {(glom >= 0).sum():,}")
    log("hops from the input: " + ", ".join(f"{d}: {int((hop == d).sum()):,}" for d in range(4))
        + f", unreachable: {int((hop == 255).sum()):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw", help="folder with the MaleCNS feather files")
    ap.add_argument("--out", default="graph")
    ap.add_argument("--download", action="store_true", help="download the files (~1.2 GB)")
    ap.add_argument("--inspect", action="store_true", help="print columns and frequencies, then exit")
    ap.add_argument("--synthetic", action="store_true", help="generate a fake graph for testing")
    ap.add_argument("--synthetic-n", type=int, default=20000)
    ap.add_argument("--require", default="superclass", help="column that must be non-empty for a neuron")
    ap.add_argument("--status", default="", help="comma-separated status allowlist (empty = no filter)")
    ap.add_argument("--keep-superclass", default="", help="comma-separated superclasses to keep, e.g. "
                    "cb_sensory,cb_intrinsic,visual_projection,descending_neuron,ascending_neuron for the "
                    "central brain alone (the subset ngxson/fly-llm-hf uses). Empty = the whole CNS")
    ap.add_argument("--min-syn", type=int, default=1, help="minimum synapses per edge (1 = full graph)")
    ap.add_argument("--keep-autapses", action="store_true")
    ap.add_argument("--syn-points", default="", help="syn-points-*.feather: use the mean synapse position of a "
                                                    "neuron instead of its soma, which makes the brain shape visible")
    ap.add_argument("--jitter", type=float, default=0.0, help="random offset so points do not collapse onto each other")
    args = ap.parse_args()

    if args.synthetic:
        pos, sc, types, pre, post, w, nt = synthetic(args.synthetic_n)
        n = len(sc)
        pre_u, post_u, cnt = aggregate_edges(pre, post, w, n, args.keep_autapses)
        if args.min_syn > 1:
            m = cnt >= args.min_syn
            pre_u, post_u, cnt = pre_u[m], post_u[m], cnt[m]
        if args.keep_superclass:
            m = superclass_mask(sc, args.keep_superclass)
            idx, pre_u, post_u, cnt = subset_neurons(m, pre_u, post_u, cnt)
            pos, sc, types, nt, n = pos[idx], sc[idx], types[idx], nt[idx], len(idx)
            log(f"--keep-superclass {args.keep_superclass}: {n:,} neurons, {len(pre_u):,} edges")
        groups = build_groups(sc, types)
        save(args.out, np.arange(n), sc, types, nt, orient_positions(pos, sc), np.zeros(n, np.uint8),
             pre_u, post_u, cnt, groups, {"source": "synthetic", "min_syn": args.min_syn,
                                          "keep_superclass": args.keep_superclass})
        return

    if args.download:
        download(args.raw)
    paths = {k: os.path.join(args.raw, v) for k, v in FILES.items()}
    for k, p in paths.items():
        if not os.path.exists(p):
            raise SystemExit(f"missing file {p}; run with --download or put the files in {args.raw}")

    log("reading annotations")
    ann = read_feather(paths["ann"])
    c_id = pick(ann.columns, ID_CANDIDATES)
    c_sc = pick(ann.columns, SUPERCLASS_CANDIDATES)
    c_cl = pick(ann.columns, CLASS_CANDIDATES)
    c_ty = pick(ann.columns, TYPE_CANDIDATES)
    c_st = pick(ann.columns, STATUS_CANDIDATES)
    if args.inspect:
        import pyarrow.ipc as ipc, pyarrow as pa
        e_cols = ipc.open_file(pa.memory_map(paths["w"], "r")).schema.names
        nt_cols = list(read_feather(paths["nt"]).columns)
        print(f"rows in the annotations: {len(ann):,}")
        print("annotation columns:", list(ann.columns))
        print("transmitter columns:", nt_cols)
        print("edge columns:", e_cols)
        print("\npicked --- this is what the build will use:")
        print(f"  id         {c_id}")
        print(f"  superclass {c_sc}   (falls back to class '{c_cl}' when empty)")
        print(f"  type       {c_ty}")
        print(f"  status     {c_st}")
        print(f"  nt id      {pick(nt_cols, ID_CANDIDATES)}")
        print(f"  edges      pre={pick(e_cols, PRE_CANDIDATES)}, post={pick(e_cols, POST_CANDIDATES)}, "
              f"weight={pick(e_cols, WEIGHT_CANDIDATES)}")
        print(f"  positions  {[c for c in POS_CANDIDATES if pick(ann.columns, POS_CANDIDATES[c])]}")
        for c in ann.columns:
            try:
                if is_str_col(ann[c]) and ann[c].nunique() < 80:
                    print(f"\n{c}:\n{ann[c].value_counts(dropna=False).head(40).to_string()}")
            except TypeError:       # cells holding arrays (positions and the like) are not hashable
                continue
        return

    if c_id is None:
        raise SystemExit(f"no id column in the annotations: {list(ann.columns)}")
    log(f"columns: id='{c_id}', superclass='{c_sc}', class='{c_cl}', type='{c_ty}', status='{c_st}'")

    keep = np.ones(len(ann), dtype=bool)
    req = pick(ann.columns, [args.require]) if args.require else None
    if req is not None:
        keep &= ann[req].notna().to_numpy() & (ann[req].astype(str).str.strip() != "").to_numpy()
    if args.status and c_st:
        allowed = {s.strip() for s in args.status.split(",")}
        keep &= ann[c_st].isin(allowed).to_numpy()
    if args.keep_superclass:
        col = c_sc or c_cl
        if col is None:
            raise SystemExit("--keep-superclass needs a superclass or class column in the annotations")
        keep &= superclass_mask(ann[col].fillna("unknown").astype(str).to_numpy(), args.keep_superclass)
    ann = ann[keep].drop_duplicates(c_id).sort_values(c_id).reset_index(drop=True)
    ids = ann[c_id].to_numpy().astype(np.int64)
    n = len(ids)
    expected = (f" (expected ~{EXPECTED_NEURONS:,})" if not args.keep_superclass
                else f" (subset: {args.keep_superclass})")
    log(f"neurons after filtering: {n:,}{expected}")

    superclass = ann[c_sc].fillna("unknown").astype(str).to_numpy() if c_sc else np.array(["unknown"] * n)
    types = ann[c_ty].fillna("").astype(str).to_numpy() if c_ty else np.array([""] * n)
    if c_sc is None and c_cl is not None:
        superclass = ann[c_cl].fillna("unknown").astype(str).to_numpy()

    nt, nt_info = load_nt(paths["nt"], ids)
    pre, post, w, e_info = load_edges(paths["w"], ids, args.min_syn)
    pre_i, post_i = np.searchsorted(ids, pre), np.searchsorted(ids, post)
    pre_u, post_u, cnt = aggregate_edges(pre_i, post_i, w, n, args.keep_autapses)

    pos, pos_src, pos_used = find_positions(ann)
    if args.syn_points:
        syn_pos, syn_ok = synapse_positions(args.syn_points, ids)
        pos[syn_ok] = syn_pos[syn_ok]
        pos_src[syn_ok] = 5
        pos_used["synapse_mean"] = os.path.basename(args.syn_points)
    pos, pos_src = propagate_positions(pos, pos_src, pre_u, post_u)
    pos = orient_positions(pos, superclass)
    if args.jitter > 0:
        pos = pos + np.random.default_rng(0).normal(scale=args.jitter, size=pos.shape).astype(np.float32)
    groups = build_groups(superclass, types)
    info = dict(source="malecns", min_syn=args.min_syn, keep_superclass=args.keep_superclass, columns=dict(
        id=c_id, superclass=c_sc, cls=c_cl, type=c_ty, status=c_st, positions=pos_used, **nt_info, edges=e_info))
    save(args.out, ids, superclass, types, nt, pos, pos_src, pre_u, post_u, cnt, groups, info)


if __name__ == "__main__":
    sys.exit(main())
