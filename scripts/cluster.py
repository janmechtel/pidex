#!/usr/bin/env python3
"""
cluster.py — Embed packages with Bedrock Cohere, cluster with KMeans,
             auto-name clusters, write docs/data/clusters*.json
"""
import json, time
from pathlib import Path
import numpy as np
import boto3
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DATA  = Path(__file__).parent.parent / "data"
DOCS  = Path(__file__).parent.parent / "docs" / "data"
DOCS.mkdir(parents=True, exist_ok=True)

BATCH_SIZE   = 96        # Cohere max per request
CACHE_FILE   = DATA / "embeddings_cluster.npz"
BEDROCK_REGION = "us-east-1"

# Domain-noise words to suppress in auto-naming
PI_STOPS = {
    "pi","extension","skill","theme","prompt","package","plugin","tool","tools",
    "agent","coding","install","support","use","using","provide","allow","enable",
    "work","help","simple","easy","based","powered","like","add","get","set",
    "new","make","build","run","just","also","via","per","let","need","want",
    "search","fetch","query","request","response","api","url","http","https",
    "feature","features","available","current","version","latest","update",
    "project","code","file","files","text","output","input","user","users",
}

# ── Load packages ─────────────────────────────────────────────────────────────
print("Loading packages...")
records = [json.loads(f.read_text()) for f in (DATA / "enriched").glob("*.json")]
names   = [r["name"] for r in records]
# Richer text: description + up to 1000 chars of readme
texts   = [
    ((r.get("description") or "") + " " + (r.get("readme") or "")[:1000]).strip()
    for r in records
]
print(f"  {len(records)} packages loaded")

# ── Cohere embeddings (cached) ────────────────────────────────────────────────
cached_names, cached_vecs = [], []
if CACHE_FILE.exists():
    data_np = np.load(CACHE_FILE, allow_pickle=True)
    cached_names = list(data_np["names"])
    cached_vecs  = list(data_np["vecs"])
    print(f"Loaded {len(cached_names)} cached cluster embeddings")

cache_map = dict(zip(cached_names, cached_vecs))
need = [(n, t) for n, t in zip(names, texts) if n not in cache_map]
print(f"Need to embed {len(need)} packages via Bedrock Cohere...")

if need:
    bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    for i in range(0, len(need), BATCH_SIZE):
        batch = need[i:i + BATCH_SIZE]
        batch_names, batch_texts = zip(*batch)
        for attempt in range(3):
            try:
                resp = bedrock.invoke_model(
                    modelId="cohere.embed-english-v3",
                    body=json.dumps({"texts": list(batch_texts), "input_type": "search_document"}),
                    contentType="application/json",
                )
                out = json.loads(resp["body"].read())
                for name, vec in zip(batch_names, out["embeddings"]):
                    cache_map[name] = np.array(vec, dtype=np.float32)
                print(f"  embedded {min(i+BATCH_SIZE, len(need))}/{len(need)}")
                break
            except Exception as e:
                if attempt == 2: raise
                print(f"  retry {attempt+1}: {e}")
                time.sleep(2 ** attempt)
    all_names = list(cache_map.keys())
    all_vecs  = np.array([cache_map[n] for n in all_names], dtype=np.float32)
    np.savez_compressed(CACHE_FILE, names=all_names, vecs=all_vecs)
    print(f"Cache saved → {CACHE_FILE} ({CACHE_FILE.stat().st_size//1024}KB)")

# Align to records order
vec_matrix = np.array([cache_map[n] for n in names], dtype=np.float32)
print(f"Embedding matrix: {vec_matrix.shape}")

# ── KMeans clustering ─────────────────────────────────────────────────────────
N_CLUSTERS   = 24          # overfit slightly then prune low-cohesion clusters
COHESION_MIN = 0.52        # discard clusters with avg intra-similarity below this

print(f"Running KMeans(k={N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20, max_iter=500)
labels = km.fit_predict(vec_matrix)
print(f"  inertia: {km.inertia_:.0f}")

# ── Prune low-cohesion clusters ───────────────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
print("Computing intra-cluster cohesion...")
cohesion = {}
for cid in range(N_CLUSTERS):
    idx = np.where(labels == cid)[0]
    sample = idx[:80]   # cap at 80 to keep it fast
    if len(sample) < 2:
        cohesion[cid] = 0.0
        continue
    s = _cos_sim(vec_matrix[sample])
    np.fill_diagonal(s, 0)
    cohesion[cid] = float(s.sum() / (len(sample) * (len(sample) - 1)))

good_clusters = {cid for cid, v in cohesion.items() if v >= COHESION_MIN}
print(f"  keeping {len(good_clusters)}/{N_CLUSTERS} clusters "
      f"(cohesion >= {COHESION_MIN})")
for cid in sorted(range(N_CLUSTERS), key=lambda c: cohesion[c], reverse=True):
    mark = "✓" if cid in good_clusters else "✗"
    print(f"  {mark} C{cid:02d} ({(labels==cid).sum():4d} pkgs)  cohesion={cohesion[cid]:.3f}")

# Remap: packages in discarded clusters → label -1
labels = np.where(np.isin(labels, list(good_clusters)), labels, -1)
# Re-index good clusters to 0..N-1
sorted_good = sorted(good_clusters)
remap = {old: new for new, old in enumerate(sorted_good)}
labels = np.array([remap[l] if l >= 0 else -1 for l in labels])
N_GOOD = len(sorted_good)
print(f"  {(labels >= 0).sum()} packages in {N_GOOD} clusters, "
      f"{(labels == -1).sum()} unclustered")

# ── Auto-name clusters via relative TF-IDF ────────────────────────────────────
print("Auto-naming clusters...")
stop_words = list(
    (TfidfVectorizer(stop_words="english").get_stop_words() or set()) | PI_STOPS
)
vec_tf = TfidfVectorizer(max_features=8000, stop_words=stop_words,
                          ngram_range=(1,1), min_df=5,
                          token_pattern=r"[a-zA-Z][a-zA-Z]{3,}")
tfidf = vec_tf.fit_transform(texts)
fn = vec_tf.get_feature_names_out()
global_mean = np.asarray(tfidf.mean(axis=0)).flatten()

from collections import Counter as _Counter

EXTRA_STOPS = {
    'extension','package','coding','support','agent','based','using','provides',
    'allows','simple','easily','within','directly','without','through','custom',
    'users','default','manage','access','create','content','system','across',
    'messages','integrate','integration','install','installed','adding','added',
    'lightweight','configurable','designed','built','available','interface',
    'readme','install','release','badge','english','align','center','latest','build',
    'github','httpspidev','pihttps','httpspi','pidev','pipackage',
    'following','below','above','usage','example','examples','documentation',
    'right','left','first','second','third','every','when','with',
}

cluster_names = {}
cluster_top_words = {}
for cid in range(N_GOOD):
    mask = (labels == cid)
    # Count words in descriptions of packages in this cluster
    word_counts = _Counter()
    n_pkgs = 0
    for i, is_in in enumerate(mask):
        if not is_in: continue
        n_pkgs += 1
        import re as _re
        _readme = _re.sub(r'<[^>]+>|https?://\S+|\[.*?\]\(.*?\)|#+|[\|`*_]', ' ', (records[i].get("readme") or "")[:400])
        desc = ((records[i].get("description") or "") + " " + _readme).lower()
        seen = set()
        for w in desc.split():
            w = "".join(c for c in w if c.isalpha())
            if len(w) >= 5 and w not in PI_STOPS and w not in EXTRA_STOPS and w not in seen:
                word_counts[w] += 1
                seen.add(w)
    # Words that appear in >=15% of cluster packages
    threshold = max(2, int(n_pkgs * 0.07))
    top_words = [w for w, cnt in word_counts.most_common(40)
                 if cnt >= threshold and w.isascii() and w.islower()][:8]
    cluster_top_words[cid] = top_words
    name_words = top_words[:3]
    if name_words:
        cluster_names[cid] = " / ".join(w.title() for w in name_words)
    else:
        # Fallback: name after the highest-download package in cluster
        idxs = [i for i, m in enumerate(mask) if m]
        top_i = max(idxs, key=lambda i: records[i].get("downloads_last_week") or 0)
        fallback = names[top_i].split("/")[-1].replace("pi-","").replace("-"," ").title()
        cluster_names[cid] = fallback or f"Cluster {cid}"

print("Cluster names:")
for cid in range(N_GOOD):
    mask = labels == cid
    pkgs_in = [names[i] for i in range(len(names)) if mask[i]]
    print(f"  C{cid:02d} ({mask.sum():4d} pkgs)  {cluster_names[cid]}")
    print(f"        words: {', '.join(cluster_top_words[cid][:6])}")

# ── Download stats for pattern classification ─────────────────────────────────
raw = json.loads((DATA / "raw" / "packages.json").read_text())
dl_map = {p["name"]: p.get("downloads_last_week") or 0 for p in raw}

def classify_pattern(dls):
    total = sum(dls)
    if total == 0: return "long-tail"
    mx = max(dls)
    r = mx / total
    return "blockbuster" if r > 0.6 else ("contested" if r > 0.35 else "long-tail")

# ── Write output files ────────────────────────────────────────────────────────
# 1. clusters.json
clusters_out = [
    {"name": names[i], "cluster_id": int(labels[i]), "cluster_name": cluster_names[labels[i]] if labels[i] >= 0 else None}
    for i in range(len(names))
]
(DOCS / "clusters.json").write_text(json.dumps(clusters_out, separators=(",",":")))
print(f"clusters.json → {len(clusters_out)} entries")

# 2. cluster_meta.json
meta_out = []
for cid in range(N_GOOD):
    mask = labels == cid
    cluster_pkgs = [names[i] for i in range(len(names)) if mask[i]]
    dls = [dl_map.get(n, 0) for n in cluster_pkgs]
    meta_out.append({
        "id":       cid,
        "name":     cluster_names[cid],
        "pkgs":     int(mask.sum()),
        "pattern":  classify_pattern(dls),
        "total_dl": int(sum(dls)),
    })
meta_out.sort(key=lambda x: -x["total_dl"])
(DOCS / "cluster_meta.json").write_text(json.dumps(meta_out, separators=(",",":")))
print(f"cluster_meta.json → {len(meta_out)} clusters")

# 3. cluster_wordfreqs.json (for word clouds)
cluster_freqs = {}
for cid in range(N_GOOD):
    mask = (labels == cid)
    cluster_mean = np.asarray(tfidf[mask].mean(axis=0)).flatten()
    ratio = np.where(global_mean > 1e-9,
                     cluster_mean / (global_mean + 1e-9),
                     cluster_mean * 100)
    top_idx = ratio.argsort()[::-1][:300]
    freqs = {}
    for j in top_idx:
        word = fn[j]
        if " " in word: continue
        if word in PI_STOPS: continue
        if len(word) <= 3: continue
        score = float(ratio[j])
        if score < 1.2: break
        freqs[word] = round(score, 3)
        if len(freqs) >= 80: break
    cluster_freqs[str(cid)] = freqs

(DOCS / "cluster_wordfreqs.json").write_text(json.dumps(cluster_freqs, separators=(",",":")))
print(f"cluster_wordfreqs.json → {N_GOOD} clusters")
print("Done.")
