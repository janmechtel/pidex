#!/usr/bin/env python3
"""
build_site_data.py — Produce compact JSON for the GitHub Pages frontend
Output:
  docs/data/packages.json      (compact, one record per package)
  docs/data/pidex-full.json.gz (full dataset download)
"""

import json, re, gzip
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA = Path(__file__).parent.parent / "data"
DOCS = Path(__file__).parent.parent / "docs" / "data"
DOCS.mkdir(parents=True, exist_ok=True)

# ── Load enriched ────────────────────────────────────────────────────────────
print("Loading enriched packages...")
records = [json.loads(f.read_text()) for f in (DATA / "enriched").glob("*.json")]
raw     = json.loads((DATA / "raw" / "packages.json").read_text())
pub_map = {p["name"]: p.get("publisher") for p in raw}

CI_BOTS = {'GitHub Actions', 'github-actions[bot]', 'semantic-release-bot', 'npm-publish'}

def best_author(r):
    """GitHub owner from repo URL → npm publisher (skip CI bots) → author.name"""
    import re as _re
    repo = r.get('repository') or {}
    url  = repo.get('url','') if isinstance(repo, dict) else str(repo)
    m = _re.search(r'github\.com[:/]([^/]+)/', url)
    if m: return m.group(1)
    pub = pub_map.get(r.get('name',''))
    if pub and pub not in CI_BOTS: return pub
    a = r.get('author')
    if isinstance(a, dict): a = a.get('name')
    return a if a and a not in CI_BOTS else None

MULTI_KW   = {'claude-code','codex','opencode','gemini-cli','windsurf','copilot','cline','kiro','qoder'}
MULTI_DESC = ['claude code','gemini cli','windsurf','opencode',' codex ','copilot cli','cursor ide']

def is_multi_harness(r):
    kws = set(r.get('keywords') or [])
    if kws & MULTI_KW: return True
    desc = (r.get('description') or '').lower()
    top  = (r.get('readme') or '')[:400].lower()
    hits = sum(1 for t in MULTI_DESC if t in desc or t in top)
    return hits >= 2
def extract_slug(repo_field):
    url = repo_field.get("url","") if isinstance(repo_field,dict) else str(repo_field or "")
    m = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', url)
    return f"{m.group(1)}__{m.group(2).rstrip('.git')}" if m else None

def infer_type(keywords):
    for k in (keywords or []):
        if k in ("pi-extension","pi-skill","pi-theme","pi-prompt"):
            return k.replace("pi-","")
    return "unknown"

def classify_trend(trend):
    if not trend or len(trend) < 6: return "new"
    vals = [t["downloads"] for t in trend]
    first = sum(vals[:len(vals)//2]) / (len(vals)//2)
    second = sum(vals[len(vals)//2:]) / (len(vals) - len(vals)//2)
    if first == 0: return "new"
    r = second / first
    if r > 1.5: return "growing"
    if r < 0.5: return "declining"
    return "stable"

# ── Join GitHub data ──────────────────────────────────────────────────────────
gh_dir = DATA / "github"
gh_map = {}
for r in records:
    slug = extract_slug(r.get("repository") or {})
    if slug and (gh_dir / f"{slug}.json").exists():
        gh = json.loads((gh_dir / f"{slug}.json").read_text())
        gh_map[r["name"]] = {
            "stars":     gh.get("stargazers_count"),
            "forks":     gh.get("forks_count"),
            "is_fork":   gh.get("fork", False),
            "pushed_at": gh.get("pushed_at"),
        }

# ── Hybrid TF-IDF similarity + 2-hop graph propagation ──────────────────────
TOP_N      = 20      # neighbours per package in final output
SIM_THRESH = 0.15    # minimum score to include
HOP_DECAY  = 0.7     # indirect score = score_AB * score_BC * HOP_DECAY
W_SPECIFIC = 0.55    # weight for name+keywords signal
W_PROSE    = 0.45    # weight for description+readme signal

import scipy.sparse as _sp

# Domain-generic words that appear in nearly every pi package — strip them so
# they don't create false connections (e.g. "search" linking fff to websearch)
PI_STOPS = {
    "pi","extension","skill","theme","prompt","package","plugin","tool","tools",
    "agent","coding","install","support","use","using","provide","allow","enable",
    "work","help","simple","easy","based","powered","like","add","get","set",
    "new","make","build","run","just","also","via","per","let","need","want",
    "search","fetch","query","request","response","api","url","http","https",
}

names = [r["name"] for r in records]
name_idx = {n: i for i, n in enumerate(names)}

def name_tokens(n):
    """Turn a package name like @scope/pi-foo-bar into 'foo bar'."""
    import re as _re
    n = n.split("/")[-1]                     # drop scope
    n = _re.sub(r"^pi[-_]?|[-_]?pi$", "", n) # strip leading/trailing 'pi'
    return _re.sub(r"[-_]", " ", n).strip()

# Signal A: name tokens + keywords (high specificity)
texts_a = [
    name_tokens(r["name"]) + " " +
    " ".join(r.get("keywords") or [])
    for r in records
]

# Signal B: description + readme prose (broader context, domain stops removed)
texts_b = [
    (r.get("description") or "") + " " + (r.get("readme") or "")[:400]
    for r in records
]

print("Computing hybrid TF-IDF similarity...")
_stop_a = list((TfidfVectorizer(stop_words="english").get_stop_words() or set()) | PI_STOPS)
_stop_b = list((TfidfVectorizer(stop_words="english").get_stop_words() or set()) | PI_STOPS)

vec_a = TfidfVectorizer(max_features=3000, stop_words=_stop_a, ngram_range=(1,2), min_df=2)
vec_b = TfidfVectorizer(max_features=4000, stop_words=_stop_b, ngram_range=(1,2), min_df=2)
tfidf_a = vec_a.fit_transform(texts_a)
tfidf_b = vec_b.fit_transform(texts_b)

# Keep tfidf alias pointing at prose matrix (used by notebook cluster export)
vec   = vec_b
tfidf = tfidf_b

# Pass 1: build direct top-N sparse matrix from hybrid score
BATCH = 500
rows_l, cols_l, data_l = [], [], []
print(f"Pass 1 (direct hybrid) in {len(records)//BATCH+1} batches...")
for start in range(0, len(records), BATCH):
    end = min(start + BATCH, len(records))
    sim_a = cosine_similarity(tfidf_a[start:end], tfidf_a).astype(np.float32)
    sim_b = cosine_similarity(tfidf_b[start:end], tfidf_b).astype(np.float32)
    batch_sim = W_SPECIFIC * sim_a + W_PROSE * sim_b
    for i, row in enumerate(batch_sim):
        row[start+i] = 0
        top_idx = np.argsort(row)[::-1][:TOP_N]
        for j in top_idx:
            if row[j] > SIM_THRESH:
                rows_l.append(start+i); cols_l.append(j); data_l.append(float(row[j]))
    print(f"  {end}/{len(records)}")

N = len(names)
S = _sp.csr_matrix((data_l, (rows_l, cols_l)), shape=(N, N))

# Pass 2: 2-hop propagation  S2[i,j] = sum_k( S[i,k] * S[k,j] ) * decay
print("Pass 2 (2-hop propagation)...")
S2 = (S @ S) * HOP_DECAY
S2 = S2.tocsr()
S2.setdiag(0); S2.eliminate_zeros()

# Combine: keep the higher of direct or indirect score
S_combined = S.maximum(S2)
S_combined.setdiag(0); S_combined.eliminate_zeros()

# Build final similar_map from combined scores
print("Building similar_map...")
similar_map = {}
for i, name in enumerate(names):
    row = np.asarray(S_combined[i].todense()).flatten()
    row[i] = 0
    np.clip(row, 0, 1, out=row)  # 2-hop sums can exceed 1.0
    top_idx = np.argsort(row)[::-1][:TOP_N]
    similar_map[name] = [
        {"name": names[j], "score": round(float(row[j]), 3)}
        for j in top_idx if row[j] > SIM_THRESH
    ]

# ── 2D layout coordinates (SVD on prose TF-IDF) ─────────────────────────────
print("Computing 2D layout (SVD)...")
from sklearn.decomposition import TruncatedSVD as _SVD
_svd = _SVD(n_components=2, random_state=42)
_coords = _svd.fit_transform(tfidf_b)
# Normalise to [-1, 1] for consistent scale across rebuilds
for _col in range(2):
    _lo, _hi = _coords[:, _col].min(), _coords[:, _col].max()
    _coords[:, _col] = 2 * (_coords[:, _col] - _lo) / (_hi - _lo + 1e-9) - 1
coords_map = {names[i]: (_coords[i, 0], _coords[i, 1]) for i in range(len(names))}
print(f"  done ({len(coords_map)} packages)")

# ── Build compact records ─────────────────────────────────────────────────────
print("Building compact records...")
compact = []
for r in records:
    name  = r["name"]
    trend = r.get("download_trend") or []
    gh    = gh_map.get(name, {})

    # Compact trend: just weekly download numbers (not dates) to save space
    trend_vals = [t["downloads"] for t in trend[-26:]]  # last 26 weeks

    compact.append({
        "n":  name,
        "d":  (r.get("description") or "")[:200],
        "t":  infer_type(r.get("keywords")),
        "p":  best_author(r),
        "dl": r.get("downloads_last_week"),
        "tr": trend_vals,
        "tt": classify_trend(trend),
        "st": gh.get("stars"),
        "fk": gh.get("is_fork", False),
        "mh": is_multi_harness(r),
        "pa": gh.get("pushed_at"),
        "si": similar_map.get(name, []),
        "kw": [k for k in (r.get("keywords") or [])
               if k not in ("pi-package","pi-extension","pi-skill","pi-theme","pi-prompt")][:8],
        "rm": (r.get("readme") or "")[:200].replace("\n"," "),
        "v":  r.get("latest_version"),
        "dt": r.get("time", {}).get("modified"),
        "dc": r.get("time", {}).get("created"),
        "gh": (r.get("repository") or {}).get("url","") if isinstance(r.get("repository"),dict) else "",
        "x":  round(float(coords_map[name][0]), 4) if name in coords_map else 0,
        "y":  round(float(coords_map[name][1]), 4) if name in coords_map else 0,
    })

out_path = DOCS / "packages.json"
out_path.write_text(json.dumps(compact, separators=(",",":")))
size_kb = out_path.stat().st_size // 1024
print(f"packages.json → {size_kb}KB  ({len(compact)} packages)")

# ── All-packages word frequencies (doc frequency, description + readme) ───────
print("Computing all_wordfreqs.json...")
import re as _re
_STOPS = {
    'this','that','with','from','have','will','your','their','they','been','also',
    'which','more','some','into','such','than','just','each','about','over','then',
    'when','where','there','these','those','what','would','could','should','other',
    'after','while','being','using','used','uses','adds','make','makes','made',
    'gets','lets','runs','take','takes','give','gives','help','helps','allow',
    'allows','build','built','based','simple','simply','easy','small','tool',
    'tools','work','works','support','provide','provides','include','includes',
    'feature','features','package','packages','extension','extensions','coding',
    'agent','install','http','https','npm','readme',
}
_doc_freq = {}
for r in records:
    text = ((r.get('description') or '') + ' ' + (r.get('readme') or '')[:200]).lower()
    words = set(_re.sub(r'[^a-z ]+', ' ', text).split())
    for w in words:
        if len(w) > 3 and w not in _STOPS:
            _doc_freq[w] = _doc_freq.get(w, 0) + 1
_all_wf = {w: v for w, v in _doc_freq.items() if v >= 2}
(DOCS / 'all_wordfreqs.json').write_text(json.dumps(_all_wf, separators=(',',':')))
print(f"all_wordfreqs.json → {len(_all_wf)} words")

# ── Full dataset download (gzipped) ──────────────────────────────────────────
print("Writing full dataset download...")
full_records = []
for r in records:
    name = r["name"]
    full_records.append({**r, "github": gh_map.get(name,{}), "similar": similar_map.get(name,[])})

full_path = DOCS / "pidex-full.json.gz"
with gzip.open(full_path, "wt", encoding="utf-8") as f:
    json.dump(full_records, f)
print(f"pidex-full.json.gz → {full_path.stat().st_size//1024}KB")
print("Done.")
