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

# ── Compute TF-IDF similarity → top-5 similar packages ───────────────────────
print("Computing TF-IDF similarity...")
texts = [(r.get("description") or "") + " " + (r.get("readme") or "")[:400]
         for r in records]
names = [r["name"] for r in records]

vec   = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1,2))
tfidf = vec.fit_transform(texts)

# Process in batches to avoid OOM on 4k x 4k matrix
BATCH = 500
similar_map = {}
print(f"Computing similarity in {len(records)//BATCH+1} batches...")
for start in range(0, len(records), BATCH):
    end = min(start + BATCH, len(records))
    batch_sim = cosine_similarity(tfidf[start:end], tfidf).astype(np.float32)
    for i, row in enumerate(batch_sim):
        row[start+i] = 0  # zero self
        top_idx = np.argsort(row)[::-1][:5]
        similar_map[names[start+i]] = [
            {"name": names[j], "score": round(float(row[j]), 3)}
            for j in top_idx if row[j] > 0.15
        ]
    print(f"  {end}/{len(records)}")

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
        "p":  pub_map.get(name),
        "dl": r.get("downloads_last_week"),
        "tr": trend_vals,
        "tt": classify_trend(trend),
        "st": gh.get("stars"),
        "fk": gh.get("is_fork", False),
        "pa": gh.get("pushed_at"),
        "si": similar_map.get(name, []),
        "kw": [k for k in (r.get("keywords") or [])
               if k not in ("pi-package","pi-extension","pi-skill","pi-theme","pi-prompt")][:8],
        "v":  r.get("latest_version"),
        "dt": r.get("time", {}).get("modified"),
        "gh": (r.get("repository") or {}).get("url","") if isinstance(r.get("repository"),dict) else "",
    })

out_path = DOCS / "packages.json"
out_path.write_text(json.dumps(compact, separators=(",",":")))
size_kb = out_path.stat().st_size // 1024
print(f"packages.json → {size_kb}KB  ({len(compact)} packages)")

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
