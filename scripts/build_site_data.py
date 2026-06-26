#!/usr/bin/env python3
"""
build_site_data.py — Produce compact JSON artifacts for the static frontend
Output:
  docs/data/packages.json       (legacy compact array, one record per package)
  docs/data/packages-db.json    (client-side filtering database + indexes)
  docs/data/pidex-full.json.gz  (full dataset download)
"""

import gzip
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = Path(__file__).parent.parent / "data"
DOCS = Path(__file__).parent.parent / "docs" / "data"
DOCS.mkdir(parents=True, exist_ok=True)

PI_TYPE_KEYWORDS = ("pi-extension", "pi-skill", "pi-theme", "pi-prompt")


def extract_slug(repo_field):
    url = repo_field.get("url", "") if isinstance(repo_field, dict) else str(repo_field or "")
    m = re.search(r"github\.com[:/]([^/]+)/([^/.]+)", url)
    return f"{m.group(1)}__{m.group(2).rstrip('.git')}" if m else None


def infer_type(keywords):
    for k in (keywords or []):
        if k in PI_TYPE_KEYWORDS:
            return k.replace("pi-", "")
    return "unknown"


def classify_trend(trend):
    if not trend or len(trend) < 6:
        return "new"
    vals = [t["downloads"] for t in trend]
    first = sum(vals[: len(vals) // 2]) / (len(vals) // 2)
    second = sum(vals[len(vals) // 2 :]) / (len(vals) - len(vals) // 2)
    if first == 0:
        return "new"
    r = second / first
    if r > 1.5:
        return "growing"
    if r < 0.5:
        return "declining"
    return "stable"


def normalize_token(token):
    t = token.strip().lower()
    if len(t) < 2:
        return None
    if len(t) > 32:
        return None
    return t


def tokenize_record(record):
    blob = " ".join(
        [
            record["n"],
            record.get("d") or "",
            " ".join(record.get("kw") or []),
            record.get("p") or "",
        ]
    ).lower()
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9-_.+/]*", blob)
    tokens = set()
    for raw in raw_tokens:
        normalized = normalize_token(raw)
        if normalized:
            tokens.add(normalized)
            if "/" in normalized:
                for part in normalized.split("/"):
                    p = normalize_token(part)
                    if p:
                        tokens.add(p)
    return sorted(tokens)


def index_records(records):
    by_type = defaultdict(list)
    by_trend = defaultdict(list)
    by_publisher = defaultdict(list)
    by_fork = {"fork": [], "non_fork": []}
    by_token = defaultdict(list)

    has_downloads = []
    has_stars = []
    has_similarity = []

    for i, rec in enumerate(records):
        by_type[rec["t"]].append(i)
        by_trend[rec["tt"]].append(i)
        by_publisher[(rec.get("p") or "")].append(i)
        (by_fork["fork"] if rec.get("fk") else by_fork["non_fork"]).append(i)

        if rec.get("dl") is not None:
            has_downloads.append(i)
        if rec.get("st") is not None:
            has_stars.append(i)
        if rec.get("si"):
            has_similarity.append(i)

        for token in tokenize_record(rec):
            by_token[token].append(i)

    def sorted_map_lists(mapping):
        return {k: sorted(v) for k, v in sorted(mapping.items())}

    sort_indexes = {
        "dl": sorted(range(len(records)), key=lambda i: (records[i].get("dl") or 0), reverse=True),
        "st": sorted(range(len(records)), key=lambda i: (records[i].get("st") or 0), reverse=True),
        "dt": sorted(range(len(records)), key=lambda i: (records[i].get("dt") or ""), reverse=True),
        "n": sorted(range(len(records)), key=lambda i: records[i]["n"].lower()),
        "tt-g": sorted(
            range(len(records)),
            key=lambda i: ({"growing": 0, "stable": 1, "new": 2, "declining": 3}.get(records[i].get("tt"), 9), records[i]["n"].lower()),
        ),
    }

    return {
        "byType": sorted_map_lists(by_type),
        "byTrend": sorted_map_lists(by_trend),
        "byPublisher": sorted_map_lists(by_publisher),
        "byFork": by_fork,
        "byToken": sorted_map_lists(by_token),
        "has": {
            "downloads": sorted(has_downloads),
            "stars": sorted(has_stars),
            "similar": sorted(has_similarity),
        },
        "sort": sort_indexes,
    }


print("Loading enriched packages...")
records = [json.loads(f.read_text()) for f in (DATA / "enriched").glob("*.json")]
records.sort(key=lambda r: r["name"].lower())
raw = json.loads((DATA / "raw" / "packages.json").read_text())
pub_map = {p["name"]: p.get("publisher") for p in raw}

print("Joining GitHub metadata...")
gh_dir = DATA / "github"
gh_map = {}
for r in records:
    slug = extract_slug(r.get("repository") or {})
    if slug and (gh_dir / f"{slug}.json").exists():
        gh = json.loads((gh_dir / f"{slug}.json").read_text())
        gh_map[r["name"]] = {
            "stars": gh.get("stargazers_count"),
            "forks": gh.get("forks_count"),
            "is_fork": gh.get("fork", False),
            "pushed_at": gh.get("pushed_at"),
        }

print("Computing TF-IDF similarity...")
texts = [(r.get("description") or "") + " " + (r.get("readme") or "")[:400] for r in records]
names = [r["name"] for r in records]

vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
tfidf = vec.fit_transform(texts)

batch = 500
similar_map = {}
print(f"Computing similarity in {len(records) // batch + 1} batches...")
for start in range(0, len(records), batch):
    end = min(start + batch, len(records))
    batch_sim = cosine_similarity(tfidf[start:end], tfidf).astype(np.float32)
    for i, row in enumerate(batch_sim):
        row[start + i] = 0
        top_idx = np.argsort(row)[::-1][:5]
        similar_map[names[start + i]] = [
            {"name": names[j], "score": round(float(row[j]), 3)} for j in top_idx if row[j] > 0.15
        ]
    print(f"  {end}/{len(records)}")

print("Building compact records...")
compact = []
for r in records:
    name = r["name"]
    trend = r.get("download_trend") or []
    gh = gh_map.get(name, {})

    compact.append(
        {
            "n": name,
            "d": (r.get("description") or "")[:200],
            "t": infer_type(r.get("keywords")),
            "p": pub_map.get(name),
            "dl": r.get("downloads_last_week"),
            "tr": [t["downloads"] for t in trend[-26:]],
            "tt": classify_trend(trend),
            "st": gh.get("stars"),
            "fk": gh.get("is_fork", False),
            "pa": gh.get("pushed_at"),
            "si": similar_map.get(name, []),
            "kw": [k for k in (r.get("keywords") or []) if k not in ("pi-package", *PI_TYPE_KEYWORDS)][:8],
            "v": r.get("latest_version"),
            "dt": r.get("time", {}).get("modified"),
            "gh": (r.get("repository") or {}).get("url", "") if isinstance(r.get("repository"), dict) else "",
        }
    )

out_path = DOCS / "packages.json"
out_path.write_text(json.dumps(compact, separators=(",", ":")))
print(f"packages.json → {out_path.stat().st_size // 1024}KB  ({len(compact)} packages)")

print("Building indexed filtering database...")
packages_db = {
    "meta": {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(compact),
    },
    "records": compact,
    "index": index_records(compact),
}

db_path = DOCS / "packages-db.json"
db_path.write_text(json.dumps(packages_db, separators=(",", ":")))
print(f"packages-db.json → {db_path.stat().st_size // 1024}KB")

print("Writing full dataset download...")
full_records = []
for r in records:
    name = r["name"]
    full_records.append({**r, "github": gh_map.get(name, {}), "similar": similar_map.get(name, [])})

full_path = DOCS / "pidex-full.json.gz"
with gzip.open(full_path, "wt", encoding="utf-8") as f:
    json.dump(full_records, f)

print(f"pidex-full.json.gz → {full_path.stat().st_size // 1024}KB")
print("Done.")