#!/usr/bin/env python3
"""
enrich.py — Fetch full metadata + README for each package in data/raw/packages.json
Output: data/enriched/<package-name>.json  (one file per package)
"""

import json
import time
import re
from pathlib import Path
import requests
from tqdm import tqdm

NPM_REGISTRY = "https://registry.npmjs.org"
NPM_DL_POINT = "https://api.npmjs.org/downloads/point/last-week"
NPM_DL_RANGE = "https://api.npmjs.org/downloads/range"
# Pull ~18 months of weekly data for trend analysis
DL_TREND_START = "2025-01-01"
DL_TREND_END   = "2026-06-25"
RAW = Path(__file__).parent.parent / "data" / "raw" / "packages.json"
OUT_DIR = Path(__file__).parent.parent / "data" / "enriched"


def safe_filename(name: str) -> str:
    """Convert @scope/package to scope__package for filesystem safety."""
    return re.sub(r"[/@]", "__", name).lstrip("_")


def fetch_package(session, name):
    # Full metadata (includes readme, all versions)
    try:
        resp = session.get(f"{NPM_REGISTRY}/{requests.utils.quote(name, safe='')}", timeout=30)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        meta = resp.json()
    except Exception as e:
        print(f"\n  [warn] {name}: metadata fetch failed — {e}")
        return None

    latest_version = meta.get("dist-tags", {}).get("latest")
    latest_meta = meta.get("versions", {}).get(latest_version, {}) if latest_version else {}

    # Download count (last week point)
    downloads = None
    try:
        dl_resp = session.get(f"{NPM_DL_POINT}/{requests.utils.quote(name, safe='')}", timeout=10)
        if dl_resp.status_code == 200:
            downloads = dl_resp.json().get("downloads")
    except Exception:
        pass

    # Download trend (daily counts over date range, for sparklines + trajectory)
    download_trend = []
    try:
        trend_url = f"{NPM_DL_RANGE}/{DL_TREND_START}:{DL_TREND_END}/{requests.utils.quote(name, safe='')}",
        tr_resp = session.get(trend_url[0], timeout=15)
        if tr_resp.status_code == 200:
            # Aggregate daily → weekly buckets
            daily = tr_resp.json().get("downloads", [])
            week, bucket = [], []
            for i, d in enumerate(daily):
                bucket.append(d["downloads"])
                if (i + 1) % 7 == 0 or i == len(daily) - 1:
                    week.append({"week_end": d["day"], "downloads": sum(bucket)})
                    bucket = []
            download_trend = week
    except Exception:
        pass

    return {
        "name": name,
        "latest_version": latest_version,
        "description": meta.get("description"),
        "readme": meta.get("readme", ""),
        "keywords": meta.get("keywords", []),
        "author": meta.get("author"),
        "license": meta.get("license"),
        "homepage": meta.get("homepage"),
        "repository": meta.get("repository"),
        "pi": latest_meta.get("pi", {}),
        "time": meta.get("time", {}),
        "versions_count": len(meta.get("versions", {})),
        "downloads_last_week": downloads,
        "download_trend": download_trend,  # list of {week_end, downloads}
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    packages = json.loads(RAW.read_text())
    print(f"Enriching {len(packages)} packages...")

    skipped = 0
    session = requests.Session()

    for pkg in tqdm(packages, unit="pkg"):
        name = pkg["name"]
        out_file = OUT_DIR / f"{safe_filename(name)}.json"

        if out_file.exists():
            continue  # resume-safe: skip already fetched

        enriched = fetch_package(session, name)
        if enriched is None:
            skipped += 1
            continue

        out_file.write_text(json.dumps(enriched, indent=2))
        time.sleep(0.3)

    print(f"\nDone. Skipped: {skipped}, Errors: {errors}")
    print(f"Enriched files → {OUT_DIR}")


if __name__ == "__main__":
    main()
