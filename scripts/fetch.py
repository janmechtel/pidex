#!/usr/bin/env python3
"""
fetch.py — Enumerate all npm packages with keyword "pi-package"
Output: data/raw/packages.json
Supports resume: saves every page, restarts from last saved offset.
"""

import json
import time
from pathlib import Path
import requests

NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
KEYWORD = "keywords:pi-package"
PAGE_SIZE = 250
OUT = Path(__file__).parent.parent / "data" / "raw" / "packages.json"
PROGRESS = Path(__file__).parent.parent / "data" / "raw" / "fetch_progress.json"


def save(packages, offset, total):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(packages, indent=2))
    PROGRESS.write_text(json.dumps({"offset": offset, "total": total}))


def fetch_all():
    # Resume from last saved state if available
    packages = []
    offset = 0
    if OUT.exists() and PROGRESS.exists():
        packages = json.loads(OUT.read_text())
        progress = json.loads(PROGRESS.read_text())
        offset = progress["offset"]
        total = progress["total"]
        print(f"Resuming from offset={offset}, have {len(packages)} packages already")
    else:
        total = None

    session = requests.Session()

    while True:
        retries = 0
        while True:
            resp = session.get(NPM_SEARCH, params={
                "text": KEYWORD,
                "size": PAGE_SIZE,
                "from": offset,
            }, timeout=30)

            if resp.status_code == 429:
                wait = 10 * (2 ** retries)
                print(f"  Rate limited — waiting {wait}s before retry {retries+1}...")
                time.sleep(wait)
                retries += 1
                if retries > 6:
                    raise RuntimeError("Too many retries on 429")
                continue

            resp.raise_for_status()
            break

        data = resp.json()
        objects = data.get("objects", [])
        total = data.get("total", total)

        if not objects:
            break

        for obj in objects:
            pkg = obj.get("package", {})
            packages.append({
                "name": pkg.get("name"),
                "description": pkg.get("description"),
                "author": pkg.get("author", {}).get("name") if pkg.get("author") else None,
                "publisher": pkg.get("publisher", {}).get("username"),
                "keywords": pkg.get("keywords", []),
                "date": pkg.get("date"),
                "version": pkg.get("version"),
                "links": pkg.get("links", {}),
            })

        offset += len(objects)
        save(packages, offset, total)
        print(f"  Fetched {len(packages)}/{total} (offset={offset}) — saved")

        if offset >= total:
            break

        time.sleep(2)  # polite pause between pages

    return packages, total


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching packages with keyword '{KEYWORD}'...")

    packages, total = fetch_all()
    print(f"\nTotal from API: {total}")
    print(f"Packages collected: {len(packages)}")
    print(f"Saved → {OUT}")

    # Quick summary
    by_type = {}
    for p in packages:
        for kw in p.get("keywords", []):
            if kw.startswith("pi-") and kw != "pi-package":
                by_type[kw] = by_type.get(kw, 0) + 1
    print("\nKeyword breakdown (top 15):")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
