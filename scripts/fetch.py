#!/usr/bin/env python3
"""
fetch.py — Enumerate all npm packages with keyword "pi-package"
Output: data/raw/packages.json
"""

import json
import time
from pathlib import Path
import requests

NPM_SEARCH = "https://registry.npmjs.org/-/v1/search"
KEYWORD = "keywords:pi-package"
PAGE_SIZE = 250
OUT = Path(__file__).parent.parent / "data" / "raw" / "packages.json"


def fetch_all():
    packages = []
    offset = 0
    session = requests.Session()

    while True:
        resp = session.get(NPM_SEARCH, params={
            "text": KEYWORD,
            "size": PAGE_SIZE,
            "from": offset,
        }, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        objects = data.get("objects", [])
        total = data.get("total", 0)

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

        print(f"  Fetched {len(packages)}/{total} packages (offset={offset})")
        offset += len(objects)

        if offset >= total:
            break

        time.sleep(0.5)  # be polite

    return packages, total


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching packages with keyword '{KEYWORD}'...")

    packages, total = fetch_all()
    print(f"\nTotal from API: {total}")
    print(f"Packages collected: {len(packages)}")

    OUT.write_text(json.dumps(packages, indent=2))
    print(f"Saved → {OUT}")

    # Quick summary
    by_type = {}
    for p in packages:
        for kw in p.get("keywords", []):
            if kw.startswith("pi-") and kw != "pi-package":
                by_type[kw] = by_type.get(kw, 0) + 1
    print("\nKeyword breakdown:")
    for k, v in sorted(by_type.items(), key=lambda x: -x[1])[:15]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
