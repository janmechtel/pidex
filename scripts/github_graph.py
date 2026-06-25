#!/usr/bin/env python3
"""
github_graph.py — For top packages by downloads, fetch GitHub metadata:
  - is it a fork? parent repo?
  - stars, open issues, last commit
  - README cross-references: "fork of", "inspired by", "based on", "alternative"
  - which other pi packages mention this package by name?

Output: data/github/<owner>__<repo>.json
        data/fork_graph.json  (edge list)
        data/cross_refs.json  (package mentions by name in other READMEs)
"""

import json, re, time, os
from pathlib import Path
import requests
from tqdm import tqdm

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
DATA = Path(__file__).parent.parent / "data"
OUT_DIR = DATA / "github"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORK_KEYWORDS = re.compile(
    r'\b(fork|forked from|based on|inspired by|alternative to|port of|'
    r'similar to|derived from|originally from|credits? to|see also)\b',
    re.IGNORECASE
)
# Will be filled with all pi package names for cross-reference scan
PI_NAMES: set = set()


def gh_headers():
    h = {"Accept": "application/vnd.github+json",
         "X-GitHub-Api-Version": "2022-11-28"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return h


def extract_github_slug(repo_url: str):
    """Return (owner, repo) from a GitHub URL, or None."""
    if not repo_url:
        return None
    m = re.search(r'github\.com[:/]([^/]+)/([^/.]+)', repo_url)
    if m:
        return m.group(1), m.group(2).rstrip('.git')
    return None


def fetch_github(session, owner, repo):
    url = f"https://api.github.com/repos/{owner}/{repo}"
    r = session.get(url, headers=gh_headers(), timeout=15)
    if r.status_code == 404:
        return None
    if r.status_code == 403:
        print(f"\n  [rate-limit] GitHub — waiting 60s")
        time.sleep(60)
        r = session.get(url, headers=gh_headers(), timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def scan_readme_for_refs(readme: str, pkg_name: str) -> dict:
    """Extract fork/inspiration signals and mentions of other pi packages."""
    fork_lines = []
    for line in readme.splitlines():
        if FORK_KEYWORDS.search(line):
            fork_lines.append(line.strip()[:200])

    mentioned = []
    for name in PI_NAMES:
        if name == pkg_name:
            continue
        # match package name as word boundary (handles scoped @foo/bar too)
        pattern = re.escape(name.split('/')[-1])  # just the unscoped part
        if len(pattern) > 4 and re.search(pattern, readme, re.IGNORECASE):
            mentioned.append(name)

    return {"fork_signals": fork_lines, "mentions_packages": mentioned}


def main():
    # Load enriched packages, filter to those with downloads data, sort by downloads
    records = [json.loads(f.read_text()) for f in (DATA / "enriched").glob("*.json")]
    raw = json.loads((DATA / "raw" / "packages.json").read_text())
    pub_map = {p["name"]: p.get("publisher") for p in raw}

    for r in records:
        r["publisher"] = pub_map.get(r["name"])
        PI_NAMES.add(r["name"])

    # Focus on packages with downloads OR repos (don't skip no-download packages if they have repos)
    has_repo = [r for r in records if extract_github_slug(
        (r.get("repository") or {}).get("url", "") if isinstance(r.get("repository"), dict)
        else str(r.get("repository") or "")
    )]

    # Sort: packages with downloads first, then others
    has_dl = sorted([r for r in has_repo if r.get("downloads_last_week")],
                    key=lambda x: x["downloads_last_week"], reverse=True)
    no_dl  = [r for r in has_repo if not r.get("downloads_last_week")]
    ordered = has_dl + no_dl

    print(f"Packages with GitHub repos: {len(ordered)}")
    print(f"  with download data: {len(has_dl)}")
    if GITHUB_TOKEN:
        print("  GitHub token: set ✓")
    else:
        print("  GitHub token: NOT SET — rate limit is 60 req/hr, will be slow")

    session = requests.Session()
    fork_edges = []
    cross_refs = []

    for pkg in tqdm(ordered, unit="pkg"):
        name = pkg["name"]
        repo_field = pkg.get("repository") or {}
        repo_url = repo_field.get("url", "") if isinstance(repo_field, dict) else str(repo_field)
        slug = extract_github_slug(repo_url)
        if not slug:
            continue

        owner, repo = slug
        safe = f"{owner}__{repo}"
        out_file = OUT_DIR / f"{safe}.json"

        if not out_file.exists():
            gh = fetch_github(session, owner, repo)
            if gh:
                out_file.write_text(json.dumps(gh, indent=2))
            time.sleep(1.5 if not GITHUB_TOKEN else 0.3)
            gh_data = gh or {}
        else:
            gh_data = json.loads(out_file.read_text())

        # Fork edge
        if gh_data.get("fork") and gh_data.get("parent"):
            parent_url = gh_data["parent"].get("html_url", "")
            fork_edges.append({
                "package": name,
                "repo": f"{owner}/{repo}",
                "forked_from": gh_data["parent"].get("full_name"),
                "parent_url": parent_url,
                "stars": gh_data.get("stargazers_count"),
                "open_issues": gh_data.get("open_issues_count"),
                "pushed_at": gh_data.get("pushed_at"),
            })

        # README cross-refs
        readme = pkg.get("readme", "") or ""
        refs = scan_readme_for_refs(readme, name)
        if refs["fork_signals"] or refs["mentions_packages"]:
            cross_refs.append({
                "package": name,
                "downloads": pkg.get("downloads_last_week"),
                **refs
            })

    (DATA / "fork_graph.json").write_text(json.dumps(fork_edges, indent=2))
    (DATA / "cross_refs.json").write_text(json.dumps(cross_refs, indent=2))

    print(f"\nFork edges found: {len(fork_edges)}")
    print(f"Packages with cross-refs in README: {len(cross_refs)}")

    print("\n=== Fork graph (top 20) ===")
    for e in sorted(fork_edges, key=lambda x: x.get("stars") or 0, reverse=True)[:20]:
        print(f"  {e['package']} ← forked from {e['forked_from']}  (⭐{e['stars']})")

    print("\n=== README cross-references (top 20 by downloads) ===")
    top = sorted(cross_refs, key=lambda x: x.get("downloads") or 0, reverse=True)
    for c in top[:20]:
        print(f"\n  {c['package']} ({c['downloads']}/wk)")
        if c["fork_signals"]:
            for line in c["fork_signals"][:2]:
                print(f"    fork signal: {line[:120]}")
        if c["mentions_packages"]:
            print(f"    mentions: {c['mentions_packages'][:5]}")


if __name__ == "__main__":
    main()
