# Pidex Phase 0 — Local Data Exploration Design

**Date:** 2026-06-25  
**Status:** Approved  
**Scope:** Phase 0 only — local data exploration before any web product is built

---

## Goal

Understand the pi.dev extension ecosystem before committing to any product architecture. Concretely answer:

- How many packages exist?
- What is the breakdown by type (extension / skill / theme / prompt)?
- How much overlap/duplication is there between packages?
- Do READMEs contain enough signal for feature tagging and similarity?
- What does the download distribution look like (long tail? power law?)?

Findings will directly inform Phase 1 architecture decisions (similarity approach, tagging strategy, UI categories).

---

## Data Source

The canonical source is the **npm registry keyword index**. All pi packages include `"pi-package"` in their `keywords` array.

- **Enumerate:** `https://registry.npmjs.org/-/v1/search?text=keywords:pi-package&size=250` (paginated)
- **Enrich:** `https://registry.npmjs.org/<package-name>` — returns full metadata including README, all versions, download counts, `pi.*` fields

No scraping of pi.dev required. No authentication. Free and rate-limit-friendly.

---

## Components

### `scripts/fetch.py`

Paginates the npm search API until exhausted. Stores results as `data/raw/packages.json`.

Fields captured per package:
- `name`, `description`, `author`
- `keywords` (includes type signals like `pi-extension`, `pi-skill`, etc.)
- `date` (last published)
- `links.npm`, `links.repository`
- `publisher.username`

### `scripts/enrich.py`

For each package in `data/raw/packages.json`, fetches full metadata from the npm registry. Stores as `data/enriched/<name>.json`.

Additional fields captured:
- Full README (markdown)
- All versions + publish dates
- `pi` field from package.json (`pi.video`, `pi.image`)
- Weekly download count (from `https://api.npmjs.org/downloads/point/last-week/<name>`)

Handles missing READMEs and private/unpublished packages gracefully (skip + log).

### `notebooks/explore.ipynb`

Jupyter notebook with the following sections:

1. **Overview** — total package count, type breakdown (bar chart), author distribution
2. **Word clouds** — from descriptions and README text, per type category
3. **TF-IDF similarity** — cosine similarity matrix over descriptions + READMEs; surface top-N near-duplicate pairs
4. **Local embeddings** — `sentence-transformers` model `all-MiniLM-L6-v2` (free, CPU-runnable); UMAP cluster plot to visualize groupings
5. **Download distribution** — histogram + log-scale plot; identify power law vs. flat

### `data/findings.md`

Written after notebook analysis. Documents:
- What we found (numbers, patterns, surprises)
- Implications for Phase 1 (which similarity approach to use, how to define categories, whether tagging is tractable)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | All scripts and notebook |
| `requests` | HTTP calls to npm registry |
| `jupyter` | Interactive exploration |
| `pandas` | Data wrangling |
| `matplotlib` / `seaborn` | Charts |
| `wordcloud` | Word cloud generation |
| `scikit-learn` | TF-IDF + cosine similarity |
| `sentence-transformers` | Local embeddings (no API cost) |
| `umap-learn` | Dimensionality reduction for cluster plot |

No LLM API calls. All processing runs locally on CPU.

---

## Output Artifacts

```
data/
  raw/
    packages.json          # All packages from npm search
  enriched/
    <package-name>.json    # Full metadata + README per package
  findings.md              # Written summary of analysis
notebooks/
  explore.ipynb            # Analysis notebook
scripts/
  fetch.py
  enrich.py
requirements.txt           # Python dependencies
```

---

## Non-Goals (Phase 0)

- No web UI
- No Cloudflare deployment
- No LLM calls
- No similarity recommendations (only exploration — findings inform Phase 1 design)
- No git-sourced packages (npm keyword index only for now)
