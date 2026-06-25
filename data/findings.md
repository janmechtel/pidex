# Pidex Phase 0 — Findings

**Date:** 2026-06-25  
**Packages analysed:** 457 enriched (out of 4,420 total with `pi-package` keyword)

---

## 1. Scale & Type Breakdown

| Type | Count |
|------|-------|
| `pi-extension` | 228 |
| unknown (no type keyword) | 226 |
| `pi-skill` | 2 |
| `pi-theme` | 1 |

**~50% of packages don't declare a type keyword.** This means pi.dev's own filtering is incomplete — Pidex needs to infer type from README/description rather than relying on keywords alone.

The 4,420 total packages in the npm search results is surprisingly large. Many are likely forks, clones, or packages that added `pi-package` loosely. The enriched sample of 457 is the higher-quality slice (packages with enough metadata to fetch).

---

## 2. Publisher Landscape

- **"GitHub Actions" = 143 packages** — automated CI publishes, often forks or derivatives of other packages. Signals how many packages are vibe-coded and auto-published without much curation.
- A handful of power publishers: `juicesharp` (11), `firstpick` (11), `nicopreme` (8), `fitchmultz` (7).
- Long tail of one-package authors.

---

## 3. Download Distribution

Strongly power-law. A few packages get enormous traffic; most get almost nothing.

| Rank | Package | Downloads/week |
|------|---------|---------------|
| 1 | `pi-web-access` | 33,797 |
| 2 | `context-mode` | 31,344 |
| 3 | `pi-subagents` | 27,835 |
| 4 | `pi-mcp-adapter` | 20,802 |
| 5 | `@hypabolic/pi-hypa` | 12,754 |

The top 5 take roughly 50% of all downloads across the sample. A user picking randomly has a high chance of choosing a package with <100 downloads/week. **This is exactly the discovery problem Pidex solves.**

---

## 4. Download Trends & Trajectories

| Trajectory | Count |
|------------|-------|
| New (< 6 weeks data) | 24 |
| Stable | 18 |
| Growing | 18 |
| Declining | 7 |

The ecosystem is relatively young — most packages either brand new or still in a stable growth phase. Only 7 clearly declining, suggesting churn is still low. **Trend data is a strong signal for Pidex** — "growing" packages are better recommendations than stagnating ones.

---

## 5. Near-Duplicates — 675 Pairs

**675 pairs with TF-IDF similarity > 0.25.** This strongly validates the core Pidex hypothesis: the ecosystem is full of functionally overlapping packages that are hard to distinguish by name/description alone.

Top near-duplicate pairs (by similarity score):

| pkg_a | pkg_b | similarity |
|-------|-------|-----------|
| `@viniraioli/pi-claude-style-tools` | `pi-claude-style-tools` | 0.982 |
| `@jmfederico/pi-web` | `@chainingintention/pi-web-cn` | 0.799 |
| `@jerryan/pi-hashline-edit` | `pi-hashline-edit` | 0.796 |
| `pi-hashline-edit-pro` | `pi-hashline-edit` | 0.714 |
| `pi-simplify` | `@geminixiang/pi-simplify` | 0.611 |

Many of these are literal forks or near-copies. The similarity score is a ready-made "this package is a fork of X" signal.

---

## 6. Cluster Patterns

| Pattern | Clusters | Meaning |
|---------|----------|---------|
| Contested | 6 | Multiple packages competing in the same space |
| Long-tail | 4 | Many small niche packages, no dominant player |
| Blockbuster | 2 | One clear winner, others trail far behind |

### Blockbuster clusters

**Web search** — `pi-web-access` (33,797/wk) dominates. Competitors (`@ollama/pi-web-search`, `pi-deepseek-search`) exist but are 10x smaller. A user should just use `pi-web-access` unless they have a specific provider constraint.

**MCP adapters** — `pi-mcp-adapter` (20,802/wk) is the clear standard. `@spences10/pi-mcp`, `pi-zai-mcp` are niche alternatives.

### Contested clusters (most interesting for Pidex)

**Context management** — `context-mode` vs `@hypabolic/pi-hypa` — both tackle context window saving, different approaches. Classic "which one?" question.

**Subagents** — `pi-subagents`, `@tintinweb/pi-subagents`, `@gotgenes/pi-subagents`, `pi-subagentura` — four packages doing similar things with different APIs.

**Status bars** — `pi-powerline-footer`, `@xynogen/pix-footer`, `@npm-ken/pi-bar`, `@talex-touch/pi-powerline-footer` — all status bar extensions.

**Workflows** — `@quintinshaw/pi-dynamic-workflows` leads a contested cluster of workflow/orchestration packages.

**juicesharp suite** — `@juicesharp/rpiv-*` packages form a tightly coupled ecosystem of their own (todo, ask-user, btw, advisor, web-tools, i18n, args). These should be shown together as a "suite."

---

## 7. Implications for Phase 1

| Finding | Phase 1 implication |
|---------|-------------------|
| 675 near-duplicate pairs | "Similar packages" panel is the killer feature — pre-compute it from TF-IDF similarity |
| 50% missing type keywords | Infer type from description+README, don't rely on keywords |
| Power-law downloads | Show download rank prominently; surface the "hidden gem" packages that are growing |
| Trend data is available | Sparklines are viable — 78 weekly data points per package |
| Contested clusters are clear | Pre-define categories: Web Search, Subagents, Context, Memory, Status Bar, Workflows, MCP |
| juicesharp suite pattern | Support "package suites" — group related packages by author+naming pattern |
| Blockbuster packages exist | "Community pick" badge for dominant packages; warn on near-duplicates |

---

## Plots

All plots saved to `data/plots/`:
- `01_types.png` — package type breakdown
- `02_publishers.png` — top 20 publishers
- `03_download_dist.png` — download distribution (linear + log)
- `04_ecosystem_trend.png` — total ecosystem downloads per week
- `05_top15_trends.png` — top 15 packages trend lines
- `06_trajectories.png` — growing/stable/declining breakdown
- `07_wc_desc.png` / `08_wc_readme.png` — word clouds
- `10_clusters.png` — cluster sizes by pattern
- `11_umap.png` — UMAP cluster plot
