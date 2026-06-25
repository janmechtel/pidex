#!/usr/bin/env python3
"""
analyze.py — Phase 0 analysis on enriched pi packages
Produces plots in data/plots/ and prints a findings summary
"""

import json, re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import numpy as np

sns.set_theme(style='whitegrid')
DATA  = Path(__file__).parent.parent / "data"
PLOTS = DATA / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
records = [json.loads(f.read_text()) for f in (DATA / "enriched").glob("*.json")]
df = pd.DataFrame(records)
# also pull publisher from raw packages
raw = json.loads((DATA / "raw" / "packages.json").read_text())
pub_map = {p["name"]: p.get("publisher") for p in raw}
df["publisher"] = df["name"].map(pub_map)

print(f"Loaded {len(df)} enriched packages")

# ── 1. Type breakdown ─────────────────────────────────────────────────────────
def infer_type(keywords):
    for k in (keywords or []):
        if k in ("pi-extension", "pi-skill", "pi-theme", "pi-prompt"):
            return k.replace("pi-", "")
    return "unknown"

df["type"] = df["keywords"].apply(infer_type)
type_counts = df["type"].value_counts()
print("\n=== Package types ===")
print(type_counts.to_string())

fig, ax = plt.subplots(figsize=(7, 4))
type_counts.plot(kind="bar", ax=ax, color="steelblue")
ax.set_title(f"Pi packages by type  (n={len(df)})")
ax.set_xlabel(""); ax.set_ylabel("count")
plt.tight_layout()
plt.savefig(PLOTS / "01_types.png", dpi=130)
plt.close()

# ── 2. Top publishers ─────────────────────────────────────────────────────────
pub_counts = df["publisher"].value_counts().dropna()
print("\n=== Top publishers ===")
print(pub_counts.head(20).to_string())

fig, ax = plt.subplots(figsize=(8, 6))
pub_counts.head(20).sort_values().plot(kind="barh", ax=ax, color="coral")
ax.set_title("Top 20 publishers")
plt.tight_layout()
plt.savefig(PLOTS / "02_publishers.png", dpi=130)
plt.close()

# ── 3. Download distribution ──────────────────────────────────────────────────
dl = df[["name", "downloads_last_week", "type"]].dropna(subset=["downloads_last_week"]).copy()
dl = dl.sort_values("downloads_last_week", ascending=False)
print(f"\n=== Download distribution ({len(dl)} packages with data) ===")
print(dl.head(20)[["name", "downloads_last_week", "type"]].to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
dl["downloads_last_week"].plot(kind="hist", bins=40, ax=axes[0],
    title="Downloads last week (linear)", color="steelblue")
(dl["downloads_last_week"] + 1).plot(kind="hist", bins=40, ax=axes[1], logy=True,
    title="Downloads last week (log scale)", color="steelblue")
plt.tight_layout()
plt.savefig(PLOTS / "03_download_dist.png", dpi=130)
plt.close()

# ── 4. Download trends ────────────────────────────────────────────────────────
trend_rows = []
for _, row in df.iterrows():
    for entry in (row.get("download_trend") or []):
        trend_rows.append({"name": row["name"], "week_end": entry["week_end"],
                           "downloads": entry["downloads"]})

if trend_rows:
    tdf = pd.DataFrame(trend_rows)
    tdf["week_end"] = pd.to_datetime(tdf["week_end"])
    pivot = tdf.pivot_table(index="week_end", columns="name",
                            values="downloads", aggfunc="sum").fillna(0)

    # Ecosystem total per week
    total_weekly = pivot.sum(axis=1)
    fig, ax = plt.subplots(figsize=(13, 4))
    total_weekly.plot(ax=ax, color="steelblue")
    ax.set_title("Total pi ecosystem downloads per week (sample: enriched packages)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.tight_layout()
    plt.savefig(PLOTS / "04_ecosystem_trend.png", dpi=130)
    plt.close()

    # Top 15 trajectories
    top_names = dl.head(15)["name"].tolist()
    top_names = [n for n in top_names if n in pivot.columns]
    fig, ax = plt.subplots(figsize=(13, 6))
    pivot[top_names].plot(ax=ax)
    ax.set_title("Weekly downloads — top 15 packages")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_top15_trends.png", dpi=130)
    plt.close()

    # Trajectory classification
    def classify_trend(series):
        s = series[series > 0]
        if len(s) < 6: return "new"
        first = s.iloc[:len(s)//2].mean()
        second = s.iloc[len(s)//2:].mean()
        if first == 0: return "new"
        ratio = second / first
        if ratio > 1.5: return "growing"
        if ratio < 0.5: return "declining"
        return "stable"

    trajectories = pivot.apply(classify_trend)
    print("\n=== Trajectory breakdown ===")
    print(trajectories.value_counts().to_string())

    fig, ax = plt.subplots(figsize=(6, 3))
    trajectories.value_counts().plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_title("Package trajectory (growing/stable/declining/new)")
    plt.tight_layout()
    plt.savefig(PLOTS / "06_trajectories.png", dpi=130)
    plt.close()

# ── 5. Word clouds ────────────────────────────────────────────────────────────
def make_wordcloud(texts, title, fname):
    combined = " ".join(t for t in texts if isinstance(t, str))
    combined = re.sub(r"[#*`\[\](){}|>]", " ", combined)
    if not combined.strip():
        return
    wc = WordCloud(width=1200, height=500, background_color="white",
                   max_words=120).generate(combined)
    plt.figure(figsize=(14, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off"); plt.title(title)
    plt.tight_layout()
    plt.savefig(PLOTS / fname, dpi=120)
    plt.close()

make_wordcloud(df["description"].tolist(), "All packages — descriptions", "07_wc_desc.png")
make_wordcloud(df["readme"].fillna("").str[:800].tolist(), "All packages — READMEs", "08_wc_readme.png")

for pkg_type in df["type"].unique():
    subset = df[df["type"] == pkg_type]
    if len(subset) >= 3:
        make_wordcloud(subset["description"].tolist(),
                       f"{pkg_type} descriptions ({len(subset)} pkgs)",
                       f"09_wc_{pkg_type}.png")

# ── 6. TF-IDF near-duplicates ─────────────────────────────────────────────────
df["text"] = df["description"].fillna("") + " " + df["readme"].fillna("").str[:600]
vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1, 2))
tfidf = vec.fit_transform(df["text"])
sim = cosine_similarity(tfidf)
np.fill_diagonal(sim, 0)

pairs = []
n = len(df)
for i in range(n):
    for j in range(i+1, n):
        if sim[i, j] > 0.25:
            pairs.append({
                "pkg_a": df.iloc[i]["name"], "pkg_b": df.iloc[j]["name"],
                "similarity": round(sim[i, j], 3),
                "dl_a": df.iloc[i]["downloads_last_week"],
                "dl_b": df.iloc[j]["downloads_last_week"],
            })

pairs_df = pd.DataFrame(pairs).sort_values("similarity", ascending=False) if pairs else pd.DataFrame()
print(f"\n=== Near-duplicate pairs (TF-IDF > 0.25): {len(pairs_df)} ===")
if len(pairs_df):
    print(pairs_df.head(20).to_string(index=False))

# ── 7. Cluster analysis — blockbuster vs long tail ────────────────────────────
N_CLUSTERS = 12
svd = TruncatedSVD(n_components=min(50, tfidf.shape[1]-1), random_state=42)
reduced = svd.fit_transform(tfidf)
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
df["cluster"] = km.fit_predict(reduced)

feature_names = vec.get_feature_names_out()
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

print("\n=== Cluster Summary ===")
cluster_rows = []
for i in range(N_CLUSTERS):
    cpkgs = df[df["cluster"] == i]
    top_terms = [feature_names[ind] for ind in order_centroids[i, :6]]
    dls = cpkgs["downloads_last_week"].dropna()
    total_dl = dls.sum()
    max_dl = dls.max() if len(dls) else 0
    ratio = (max_dl / total_dl) if total_dl > 0 else 0
    pattern = "blockbuster" if ratio > 0.6 else ("contested" if ratio > 0.35 else "long-tail")
    cluster_rows.append({"cluster": i, "n_pkgs": len(cpkgs),
                          "top_terms": ", ".join(top_terms[:4]),
                          "total_dl": int(total_dl), "ratio": round(ratio, 2),
                          "pattern": pattern})
    print(f"  C{i:02d} | {len(cpkgs):3d} pkgs | {pattern:12s} | {', '.join(top_terms[:4])}")

clusters_df = pd.DataFrame(cluster_rows)
print("\nPattern counts:", clusters_df["pattern"].value_counts().to_dict())

# Drill into contested/blockbuster clusters
print("\n=== Contested & Blockbuster Clusters ===")
top_clusters = clusters_df[clusters_df["pattern"].isin(["contested","blockbuster"])]\
    .sort_values("total_dl", ascending=False).head(6)
for _, cr in top_clusters.iterrows():
    cid = cr["cluster"]
    cpkgs = df[df["cluster"] == cid][["name","description","downloads_last_week","type"]]\
        .sort_values("downloads_last_week", ascending=False)
    print(f"\n  Cluster {cid} [{cr['pattern']}] — {cr['top_terms']}")
    print(cpkgs.head(8).to_string(index=False))

# Cluster sizes chart
fig, ax = plt.subplots(figsize=(10, 4))
clusters_df_plot = clusters_df.copy()
clusters_df_plot["label"] = clusters_df_plot.apply(
    lambda r: f"C{r['cluster']}: {r['top_terms'].split(',')[0]}", axis=1)
colors = {"blockbuster":"crimson","contested":"darkorange","long-tail":"steelblue"}
clusters_df_plot.set_index("label")["n_pkgs"].plot(
    kind="bar", ax=ax, color=[colors[p] for p in clusters_df_plot["pattern"]])
ax.set_title("Packages per cluster  (red=blockbuster, orange=contested, blue=long-tail)")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig(PLOTS / "10_clusters.png", dpi=130)
plt.close()

# ── 8. UMAP (optional) ────────────────────────────────────────────────────────
try:
    import umap
    from sentence_transformers import SentenceTransformer
    print("\nComputing sentence-transformer embeddings for UMAP...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True,
                               batch_size=32)
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42)
    coords = reducer.fit_transform(embeddings)
    df["ux"], df["uy"] = coords[:, 0], coords[:, 1]

    type_colors = {"extension":"steelblue","skill":"darkorange",
                   "theme":"green","prompt":"crimson","unknown":"grey"}
    fig, ax = plt.subplots(figsize=(13, 10))
    for t, grp in df.groupby("type"):
        ax.scatter(grp["ux"], grp["uy"], c=type_colors.get(t,"grey"),
                   label=t, alpha=0.7, s=50)
    top20 = df.nlargest(20, "downloads_last_week")
    for _, row in top20.iterrows():
        ax.annotate(row["name"].split("/")[-1], (row["ux"], row["uy"]),
                    fontsize=6.5, ha="center", va="bottom")
    ax.legend()
    ax.set_title("UMAP — pi packages (labelled = top 20 by downloads)")
    plt.tight_layout()
    plt.savefig(PLOTS / "11_umap.png", dpi=130)
    plt.close()
    print("UMAP saved.")
except Exception as e:
    print(f"UMAP skipped: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FINDINGS SUMMARY")
print("="*60)
print(f"  Enriched packages analysed : {len(df)}")
print(f"  Package types              : {type_counts.to_dict()}")
print(f"  Near-duplicate pairs       : {len(pairs_df)}")
print(f"  Cluster patterns           : {clusters_df['pattern'].value_counts().to_dict()}")
if trend_rows:
    print(f"  Trajectory breakdown       : {trajectories.value_counts().to_dict()}")
print(f"\nPlots saved to: {PLOTS}")
