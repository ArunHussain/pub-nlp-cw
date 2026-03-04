import csv
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
from transformers import pipeline

DATA_DIR = Path("data")
PLOT_DIR = Path("plots/eda")
SEED = 1
TSNE_SAMPLE_SIZE = 5000
POS_TAG_SAMPLE_SIZE = 2000

def tokenize(text):
    return re.findall(r"[A-Za-z0-9](?:[A-Za-z0-9'-]*[A-Za-z0-9])?", text.lower())


def save_plot(fig, filename, save_dir):
    out = save_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved figure {out}")
    plt.close(fig)


def load_dataset(path):
    cols = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df = pd.read_csv(
        path, sep="\t", skiprows=4, header=None, names=cols,
        quoting=csv.QUOTE_NONE, on_bad_lines="skip",
    )
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["label_binary"] = (df["label"] >= 2).astype(int)
    df["text"] = df["text"].astype(str)

    df["tokens"] = df["text"].apply(tokenize)
    df["token_count"] = df["tokens"].apply(len)
    df["char_count"] = df["text"].str.len()
    df["sentence_count"] = df["text"].str.count(r"[.!?]+").clip(lower=1)
    df["avg_sentence_len"] = df["token_count"] / df["sentence_count"]
    return df.reset_index(drop=True)


def basic_statistical_profiling(df, save_dir):
    print("basic stats")

    label_counts = df["label"].value_counts().sort_index()
    binary_counts = df["label_binary"].value_counts().sort_index()
    binary_pct = (binary_counts / len(df) * 100).round(2)
    vocab = {tok for toks in df["tokens"] for tok in toks}

    print(f"class distribution {label_counts.to_string()}")

    btable = pd.DataFrame({"count": binary_counts, "percent": binary_pct})
    btable.index = ["No PCL", "PCL"]
    print(f"binary class distribution {btable.to_string()}")

    print(f"vocab size {len(vocab)}")

    pctiles = np.percentile(df["token_count"], [50, 95, 99])
    print(
        "token length stats "
        f"min={df['token_count'].min()}, "
        f"mean={df['token_count'].mean():.2f}, "
        f"median={pctiles[0]:.0f}, "
        f"p95={pctiles[1]:.0f}, "
        f"p99={pctiles[2]:.0f}, "
        f"max={df['token_count'].max()}"
    )

    by_class = (
        df.groupby("label_binary")["token_count"]
        .agg(["mean", "median", "min", "max"])
        .rename(index={0: "No PCL (0)", 1: "PCL (1)"})
    )
    print(f"token length by class {by_class.to_string(float_format=lambda x: f'{x:.2f}')}")

    imbalance = binary_counts[0] / max(binary_counts[1], 1)
    suggested_maxlen = int(min(512, np.ceil(pctiles[1] / 32) * 32))
    print(f"imbalance no pcl to pcl {imbalance:.2f} to 1")
    print(f"suggested max sequence length {suggested_maxlen}")

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(["No PCL", "PCL"], binary_counts.values, color=["steelblue", "salmon"])
    ax1.set_title("Binary Class Counts")
    ax1.set_ylabel("Count")
    fig1.tight_layout()
    save_plot(fig1, "binary_class_counts.png", save_dir)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(df[df["label_binary"] == 0]["token_count"], bins=50, alpha=0.7,
             label="No PCL", color="steelblue")
    ax2.hist(df[df["label_binary"] == 1]["token_count"], bins=50, alpha=0.7,
             label="PCL", color="salmon")
    ax2.axvline(pctiles[1], linestyle="--", color="black", linewidth=1, label="p95")
    ax2.set_title("Token Length Distribution")
    ax2.set_xlabel("Token count")
    ax2.legend()
    fig2.tight_layout()
    save_plot(fig2, "token_length_distribution.png", save_dir)

    fig3, ax3 = plt.subplots(figsize=(6, 4.5))
    ax3.boxplot(
        [df[df["label_binary"] == 0]["token_count"], df[df["label_binary"] == 1]["token_count"]],
        tick_labels=["No PCL", "PCL"], showmeans=True,
    )
    ax3.set_title("Token Length by Class")
    ax3.set_ylabel("Token count")
    fig3.tight_layout()
    save_plot(fig3, "token_length_by_class.png", save_dir)


def lexical_analysis(df, save_dir):
    print("lexical stats")

    stopwords = set(ENGLISH_STOP_WORDS)
    df = df.copy()
    df["stop_word_density"] = df["tokens"].apply(
        lambda toks: np.mean([t in stopwords for t in toks]) if toks else 0.0
    )

    sw_stats = (
        df.groupby("label_binary")["stop_word_density"]
        .agg(["mean", "median"])
        .rename(index={0: "No PCL", 1: "PCL"})
    )
    print(f"stop-word density by class {sw_stats.to_string(float_format=lambda x: f'{x:.2f}')}")

    ngram_vec = CountVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(2, 3), min_df=5, max_features=12000,
    )
    ngram_mat = ngram_vec.fit_transform(df["text"])
    terms = np.array(ngram_vec.get_feature_names_out())
    freqs = np.asarray(ngram_mat.sum(axis=0)).ravel()
    top_idx = np.argsort(freqs)[::-1][:20]
    top_ngrams = pd.DataFrame({"ngram": terms[top_idx], "frequency": freqs[top_idx]})
    print("top 20 frequent bigrams/trigrams")
    print(top_ngrams.to_string(index=False))

    uni_vec = CountVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 1), min_df=5, max_features=10000,
    )
    uni_mat = uni_vec.fit_transform(df["text"])
    uni_terms = np.array(uni_vec.get_feature_names_out())

    pcl_mask = df["label_binary"].values == 1
    pcl_freq = np.asarray(uni_mat[pcl_mask].sum(axis=0)).ravel()
    nopcl_freq = np.asarray(uni_mat[~pcl_mask].sum(axis=0)).ravel()

    pcl_rate = pcl_freq / pcl_freq.sum()
    nopcl_rate = nopcl_freq / nopcl_freq.sum()
    eps = 1e-9
    log_ratio = np.log((pcl_rate + eps) / (nopcl_rate + eps))

    top_pcl_idx = np.argsort(log_ratio)[::-1][:15]
    top_nopcl_idx = np.argsort(log_ratio)[:15]

    top_pcl = pd.DataFrame({
        "term": uni_terms[top_pcl_idx], "pcl_freq": pcl_freq[top_pcl_idx],
        "no_pcl_freq": nopcl_freq[top_pcl_idx], "log_ratio": log_ratio[top_pcl_idx],
    })
    top_nopcl = pd.DataFrame({
        "term": uni_terms[top_nopcl_idx], "pcl_freq": pcl_freq[top_nopcl_idx],
        "no_pcl_freq": nopcl_freq[top_nopcl_idx], "log_ratio": log_ratio[top_nopcl_idx],
    })

    print("top unigram terms linked to pcl")
    print(top_pcl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("top unigram terms linked to no pcl")
    print(top_nopcl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.barh(top_pcl["term"][::-1], top_pcl["log_ratio"][::-1], color="salmon")
    ax1.set_title("Top PCL-Leaning Terms (log-ratio)")
    ax1.set_xlabel("log((PCL+1)/(NoPCL+1))")
    fig1.tight_layout()
    save_plot(fig1, "pcl_leaning_terms.png", save_dir)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.barh(top_nopcl["term"], np.abs(top_nopcl["log_ratio"]), color="steelblue")
    ax2.set_title("Top No-PCL-Leaning Terms (abs log-ratio)")
    ax2.set_xlabel("|log((PCL+1)/(NoPCL+1))|")
    fig2.tight_layout()
    save_plot(fig2, "no_pcl_leaning_terms.png", save_dir)


def pos_analysis(df, save_dir, sample_size, seed):
    print("pos tagging")

    sample = df.sample(n=min(sample_size, len(df)), random_state=seed).copy()
    try:
        import torch
        model_device = 0 if torch.cuda.is_available() else -1
    except Exception:
        model_device = -1

    pos_tagger = pipeline(
        "token-classification",
        model="vblagoje/bert-english-uncased-finetuned-pos",
        aggregation_strategy="none", device=model_device, batch_size=32,
    )

    texts = sample["text"].tolist()
    labels = sample["label_binary"].values

    pos_by_class = {0: Counter(), 1: Counter()}
    for i in range(0, len(texts), 64):
        batch = [t[:512] for t in texts[i:i + 64]]
        blabels = labels[i:i + 64]
        results = pos_tagger(batch)
        for tags, lab in zip(results, blabels):
            for t in tags:
                pos_by_class[lab][t["entity"]] += 1

    tags_of_interest = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET", "ADP", "PROPN"]
    total0 = sum(pos_by_class[0].values())
    total1 = sum(pos_by_class[1].values())
    rows = []
    for tag in tags_of_interest:
        r0 = pos_by_class[0].get(tag, 0) / max(total0, 1) * 100
        r1 = pos_by_class[1].get(tag, 0) / max(total1, 1) * 100
        rows.append({"POS": tag, "non PCL (%)": round(r0, 2), "PCL (%)": round(r1, 2)})

    pos_df = pd.DataFrame(rows)
    print("pos tag distribution by class (percent of all tokens in class)")
    print(pos_df.to_string(index=False))

    x = np.arange(len(tags_of_interest))
    w = 0.35
    fig_pos, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, pos_df["non PCL (%)"], w, label="No PCL", color="steelblue")
    ax.bar(x + w/2, pos_df["PCL (%)"], w, label="PCL", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(tags_of_interest)
    ax.set_ylabel("percent of tokens")
    ax.set_title("POS Tag Distribution by Class")
    ax.legend()
    fig_pos.tight_layout()
    save_plot(fig_pos, "pos_distribution_by_class.png", save_dir)


def semantic_syntactic_exploration(df, save_dir, tsne_sample, pos_tag_sample_size, seed):
    print("semantic and syntax checks")

    df = df.copy()
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")
    df["comma_count"] = df["text"].str.count(",")
    df["punctuation_per_100_tokens"] = (
        (df["exclamation_count"] + df["question_count"] + df["comma_count"])
        / df["token_count"].clip(lower=1)
    ) * 100

    syn_summary = (
        df.groupby("label_binary")[["avg_sentence_len", "punctuation_per_100_tokens"]]
        .agg(["mean", "median"])
        .rename(index={0: "No PCL", 1: "PCL"})
    )
    print("syntactic summary by class")
    print(syn_summary.to_string(float_format=lambda x: f"{x:.3f}"))

    n = min(tsne_sample, len(df))
    sdf = df.sample(n=n, random_state=seed).copy()

    tfidf = TfidfVectorizer(
        lowercase=True, stop_words="english",
        ngram_range=(1, 2), min_df=3, max_features=5000,
    )
    X = tfidf.fit_transform(sdf["text"])

    svd_dim = 40
    X_reduced = TruncatedSVD(n_components=svd_dim, random_state=seed).fit_transform(X)

    perp = 30
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perp,
                init="pca", learning_rate="auto")
    coords = tsne.fit_transform(X_reduced)
    sdf["tsne_x"] = coords[:, 0]
    sdf["tsne_y"] = coords[:, 1]

    c0 = coords[sdf["label_binary"].values == 0]
    c1 = coords[sdf["label_binary"].values == 1]
    cdist = float(np.linalg.norm(c0.mean(axis=0) - c1.mean(axis=0)))

    print(f"t-sne details sample={n} tfidf_features={X.shape[1]} svd_dims={svd_dim} tsne_perplexity={perp}")
    print(f"centroid distance in 2d t-sne space (non pcl vs pcl) {cdist:.3f}")

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = {0: "steelblue", 1: "salmon"}
    for cl, name in [(0, "No PCL"), (1, "PCL")]:
        subset = sdf[sdf["label_binary"] == cl]
        ax1.scatter(subset["tsne_x"], subset["tsne_y"], s=10, alpha=0.6,
                    c=colors[cl], label=name)
    ax1.set_title("t-SNE of TF-IDF Embeddings")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    ax1.legend()
    fig1.tight_layout()
    save_plot(fig1, "tsne_tfidf_embeddings.png", save_dir)

    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.boxplot(
        [df[df["label_binary"] == 0]["avg_sentence_len"],
         df[df["label_binary"] == 1]["avg_sentence_len"]],
        tick_labels=["No PCL", "PCL"], showmeans=True,
    )
    ax2.set_title("Average Sentence Length by Class")
    ax2.set_ylabel("Tokens per sentence")
    fig2.tight_layout()
    save_plot(fig2, "avg_sentence_length_by_class.png", save_dir)

    pos_analysis(df, save_dir, pos_tag_sample_size, seed)


def noise_and_artifacts(df, save_dir):
    print("noise and artifact checks")

    df = df.copy()
    df["normalised_text"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True).str.strip()

    exact_dupes = df.duplicated(subset=["text"], keep=False)
    norm_dupes = df.duplicated(subset=["normalised_text"], keep=False)

    q1, q3 = df["token_count"].quantile(0.25), df["token_count"].quantile(0.75)
    iqr = q3 - q1
    iqr_outliers = (df["token_count"] < max(0, q1 - 1.5*iqr)) | (df["token_count"] > q3 + 1.5*iqr)

    p99 = df["token_count"].quantile(0.99)
    very_short = df["token_count"] <= 3
    very_long = df["token_count"] >= p99

    patterns = {
        "html_tag": r"<[^>]+>",
        "url": r"https?://|www\.",
        "non_ascii": r"[^\x00-\x7F]",
    }
    artifact_hits = {name: int(df["text"].str.contains(pat, regex=True).sum())
                     for name, pat in patterns.items()}

    summary = pd.DataFrame(
        [
            ("exact_duplicates", int(exact_dupes.sum())),
            ("normalized_duplicates", int(norm_dupes.sum())),
            ("iqr_length_outliers", int(iqr_outliers.sum())),
            ("very_short_<=3_tokens", int(very_short.sum())),
            ("very_long_>=99th percentile", int(very_long.sum())),
        ] + [(k, v) for k, v in artifact_hits.items()],
        columns=["artifact_type", "count"],
    )
    summary["percent"] = (summary["count"] / len(df) * 100).round(2)

    print("noise and artifact summary")
    print(summary.to_string(index=False))

    print("sample normalized duplicates")
    dupe_rows = df.loc[norm_dupes, ["par_id", "label_binary", "text"]].head(5)
    if dupe_rows.empty:
        print("no duplicate examples found")
    else:
        for _, row in dupe_rows.iterrows():
            preview = re.sub(r"\s+", " ", row["text"])[:140]
            print(f"- par_id={row['par_id']} label={row['label_binary']} text='{preview}...'")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    s = summary.sort_values("count", ascending=False)
    ax.bar(s["artifact_type"], s["count"], color="teal")
    ax.set_title("Detected Noise / Artifact Counts")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    save_plot(fig, "noise_artifacts.png", save_dir)


if __name__ == "__main__":
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(DATA_DIR / "dontpatronizeme_pcl.tsv")

    print(f"loaded {len(df):,} rows from {DATA_DIR / 'dontpatronizeme_pcl.tsv'}")

    basic_statistical_profiling(df, PLOT_DIR)
    lexical_analysis(df, PLOT_DIR)
    semantic_syntactic_exploration(df, PLOT_DIR, TSNE_SAMPLE_SIZE, POS_TAG_SAMPLE_SIZE, SEED)
    noise_and_artifacts(df, PLOT_DIR)
