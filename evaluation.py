import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    auc, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from BestModel.model import PCLClassifier, PCLDataset, clean_text

DATA_DIR = Path("data")
BEST_MODEL_DIR = Path("BestModel")
PLOT_DIR = Path("plots/evaluation")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-large"
MAX_LEN = 256
BEST_MODEL = BEST_MODEL_DIR / "best_model.pt"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_raw_pcl():
    cols = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    df = pd.read_csv(
        DATA_DIR / "dontpatronizeme_pcl.tsv", sep="\t", skiprows=4,
        header=None, names=cols, quoting=csv.QUOTE_NONE, on_bad_lines="skip",
    )
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["par_id"] = df["par_id"].astype(str)
    df["label_binary"] = (df["label"].astype(int) >= 2).astype(int)
    df["text_clean"] = df["text"].astype(str).apply(clean_text)
    df["community"] = df["keyword"].astype(str)
    return df


def load_dev_set():
    raw = load_raw_pcl()
    dev_ids = set(pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")["par_id"].astype(str))
    return raw[raw["par_id"].isin(dev_ids)].reset_index(drop=True)


def get_comm_columns():
    raw = load_raw_pcl()
    comms = set(raw["community"].dropna().unique())

    test_path = DATA_DIR / "task4_test.tsv"
    if test_path.exists():
        tcols = ["par_id", "art_id", "community", "country", "text"]
        tdf = pd.read_csv(test_path, sep="\t", header=None, names=tcols,
                          quoting=csv.QUOTE_NONE, on_bad_lines="skip")
        comms |= set(tdf["community"].dropna().astype(str).unique())

    return sorted(comms)


@torch.no_grad()
def get_probs(model, loader):
    model.eval()
    all_logits = []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        comms = batch["communities"].to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(ids, mask, comms)
        all_logits.append(logits.cpu())

    logits = torch.cat(all_logits).squeeze(-1).float()
    return torch.sigmoid(logits).numpy()


def find_best_threshold(probs, labels):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.10, 0.90, 0.01):
        preds = (probs >= t).astype(int)
        score = f1_score(labels, preds, pos_label=1)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No PCL", "PCL"], yticklabels=["No PCL", "PCL"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Dev Set)")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"FP rate: {fp / (fp + tn):.3f}   FN rate: {fn / (fn + tp):.3f}")
    return cm


def plot_pr_curve(y_true, probs, threshold):
    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec, prec)

    preds = (probs >= threshold).astype(int)
    op_prec = precision_score(y_true, preds)
    op_rec = recall_score(y_true, preds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rec, prec, linewidth=2, label=f"PR AUC = {pr_auc:.3f}")
    ax.scatter([op_rec], [op_prec], color="red", zorder=5, s=80,
               label=f"t={threshold:.2f} (P={op_prec:.2f}, R={op_rec:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "precision_recall_curve.png", dpi=150)
    plt.close(fig)
    print(f"PR AUC: {pr_auc:.4f}")


def plot_threshold_sweep(y_true, probs):
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1s, precs, recs = [], [], []
    for t in thresholds:
        p = (probs >= t).astype(int)
        f1s.append(f1_score(y_true, p, pos_label=1, zero_division=0))
        precs.append(precision_score(y_true, p, zero_division=0))
        recs.append(recall_score(y_true, p, zero_division=0))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.plot(thresholds, precs, label="Precision", linewidth=1.5, linestyle="--")
    ax.plot(thresholds, recs, label="Recall", linewidth=1.5, linestyle="--")

    best_idx = np.argmax(f1s)
    ax.axvline(thresholds[best_idx], color="red", linestyle=":", alpha=0.7,
               label=f"Best t={thresholds[best_idx]:.2f} (F1={f1s[best_idx]:.3f})")
    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sensitivity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "threshold_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_confidence(y_true, probs, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    correct = (probs >= threshold).astype(int) == y_true
    wrong = ~correct

    ax = axes[0]
    ax.hist(probs[y_true == 0], bins=40, alpha=0.6, label="True No-PCL", color="steelblue")
    ax.hist(probs[y_true == 1], bins=40, alpha=0.6, label="True PCL", color="salmon")
    ax.set_xlabel("P(PCL)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Probability by True Class")
    ax.legend()

    conf = np.where(probs >= threshold, probs, 1 - probs)
    ax = axes[1]
    ax.hist(conf[correct], bins=30, alpha=0.6, label="Correct", color="green")
    ax.hist(conf[wrong], bins=30, alpha=0.6, label="Incorrect", color="red")
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence: Correct vs Incorrect")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "confidence_distribution.png", dpi=150)
    plt.close(fig)


def error_analysis(dev_df, probs, threshold):
    preds = (probs >= threshold).astype(int)
    y_true = dev_df["label_binary"].values

    rows = dev_df.copy()
    rows["pred"] = preds
    rows["prob"] = probs

    fp = rows[(rows["pred"] == 1) & (y_true == 0)].sort_values("prob", ascending=False)
    fn = rows[(rows["pred"] == 0) & (y_true == 1)].sort_values("prob", ascending=True)

    print("error analysis")
    print(f"total errors: {len(fp) + len(fn)} / {len(rows)}")
    print(f"false positives (predicted PCL, actually not): {len(fp)}")
    print(f"false negatives (missed PCL): {len(fn)}")

    print("top 10 false positives")
    for i, (_, r) in enumerate(fp.head(10).iterrows(), start=1):
        print(f"FP{i} [p={r['prob']:.3f}] keyword={r['keyword']}")
        print(f"\"{r['text_clean'][:120]}...\"")

    print("top 10 false negatives")
    for i, (_, r) in enumerate(fn.head(10).iterrows(), start=1):
        print(f"FN{i} [p={r['prob']:.3f}] keyword={r['keyword']} orig_label={int(r['label'])}")
        print(f"\"{r['text_clean'][:120]}...\"")

    print("error rates by keyword")
    kw_rows = []
    for kw, grp in rows.groupby("keyword"):
        n_pcl = (grp["label_binary"] == 1).sum()
        fp_n = ((grp["pred"] == 1) & (grp["label_binary"] == 0)).sum()
        fn_n = ((grp["pred"] == 0) & (grp["label_binary"] == 1)).sum()
        kw_f1 = f1_score(grp["label_binary"], grp["pred"], pos_label=1, zero_division=0)
        kw_rows.append({"keyword": kw, "n": len(grp), "pcl": n_pcl,
                         "errors": fp_n + fn_n, "FP": fp_n, "FN": fn_n, "f1": kw_f1})

    kw_df = pd.DataFrame(kw_rows).sort_values("f1")
    print(kw_df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d9534f" if s < 0.4 else "#f0ad4e" if s < 0.6 else "#5cb85c" for s in kw_df["f1"]]
    ax.barh(kw_df["keyword"], kw_df["f1"], color=colors)
    ax.set_xlabel("F1 Score (PCL class)")
    ax.set_title("Per-Topic F1 Score")
    ax.axvline(0.5, color="grey", linestyle="--", alpha=0.5)
    fig.savefig(PLOT_DIR / "f1_by_keyword.png", dpi=150)
    plt.close(fig)

    rows["n_tokens"] = rows["text_clean"].str.split().str.len()
    rows["correct"] = (rows["pred"] == rows["label_binary"]).astype(int)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = [0, 20, 40, 60, 80, 100, 150, 300]
    rows["len_bin"] = pd.cut(rows["n_tokens"], bins=bins)
    err_by_len = (
        rows.groupby("len_bin", observed=True)
        .agg(error_rate=("correct", lambda x: 1 - x.mean()), count=("correct", "count"))
        .reset_index()
    )
    ax.bar(range(len(err_by_len)), err_by_len["error_rate"], color="steelblue")
    ax.set_xticks(range(len(err_by_len)))
    ax.set_xticklabels([str(b) for b in err_by_len["len_bin"]], rotation=30, ha="right")
    ax.set_ylabel("Error Rate")
    ax.set_title("Error Rate by Text Length (tokens)")
    for i, (_, r) in enumerate(err_by_len.iterrows()):
        ax.text(i, r["error_rate"] + 0.005, f"n={r['count']}", ha="center", fontsize=8)
    fig.savefig(PLOT_DIR / "error_rate_by_length.png", dpi=150)
    plt.close(fig)

    return fp, fn


def ablation_threshold(y_true, probs, tuned_t):
    print("ablation: threshold tuning")

    default_preds = (probs >= 0.5).astype(int)
    tuned_preds = (probs >= tuned_t).astype(int)

    for name, fn in [("F1", f1_score), ("Precision", precision_score), ("Recall", recall_score)]:
        d = fn(y_true, default_preds, pos_label=1, zero_division=0)
        t = fn(y_true, tuned_preds, pos_label=1, zero_division=0)
        print(f"  {name}: t=0.50 -> {d:.4f}, t={tuned_t:.2f} -> {t:.4f} (delta={t - d:+.4f})")

    flipped = (default_preds != tuned_preds).sum()
    print(f"{flipped} predictions flipped, default={default_preds.sum()} tuned={tuned_preds.sum()}")


def ablation_community(dev_df, tokenizer, model, comm_cols, threshold):
    print("ablation: community features")
    y_true = dev_df["label_binary"].values

    full_ds = PCLDataset(dev_df, tokenizer, comm_cols, MAX_LEN, text_col="text_clean", label_col="label_binary")
    full_probs = get_probs(model, DataLoader(full_ds, batch_size=BATCH_SIZE))
    full_preds = (full_probs >= threshold).astype(int)

    zeroed_df = dev_df.copy()
    zeroed_df["community"] = "__NONE__"
    zeroed_ds = PCLDataset(zeroed_df, tokenizer, comm_cols, MAX_LEN, text_col="text_clean", label_col="label_binary")
    zeroed_probs = get_probs(model, DataLoader(zeroed_ds, batch_size=BATCH_SIZE))
    zeroed_preds = (zeroed_probs >= threshold).astype(int)

    f1_with = f1_score(y_true, full_preds, pos_label=1)
    f1_without = f1_score(y_true, zeroed_preds, pos_label=1)
    print(f"F1 with community: {f1_with:.4f}, without: {f1_without:.4f} (delta={f1_with - f1_without:+.4f})")
    print(f"Precision with: {precision_score(y_true, full_preds):.4f} without: {precision_score(y_true, zeroed_preds):.4f}")
    print(f"Recall with: {recall_score(y_true, full_preds):.4f} without: {recall_score(y_true, zeroed_preds):.4f}")
    print(f"{(full_preds != zeroed_preds).sum()} predictions flipped")


def per_label_analysis(dev_df, probs, threshold):
    print("performance by original label severity (0-4)")
    print("0,1 = no PCL    2,3,4 = PCL")

    preds = (probs >= threshold).astype(int)
    rows = dev_df.copy()
    rows["pred"] = preds

    print("  label    count    binary   pred PCL %    correct %")

    summary = []
    for lbl in sorted(rows["label"].unique()):
        grp = rows[rows["label"] == lbl]
        binary = 1 if lbl >= 2 else 0
        pred_pct = grp["pred"].mean() * 100
        acc = (grp["pred"] == binary).mean() * 100
        print(f"  {int(lbl):<8} {len(grp):<8} {binary:<8} {pred_pct:<12.1f} {acc:<10.1f}")
        summary.append({"label": int(lbl), "count": len(grp), "pred_pcl_pct": pred_pct, "accuracy": acc})

    sdf = pd.DataFrame(summary)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(sdf["label"], sdf["pred_pcl_pct"],
           color=["steelblue", "steelblue", "salmon", "salmon", "darkred"])
    ax.set_xlabel("original annotation label")
    ax.set_ylabel("% predicted as PCL")
    ax.set_title("PCL prediction rate by label severity")
    ax.set_xticks([0, 1, 2, 3, 4])
    ax.axhline(50, color="grey", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "pred_rate_by_orig_label.png", dpi=150)
    plt.close(fig)


def main():
    print("loading dev set")
    dev_df = load_dev_set()
    y_true = dev_df["label_binary"].values
    print(f"{len(dev_df)} samples, {y_true.sum()} PCL, {(y_true == 0).sum()} No PCL")

    print("building community columns")
    comm_cols = get_comm_columns()
    print(f"{len(comm_cols)} community features")

    print("loading model and tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = PCLClassifier(MODEL_NAME, n_communities=len(comm_cols), dropout=0.3).to(device)
    model.load_state_dict(torch.load(BEST_MODEL, map_location=device))
    model.eval()

    print("running inference")
    dev_ds = PCLDataset(dev_df, tokenizer, comm_cols, MAX_LEN, text_col="text_clean", label_col="label_binary")
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE)
    probs = get_probs(model, dev_loader)

    threshold, best_f1 = find_best_threshold(probs, y_true)
    preds = (probs >= threshold).astype(int)

    print(f"best threshold: {threshold:.2f}, f1={best_f1:.4f}")
    print(f"precision {precision_score(y_true, preds):.4f} recall {recall_score(y_true, preds):.4f}")
    print(classification_report(y_true, preds, target_names=["No PCL", "PCL"]))

    print("confusion matrix")
    plot_confusion(y_true, preds)

    print("pr curve")
    plot_pr_curve(y_true, probs, threshold)

    print("threshold sensitivity")
    plot_threshold_sweep(y_true, probs)

    print("confidence distribution")
    plot_confidence(y_true, probs, threshold)

    error_analysis(dev_df, probs, threshold)
    per_label_analysis(dev_df, probs, threshold)

    ablation_threshold(y_true, probs, threshold)
    ablation_community(dev_df, tokenizer, model, comm_cols, threshold)

    print(f"all plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
