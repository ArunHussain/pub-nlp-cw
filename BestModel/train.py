import csv
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from model import PCLClassifier, PCLDataset, clean_text


DATA_DIR = Path("data")
MODEL_SAVE_DIR = Path("BestModel")
PREDICTIONS_DIR = Path("my-predictions")
THRESHOLD_META_PATH = MODEL_SAVE_DIR / "threshold.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {'gpu' if device.type == 'cuda' else 'cpu'}")

# notebook-aligned defaults
SEED = 42
MODEL_NAME = "roberta-large"
MAX_LEN = 256
BS = 16
LR = 1.5e-5
WD = 0.001
EPOCHS = 10
PATIENCE = 3
MIN_EPOCHS = 4
WARMUP_FRAC = 0.1
DROPOUT = 0.3
NUM_AUG = 4
LLRD_FACTOR = 0.95
LLRD_MIN_FACTOR = 0.2
ENABLE_GRAD_CKPT = True
TARGET_POS_TOTAL = 3150
INTERNAL_DEV_FRAC = 0.2


RAND_NAMES = [
    "James", "Marina", "Raya", "Mike", "Simon", "Emma", "Nico", "Sofia", "Maggie", "Sam",
]
RAND_LOCATIONS = [
    "London", "Paris", "Amsterdam", "Birmingham", "Madrid", "Toronto", "Sydney", "Chicago", "New York", "Dublin",
]
CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "i'm": "i am",
    "you're": "you are",
}
INV_CONTRACTIONS = {v: k for k, v in CONTRACTIONS.items()}


def read_raw_pcl():
    cols = ["par_id", "art_id", "keyword", "country_code", "text", "label_raw"]
    df = pd.read_csv(
        DATA_DIR / "dontpatronizeme_pcl.tsv",
        sep="\t",
        skiprows=4,
        header=None,
        names=cols,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["par_id", "keyword", "text", "label_raw"]).copy()
    df["label_raw"] = pd.to_numeric(df["label_raw"], errors="coerce")
    df = df.dropna(subset=["label_raw"]).copy()

    df["par_id"] = df["par_id"].astype(str)
    df["keyword"] = df["keyword"].astype(str)
    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = (df["label_raw"].astype(int) >= 2).astype(int)
    return df[["par_id", "keyword", "text", "label"]]


def rebuild_data(ids_df, raw_df, split_name):
    ids = ids_df["par_id"].astype(str).tolist()
    by_id = raw_df.set_index("par_id")

    rows = []
    missing = 0
    for par_id in ids:
        if par_id not in by_id.index:
            print(f"Id {par_id} found which was not found in original .tsv!!!")
            missing += 1
            # we keep row count aligned with provided id even when source data is missing as we were told on edstem
            rows.append(
                {
                    "par_id": par_id,
                    "keyword": "unknown",
                    "text": "",
                    "label": 0,
                }
            )
            continue

        row = by_id.loc[par_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        rows.append(
            {
                "par_id": par_id,
                "keyword": row["keyword"],
                "text": row["text"],
                "label": int(row["label"]),
            }
        )

    out = pd.DataFrame(rows)
    if missing > 0:
        print(f"warning: {split_name} is missing {missing} ids that were not found in the raw tsv")
    return out.reset_index(drop=True)


def load_train_dev():
    raw = read_raw_pcl()
    train_ids = pd.read_csv(DATA_DIR / "train_semeval_parids-labels.csv")
    dev_ids = pd.read_csv(DATA_DIR / "dev_semeval_parids-labels.csv")
    train_df = rebuild_data(train_ids, raw, "train")
    dev_df = rebuild_data(dev_ids, raw, "dev")
    return train_df, dev_df


def load_test():
    cols = ["par_id", "art_id", "keyword", "country", "text"]
    test_df = pd.read_csv(
        DATA_DIR / "task4_test.tsv",
        sep="\t",
        header=None,
        names=cols,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
    )
    test_df["par_id"] = test_df["par_id"].astype(str)
    test_df["keyword"] = test_df["keyword"].astype(str)
    test_df["text"] = test_df["text"].astype(str).apply(clean_text)
    test_df["label"] = 0
    return test_df[["par_id", "keyword", "text", "label"]]


def split_internal_dev(train_df, frac, seed):
    train_part, dev_part = train_test_split(
        train_df,
        test_size=frac,
        random_state=seed,
        stratify=train_df["label"].astype(int),
    )
    return train_part.reset_index(drop=True), dev_part.reset_index(drop=True)


def _replace_numbers(text):
    numbers = list(re.finditer(r"\b\d+\b", text))
    if not numbers:
        return text
    m = random.choice(numbers)
    n = int(m.group())
    delta = max(1, int(round(abs(n) * random.uniform(0.1, 0.3))))
    new_n = max(0, n + random.choice([-delta, delta]))
    return text[:m.start()] + str(new_n) + text[m.end():]


def _replace_name(text):
    tokens = list(re.finditer(r"\b[A-Z][a-z]{2,}\b", text))
    if not tokens:
        return text
    m = random.choice(tokens)
    replacement = random.choice(RAND_NAMES)
    return text[:m.start()] + replacement + text[m.end():]


def _replace_location(text):
    for loc in RAND_LOCATIONS:
        if re.search(rf"\b{re.escape(loc)}\b", text):
            replacement = random.choice([x for x in RAND_LOCATIONS if x != loc])
            return re.sub(rf"\b{re.escape(loc)}\b", replacement, text, count=1)
    return text


def _toggle_contraction(text):
    lower = text.lower()
    for c, expanded in CONTRACTIONS.items():
        if c in lower:
            return re.sub(rf"\b{re.escape(c)}\b", expanded, text, flags=re.IGNORECASE, count=1)
    for expanded, c in INV_CONTRACTIONS.items():
        if expanded in lower:
            return re.sub(rf"\b{re.escape(expanded)}\b", c, text, flags=re.IGNORECASE, count=1)
    return text


def _word_change_ratio(a, b):
    wa = a.split()
    wb = b.split()
    if not wa:
        return 0.0
    m = max(len(wa), len(wb))
    diff = abs(len(wa) - len(wb))
    for i in range(min(len(wa), len(wb))):
        if wa[i].lower() != wb[i].lower():
            diff += 1
    return diff / max(1, m)


def augment_text(text, target_ratio=0.2):
    ops = [_replace_numbers, _replace_name, _replace_location, _toggle_contraction]
    out = text
    for _ in range(8):
        op = random.choice(ops)
        new_out = op(out)
        if new_out != out:
            out = new_out
        if _word_change_ratio(text, out) >= target_ratio:
            break
    return out


def build_augmented(train_df):
    pos = train_df[train_df["label"] == 1]
    rows = []
    for _, row in pos.iterrows():
        seen = set()
        for _ in range(NUM_AUG * 3):
            aug_text = augment_text(row["text"], target_ratio=0.2)
            if aug_text == row["text"] or aug_text in seen:
                continue
            seen.add(aug_text)
            rows.append(
                {
                    "par_id": row["par_id"],
                    "keyword": row["keyword"],
                    "text": aug_text,
                    "label": 1,
                }
            )
            if len(seen) >= NUM_AUG:
                break
    aug = pd.DataFrame(rows)
    aug.drop_duplicates(subset=["text"], inplace=True)
    aug.reset_index(drop=True, inplace=True)
    target_aug = max(0, TARGET_POS_TOTAL - int(train_df["label"].sum()))
    if len(aug) > target_aug:
        aug = aug.sample(n=target_aug, random_state=SEED).reset_index(drop=True)
    print(f"built {len(aug)} augmentation rows")
    return aug


def inverse_sqrt_sampler(labels):
    y = np.array(labels, dtype=np.int64)
    frac_pos = y.mean()
    frac_neg = 1.0 - frac_pos
    w_pos = 1.0 / np.sqrt(max(frac_pos, 1e-8))
    w_neg = 1.0 / np.sqrt(max(frac_neg, 1e-8))
    sample_weights = np.where(y == 1, w_pos, w_neg)
    return WeightedRandomSampler(sample_weights, num_samples=len(y), replacement=True)


def build_optimizer_with_llrd(model, base_lr, wd, decay):
    def use_weight_decay(param_name):
        if param_name.endswith("bias"):
            return False
        if "LayerNorm.weight" in param_name:
            return False
        if "LayerNorm.bias" in param_name:
            return False
        return True

    n_layers = model.backbone.config.num_hidden_layers
    layer_decay = [[] for _ in range(n_layers)]
    layer_no_decay = [[] for _ in range(n_layers)]
    head_decay, head_no_decay = [], []
    emb_decay, emb_no_decay = [], []
    leftover = []

    for name, param in model.named_parameters():
        decays = use_weight_decay(name)
        if name.startswith("hidden") or name.startswith("classifier"):
            if decays:
                head_decay.append(param)
            else:
                head_no_decay.append(param)
            continue

        layer_match = re.search(r"backbone\.encoder\.layer\.(\d+)\.", name)
        if layer_match is not None:
            idx = int(layer_match.group(1))
            if 0 <= idx < n_layers:
                if decays:
                    layer_decay[idx].append(param)
                else:
                    layer_no_decay[idx].append(param)
                continue

        if "backbone.embeddings" in name:
            if decays:
                emb_decay.append(param)
            else:
                emb_no_decay.append(param)
            continue

        leftover.append(param)

    groups = []
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": wd})
    if head_no_decay:
        groups.append({"params": head_no_decay, "lr": base_lr, "weight_decay": 0.0})

    for i in range(n_layers - 1, -1, -1):
        layer_lr = base_lr * (decay ** (n_layers - 1 - i))
        if layer_decay[i]:
            groups.append({"params": layer_decay[i], "lr": layer_lr, "weight_decay": wd})
        if layer_no_decay[i]:
            groups.append({"params": layer_no_decay[i], "lr": layer_lr, "weight_decay": 0.0})

    emb_lr = max(base_lr * (decay ** n_layers), base_lr * LLRD_MIN_FACTOR)
    if emb_decay:
        groups.append({"params": emb_decay, "lr": emb_lr, "weight_decay": wd})
    if emb_no_decay:
        groups.append({"params": emb_no_decay, "lr": emb_lr, "weight_decay": 0.0})
    if leftover:
        groups.append({"params": leftover, "lr": emb_lr, "weight_decay": 0.0})

    return torch.optim.AdamW(groups)


def train_epoch(model, loader, optimizer, scheduler, criterion, scaler):
    model.train()
    running_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        keyword_features = batch["keywords"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(input_ids, attention_mask, keyword_features)
            loss = criterion(logits, labels)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            
            if scaler.get_scale() >= scale_before:
                scheduler.step()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(loader.dataset)


@torch.no_grad()
def predict(
    model,
    loader,
    criterion=None,
):
    model.eval()
    probs, labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        keyword_features = batch["keywords"].to(device)
        y = batch["labels"].to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, keyword_features)
            if criterion is not None:
                total_loss += criterion(logits, y).item() * y.size(0)
        probs.append(torch.sigmoid(logits).float().cpu())
        labels.append(batch["labels"])

    p = torch.cat(probs).squeeze(-1).numpy()
    y = torch.cat(labels).squeeze(-1).numpy()
    loss = (total_loss / len(loader.dataset)) if criterion is not None else None
    return p, y, loss


def find_best_threshold(probs, labels):
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.8, 0.01):
        f1 = f1_score(labels, (probs >= t).astype(int), pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, best_f1


def save_lines(path, preds):
    with open(path, "w") as f:
        for p in preds:
            f.write(f"{int(p)}\n")


def save_threshold_metadata(path, threshold, threshold_f1, checkpoint_threshold):
    payload = {
        "threshold": float(threshold),
        "internal_val_f1_at_threshold": float(threshold_f1),
        "checkpoint_time_threshold": float(checkpoint_threshold),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def print_binary_metrics(title, labels, preds, with_report=False):
    print(f"\n{title}")
    print(f"f1 {f1_score(labels, preds, pos_label=1):.4f}")
    print(f"precision {precision_score(labels, preds, zero_division=0):.4f}")
    print(f"recall {recall_score(labels, preds, zero_division=0):.4f}")
    if with_report:
        print(classification_report(labels, preds, target_names=["Not PCL", "PCL"], zero_division=0))


def print_threshold_usage(split_name, probs, threshold):
    preds = (probs >= threshold).astype(int)
    print(
        f"{split_name} uses threshold p >= {threshold:.2f}. "
        f"predicted pcl rate {preds.mean():.3f}, mean p(pcl) {probs.mean():.3f}"
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    MODEL_SAVE_DIR.mkdir(exist_ok=True)
    PREDICTIONS_DIR.mkdir(exist_ok=True)

    print(f"running on device {device}")
    official_train_df, official_dev_df = load_train_dev()
    test_df = load_test()
    print(f"official train set has {len(official_train_df)} rows with {int(official_train_df['label'].sum())} positives")
    print(f"official dev set has {len(official_dev_df)} rows with {int(official_dev_df['label'].sum())} positives")
    print(f"test set has {len(test_df)} rows")

    train_df, internal_dev_df = split_internal_dev(official_train_df, INTERNAL_DEV_FRAC, SEED)
    print(
        f"internal split gives train {len(train_df)} rows ({int(train_df['label'].sum())} positives) "
        f"and validation {len(internal_dev_df)} rows ({int(internal_dev_df['label'].sum())} positives)"
    )

    aug_df = build_augmented(train_df)
    final_train = pd.concat([train_df, aug_df], ignore_index=True)
    final_train = final_train.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"final training set after augmentation has {len(final_train)} rows with {int(final_train['label'].sum())} positives")

    keyword_columns = sorted(
        pd.concat(
            [
                final_train["keyword"],
                internal_dev_df["keyword"],
                official_dev_df["keyword"],
                test_df["keyword"],
            ],
            ignore_index=True,
        )
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    print(f"number of keyword features {len(keyword_columns)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = PCLDataset(final_train, tokenizer, keyword_columns, MAX_LEN)
    internal_dev_ds = PCLDataset(internal_dev_df, tokenizer, keyword_columns, MAX_LEN)
    official_dev_ds = PCLDataset(official_dev_df, tokenizer, keyword_columns, MAX_LEN)
    test_ds = PCLDataset(test_df, tokenizer, keyword_columns, MAX_LEN)

    train_sampler = inverse_sqrt_sampler(final_train["label"].tolist())
    train_loader = DataLoader(train_ds, batch_size=BS, sampler=train_sampler)
    internal_dev_loader = DataLoader(internal_dev_ds, batch_size=BS, shuffle=False)
    official_dev_loader = DataLoader(official_dev_ds, batch_size=BS, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False)

    model = PCLClassifier(MODEL_NAME, n_keywords=len(keyword_columns), dropout=DROPOUT, grad_ckpt=ENABLE_GRAD_CKPT).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = build_optimizer_with_llrd(model, LR, WD, LLRD_FACTOR)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    total_steps = max(1, len(train_loader) * EPOCHS)
    warmup_steps = int(total_steps * WARMUP_FRAC)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_f1 = -1.0
    best_val_t_at_ckpt = 0.5
    best_model_path = MODEL_SAVE_DIR / "best_model.pt"
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            scaler,
        )
        va_probs, va_labels, va_loss = predict(model, internal_dev_loader, criterion)
        assert va_loss is not None
        va_preds = (va_probs >= 0.5).astype(int)
        va_f1 = f1_score(va_labels, va_preds, pos_label=1)
        epoch_best_t, epoch_best_f1 = find_best_threshold(va_probs, va_labels)
        epoch_preds_best_t = (va_probs >= epoch_best_t).astype(int)
        epoch_best_prec = precision_score(
            va_labels,
            epoch_preds_best_t,
            zero_division=0,
        )
        epoch_best_rec = recall_score(
            va_labels,
            epoch_preds_best_t,
            zero_division=0,
        )

        print(
            f"epoch {epoch:02d} train loss {tr_loss:.4f}, internal validation loss {va_loss:.4f}, "
            f"internal validation f1 at p>=0.50 is {va_f1:.4f}, "
            f"and at p>={epoch_best_t:.2f} is {epoch_best_f1:.4f}"
        )
        print(
            f"internal validation at p>={epoch_best_t:.2f} gives "
            f"precision {epoch_best_prec:.4f}, "
            f"recall {epoch_best_rec:.4f}, "
            f"predicted pcl rate {epoch_preds_best_t.mean():.3f}"
        )

        # Keep best checkpoint by threshold-tuned F1 on internal dev (no official dev peeking).
        if epoch_best_f1 > best_val_f1:
            print(
                f"new best internal validation score with f1 {epoch_best_f1:.4f} "
                f"at p>={epoch_best_t:.2f}"
            )
            best_val_f1 = epoch_best_f1
            best_val_t_at_ckpt = epoch_best_t
            no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            no_improve += 1
            if epoch >= MIN_EPOCHS and no_improve >= PATIENCE:
                print(
                    f"early stopping at epoch {epoch} while monitoring "
                    f"internal validation f1 at the best threshold"
                )
                break

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_probs_best, val_labels_best, _ = predict(model, internal_dev_loader)
    val_preds_050 = (val_probs_best >= 0.5).astype(int)
    val_best_t, val_best_f1 = find_best_threshold(
        val_probs_best,
        val_labels_best,
    )
    val_preds_best_t = (val_probs_best >= val_best_t).astype(int)
    print_threshold_usage(
        "internal validation set (held-out split, from reloaded best model)",
        val_probs_best,
        val_best_t,
    )
    print_binary_metrics(
        "internal validation metrics at p>=0.50",
        val_labels_best,
        val_preds_050,
        with_report=True,
    )
    print_binary_metrics(
        f"internal validation metrics at best threshold p>={val_best_t:.2f}",
        val_labels_best,
        val_preds_best_t,
    )
    print(
        f"internal validation threshold sweep summary from the reloaded best model: "
        f"best threshold is p>={val_best_t:.2f} "
        f"with f1 {val_best_f1:.4f}, and the threshold saved at checkpoint time "
        f"was {best_val_t_at_ckpt:.2f}"
    )
    save_threshold_metadata(
        THRESHOLD_META_PATH,
        threshold=val_best_t,
        threshold_f1=val_best_f1,
        checkpoint_threshold=best_val_t_at_ckpt,
    )
    print(f"saved threshold metadata to {THRESHOLD_META_PATH}")

    dev_probs, dev_labels, _ = predict(model, official_dev_loader)
    dev_preds = (dev_probs >= 0.5).astype(int)
    dev_preds_best_t = (dev_probs >= val_best_t).astype(int)
    print_threshold_usage("official dev set", dev_probs, val_best_t)

    print_binary_metrics(
        "official dev metrics after training at p>=0.50",
        dev_labels,
        dev_preds,
        with_report=True,
    )
    print_binary_metrics(
        f"official dev metrics after training at the internal validation threshold "
        f"p>={val_best_t:.2f}",
        dev_labels,
        dev_preds_best_t,
    )

    save_lines(PREDICTIONS_DIR / "dev.txt", dev_preds_best_t)
    print(
        f"wrote {len(dev_preds_best_t)} lines to output/dev.txt "
        f"using p>={val_best_t:.2f}"
    )

    test_probs, _, _ = predict(model, test_loader)
    test_preds = (test_probs >= val_best_t).astype(int)
    print_threshold_usage("test set (labels unavailable)", test_probs, val_best_t)
    save_lines(PREDICTIONS_DIR / "test.txt", test_preds)
    print(f"wrote {len(test_preds)} lines to output/test.txt using p>={val_best_t:.2f}")


if __name__ == "__main__":
    main()
