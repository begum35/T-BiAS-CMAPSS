# ================================================================
# C-MAPSS TBAS-XGB FINAL DEVELOPMENT / EVALUATION PIPELINE
#
# TCN + BiGRU + Attention
# + 204 Semantic Features
# + XGBoost Ensemble
#
# DEVELOPMENT PROTOCOL
# ---------------------------------------------------------------
# NASA train set
#       |
#       +-- 80% TRAIN engines
#       |      - scaler fit
#       |      - encoder training
#       |      - XGBoost GroupKFold RandomizedSearch
#       |
#       +-- 20% VALIDATION engines
#              - encoder checkpoint selection using F2 @ 0.5
#              - final XGBoost early stopping
#              - decision-threshold selection from PR curve, MAX F1
#
# NASA official test
#       |
#       +-- FINAL EVALUATION ONLY
#       +-- one final 40-cycle window per engine
#       +-- frozen validation threshold
#
# IMPORTANT
# ---------------------------------------------------------------
# Split is ENGINE-LEVEL.
# Windows from one engine can NEVER be in both train and validation.
#
# XGBoost RandomizedSearch:
#   only TRAIN engines
#   GroupKFold(n_splits=3)
#   scoring = F2
#
# Threshold:
#   selected only from VALIDATION
#   candidates = validation prediction-score boundaries from
#                precision_recall_curve
#   criterion  = maximum F1
#   tie-break  = recall -> precision -> higher threshold
#
# No arbitrary threshold grid.
# No isotonic calibration.
# No minimum-recall constraint.
# ================================================================

# In Google Colab run once if xgboost is unavailable:
# !pip install -q xgboost

import gc
import copy
import json
import time
import random
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    make_scorer, confusion_matrix, roc_auc_score,
    average_precision_score, precision_recall_curve
)
from xgboost import XGBClassifier, DMatrix, train as xgb_train

warnings.filterwarnings("ignore")


# ================================================================
# CONFIG
# ================================================================

CFG = {
    "seed": 42,
    "deterministic": True,
    "data_dir": "/content",
    "datasets": ["FD001", "FD002", "FD003", "FD004"],

    "rul_threshold": 30,
    "seq_len": 40,
    "stride": 1,
    "val_engine_frac": 0.20,

    "tcn_channels": [64, 128, 128, 64],
    "tcn_kernel": 3,
    "tcn_dropout": 0.20,
    "gru_hidden": 128,
    "gru_layers": 2,
    "gru_dropout": 0.30,
    "embed_dim": 128,

    "epochs": 60,
    "batch_size": 256,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "patience": 12,
    "min_delta": 1e-4,
    "recon_weight": 0.30,
    "use_amp": True,

    "random_search_iterations": 100,
    "search_cv_folds": 3,
    "search_n_jobs": 1,
    "search_xgb_estimators": 200,

    "xgb_param_distributions": {
        "max_depth": [3, 4, 6],
        "learning_rate": [0.03, 0.05, 0.10],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 2.0],
    },

    "ensemble_size": 5,
    "xgb_rounds": 800,
    "xgb_es_rounds": 40,

    # No threshold_grid. Candidate thresholds come from PR curve.
    "save_artifacts": True,
    "artifact_dir": "/content/artifacts/final_80_20_pr_f1_protocol",

    # Minimum validation precision a candidate threshold must reach
    # before it is eligible for max-F1 selection. Set to 0.0 to
    # fully recover the original unconstrained behavior.
    "min_precision_floor": 0.90,
}


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print("DEVICE:", DEVICE)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 100)

ARTIFACT_DIR = Path(CFG["artifact_dir"])
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# REPRODUCIBILITY
# ================================================================

def seed_everything(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


seed_everything(CFG["seed"], CFG["deterministic"])


# ================================================================
# C-MAPSS DATA
# ================================================================

COLUMNS = (
    ["unit_number", "time_in_cycles"]
    + [f"op_{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)


def resolve_data_dir(fd):
    candidates = [
        Path(CFG["data_dir"]),
        Path("/content"),
        Path("/content/data"),
        Path("/content/CMAPSSData"),
        Path.cwd(),
    ]

    for directory in candidates:
        if (
            (directory / f"train_{fd}.txt").exists()
            and (directory / f"test_{fd}.txt").exists()
            and (directory / f"RUL_{fd}.txt").exists()
        ):
            return directory

    raise FileNotFoundError(f"C-MAPSS files for {fd} could not be found.")


def load_cmapss(fd):
    directory = resolve_data_dir(fd)

    train_df = pd.read_csv(
        directory / f"train_{fd}.txt", sep=r"\s+", header=None, names=COLUMNS
    )
    test_df = pd.read_csv(
        directory / f"test_{fd}.txt", sep=r"\s+", header=None, names=COLUMNS
    )
    test_rul = pd.read_csv(
        directory / f"RUL_{fd}.txt", sep=r"\s+", header=None
    ).iloc[:, 0].to_numpy(dtype=np.float32)

    return train_df, test_df, test_rul


def add_train_rul(df):
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_in_cycles"].max()
    df = df.join(max_cycles.rename("max_cycle"), on="unit_number")
    df["RUL"] = df["max_cycle"] - df["time_in_cycles"]
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def get_feature_columns():
    return [
        "op_1", "op_2", "op_3",
        "s2", "s3", "s4", "s7", "s8", "s9",
        "s11", "s12", "s13", "s14", "s15",
        "s17", "s20", "s21",
    ]


# ================================================================
# ENGINE-LEVEL 80/20 SPLIT
# ================================================================

def split_train_validation(df, validation_fraction, seed):
    engines = np.asarray(sorted(df["unit_number"].unique()))

    rng = np.random.RandomState(seed)
    rng.shuffle(engines)

    number_validation = max(
        1, int(round(len(engines) * validation_fraction))
    )

    validation_engines = set(engines[:number_validation])
    training_engines = set(engines[number_validation:])

    overlap = training_engines & validation_engines
    assert len(overlap) == 0

    train_df = df[df["unit_number"].isin(training_engines)].copy()
    validation_df = df[df["unit_number"].isin(validation_engines)].copy()

    print("\nENGINE SPLIT")
    print("-" * 60)
    print("Total engines      :", len(engines))
    print("Training engines   :", len(training_engines))
    print("Validation engines :", len(validation_engines))
    print("Overlap            :", len(overlap))

    return train_df, validation_df


# ================================================================
# NUMERIC CHECK
# ================================================================

def check_numeric_array(name, array):
    array = np.asarray(array)
    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())

    print(
        f"{name}: shape={array.shape} | NaN={nan_count} | Inf={inf_count}"
    )

    if nan_count > 0:
        raise ValueError(f"{name} contains NaN.")
    if inf_count > 0:
        raise ValueError(f"{name} contains Inf.")


# ================================================================
# WINDOWING
# ================================================================

def make_sliding_windows(
    df, feature_columns, seq_len, stride, rul_threshold
):
    X, y, ruls, engine_ids = [], [], [], []

    for engine_id, group in df.groupby("unit_number", sort=True):
        group = group.sort_values("time_in_cycles")
        values = group[feature_columns].to_numpy(dtype=np.float32)
        engine_ruls = group["RUL"].to_numpy(dtype=np.float32)

        if len(values) < seq_len:
            continue

        for start in range(0, len(values) - seq_len + 1, stride):
            end = start + seq_len
            current_rul = float(engine_ruls[end - 1])

            X.append(values[start:end])
            y.append(int(current_rul <= rul_threshold))
            ruls.append(current_rul)
            engine_ids.append(int(engine_id))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(ruls, dtype=np.float32),
        np.asarray(engine_ids, dtype=np.int64),
    )


def make_official_test_windows(
    test_df, test_rul, feature_columns, seq_len, rul_threshold
):
    X, y, ruls, engine_ids = [], [], [], []
    engines = sorted(test_df["unit_number"].unique())

    if len(engines) != len(test_rul):
        raise ValueError(
            "Number of test engines and RUL values do not match."
        )

    for index, engine_id in enumerate(engines):
        group = (
            test_df[test_df["unit_number"] == engine_id]
            .sort_values("time_in_cycles")
        )
        values = group[feature_columns].to_numpy(dtype=np.float32)

        if len(values) >= seq_len:
            window = values[-seq_len:]
        else:
            padding_length = seq_len - len(values)
            padding = np.repeat(values[0:1], padding_length, axis=0)
            window = np.concatenate([padding, values], axis=0)

        true_rul = float(test_rul[index])

        X.append(window)
        y.append(int(true_rul <= rul_threshold))
        ruls.append(true_rul)
        engine_ids.append(int(engine_id))

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(ruls, dtype=np.float32),
        np.asarray(engine_ids, dtype=np.int64),
    )


# ================================================================
# SEMANTIC FEATURES: 12 x 17 = 204
# ================================================================

def extract_semantic_features(X):
    T = X.shape[1]
    t = np.arange(T, dtype=np.float32)
    t_mean = t.mean()
    denominator = max(float(np.sum((t - t_mean) ** 2)), 1e-12)

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    minimum = X.min(axis=1)
    maximum = X.max(axis=1)
    value_range = maximum - minimum
    drift = X[:, -1, :] - X[:, 0, :]

    centered_t = t[None, :, None] - t_mean
    centered_x = X - mean[:, None, :]
    slope = (centered_t * centered_x).sum(axis=1) / denominator

    differences = np.diff(X, axis=1)
    diff_abs_mean = np.abs(differences).mean(axis=1)
    diff_std = differences.std(axis=1)
    energy = (X ** 2).mean(axis=1)
    positive_fraction = (differences > 0).mean(axis=1)
    negative_fraction = (differences < 0).mean(axis=1)

    semantic = np.concatenate(
        [
            mean, std, minimum, maximum, value_range, drift,
            slope, diff_abs_mean, diff_std, energy,
            positive_fraction, negative_fraction,
        ],
        axis=1,
    ).astype(np.float32)

    check_numeric_array("Semantic", semantic)

    if semantic.shape[1] != 204:
        raise RuntimeError(
            f"Expected 204 semantic features, got {semantic.shape[1]}."
        )

    return semantic


# ================================================================
# TCN
# ================================================================

class CausalConvBlock(nn.Module):
    def __init__(
        self, input_channels, output_channels, kernel_size, dilation, dropout
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            input_channels, output_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            output_channels, output_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x):
        sequence_length = x.size(2)

        out = self.conv1(x)[:, :, :sequence_length]
        out = self.activation(out)
        out = self.dropout(out)

        out = self.conv2(out)[:, :, :sequence_length]
        out = self.activation(out)
        out = self.dropout(out)

        return self.activation(out + self.residual(x))


class TCN(nn.Module):
    def __init__(self, input_channels, channels, kernel_size, dropout):
        super().__init__()

        blocks = []
        current_channels = input_channels

        for index, output_channels in enumerate(channels):
            blocks.append(
                CausalConvBlock(
                    current_channels,
                    output_channels,
                    kernel_size,
                    dilation=2 ** index,
                    dropout=dropout,
                )
            )
            current_channels = output_channels

        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        return x.permute(0, 2, 1)


class AttentionPool(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.score = nn.Linear(dimension, 1)

    def forward(self, x):
        weights = torch.softmax(self.score(x), dim=1)
        return (weights * x).sum(dim=1)


class TCNBiGRUAttention(nn.Module):
    def __init__(self, input_features, cfg):
        super().__init__()

        self.tcn = TCN(
            input_features,
            cfg["tcn_channels"],
            cfg["tcn_kernel"],
            cfg["tcn_dropout"],
        )

        self.gru = nn.GRU(
            input_size=cfg["tcn_channels"][-1],
            hidden_size=cfg["gru_hidden"],
            num_layers=cfg["gru_layers"],
            dropout=cfg["gru_dropout"] if cfg["gru_layers"] > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        gru_output_dimension = 2 * cfg["gru_hidden"]

        self.attention = AttentionPool(gru_output_dimension)

        self.projection = nn.Sequential(
            nn.Linear(gru_output_dimension, cfg["embed_dim"]),
            nn.LayerNorm(cfg["embed_dim"]),
            nn.GELU(),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Linear(cfg["embed_dim"], 1)

        self.reconstruction = nn.Sequential(
            nn.Linear(cfg["embed_dim"], 64),
            nn.ReLU(),
            nn.Linear(64, input_features),
        )

    def embed(self, x):
        x = self.tcn(x)
        x, _ = self.gru(x)
        x = self.attention(x)
        return self.projection(x)

    def forward(self, x):
        embedding = self.embed(x)
        logits = self.classifier(embedding).squeeze(-1)
        reconstruction = self.reconstruction(embedding)
        return logits, reconstruction, embedding


# ================================================================
# DATA LOADERS
# ================================================================

def create_train_loader(X, y, cfg):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        pin_memory=(DEVICE.type == "cuda"),
    )


def create_validation_loader(X, y, cfg):
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        pin_memory=(DEVICE.type == "cuda"),
    )


# ================================================================
# ENCODER TRAINING
#
# BCE + recon_weight * last-step MSE
# checkpoint = validation F2 @ fixed 0.5
# ================================================================

def train_encoder(X_train, y_train, X_validation, y_validation, cfg):
    seed_everything(cfg["seed"], cfg["deterministic"])

    model = TCNBiGRUAttention(X_train.shape[2], cfg).to(DEVICE)

    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["epochs"],
        eta_min=1e-5,
    )

    train_loader = create_train_loader(X_train, y_train, cfg)
    validation_loader = create_validation_loader(
        X_validation, y_validation, cfg
    )

    use_amp = cfg["use_amp"] and DEVICE.type == "cuda"

    try:
        amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except Exception:
        amp_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_state = copy.deepcopy(model.state_dict())
    best_f2 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()

        total_loss = 0.0
        total_bce = 0.0
        total_reconstruction = 0.0
        sample_count = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            if not torch.isfinite(xb).all():
                raise RuntimeError("Training X contains NaN/Inf.")

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, reconstruction, _ = model(xb)
                    classification_loss = bce(logits, yb)
                    reconstruction_loss = mse(reconstruction, xb[:, -1, :])
                    loss = (
                        classification_loss
                        + cfg["recon_weight"] * reconstruction_loss
                    )
            else:
                logits, reconstruction, _ = model(xb)
                classification_loss = bce(logits, yb)
                reconstruction_loss = mse(reconstruction, xb[:, -1, :])
                loss = (
                    classification_loss
                    + cfg["recon_weight"] * reconstruction_loss
                )

            if not torch.isfinite(logits).all():
                raise RuntimeError("Encoder logits became NaN/Inf.")
            if not torch.isfinite(reconstruction).all():
                raise RuntimeError("Reconstruction became NaN/Inf.")
            if not torch.isfinite(loss):
                raise RuntimeError("Total loss became NaN/Inf.")

            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            amp_scaler.step(optimizer)
            amp_scaler.update()

            batch_size = xb.size(0)

            total_loss += loss.item() * batch_size
            total_bce += classification_loss.item() * batch_size
            total_reconstruction += reconstruction_loss.item() * batch_size
            sample_count += batch_size

        model.eval()
        validation_probabilities = []
        validation_labels = []

        with torch.no_grad():
            for xb, yb in validation_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                logits, _, _ = model(xb)
                probabilities = torch.sigmoid(logits)

                validation_probabilities.extend(
                    probabilities.cpu().numpy()
                )
                validation_labels.extend(yb.numpy())

        validation_probabilities = np.asarray(validation_probabilities)
        validation_labels = np.asarray(validation_labels, dtype=np.int64)

        validation_predictions = (
            validation_probabilities >= 0.5
        ).astype(np.int64)

        val_precision = precision_score(
            validation_labels, validation_predictions, zero_division=0
        )
        val_recall = recall_score(
            validation_labels, validation_predictions, zero_division=0
        )
        val_f1 = f1_score(
            validation_labels, validation_predictions, zero_division=0
        )
        val_f2 = fbeta_score(
            validation_labels, validation_predictions,
            beta=2, zero_division=0
        )

        average_loss = total_loss / sample_count
        average_bce = total_bce / sample_count
        average_reconstruction = total_reconstruction / sample_count

        history.append({
            "Epoch": epoch,
            "Loss": average_loss,
            "BCE": average_bce,
            "Reconstruction": average_reconstruction,
            "ValPrecision": val_precision,
            "ValRecall": val_recall,
            "ValF1": val_f1,
            "ValF2": val_f2,
        })

        print(
            f"Epoch {epoch:03d} | "
            f"Loss={average_loss:.5f} | "
            f"BCE={average_bce:.5f} | "
            f"Recon={average_reconstruction:.5f} | "
            f"ValP={val_precision:.4f} | "
            f"ValR={val_recall:.4f} | "
            f"ValF1={val_f1:.4f} | "
            f"ValF2={val_f2:.4f}"
        )

        if val_f2 > best_f2 + cfg["min_delta"]:
            best_f2 = val_f2
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            print("\nEarly stopping:", epoch)
            break

        scheduler.step()

    model.load_state_dict(best_state)

    print("\n" + "=" * 80)
    print("BEST ENCODER EPOCH:", best_epoch)
    print("BEST VALIDATION F2:", f"{best_f2:.4f}")
    print("=" * 80)

    return model, pd.DataFrame(history)


# ================================================================
# EMBEDDINGS / HYBRID FEATURES
# ================================================================

@torch.no_grad()
def extract_embeddings(model, X, batch_size=512):
    model.eval()

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []

    for (xb,) in loader:
        xb = xb.to(DEVICE)
        embedding = model.embed(xb)
        embeddings.append(embedding.cpu().numpy())

    embeddings = np.vstack(embeddings).astype(np.float32)
    check_numeric_array("Encoder embeddings", embeddings)
    return embeddings


def build_hybrid_features(model, X):
    embedding = extract_embeddings(model, X)
    semantic = extract_semantic_features(X)

    hybrid = np.concatenate(
        [embedding, semantic], axis=1
    ).astype(np.float32)

    print(
        "Embedding:", embedding.shape,
        "| Semantic:", semantic.shape,
        "| Hybrid:", hybrid.shape
    )

    assert embedding.shape[1] == 128
    assert semantic.shape[1] == 204
    assert hybrid.shape[1] == 332

    check_numeric_array("Hybrid features", hybrid)
    return hybrid


# ================================================================
# XGBOOST RANDOMIZED SEARCH -- F2
# ================================================================

F2_SCORER = make_scorer(
    fbeta_score, beta=2, zero_division=0
)


def randomized_xgb_search(
    X_train, y_train, train_engine_ids, cfg
):
    print("\n" + "=" * 100)
    print("XGBOOST RANDOMIZED SEARCH")
    print("TRAIN ONLY | GROUPKFOLD | F2")
    print("=" * 100)

    print("Training samples:", len(y_train))
    print("Training engines:", len(np.unique(train_engine_ids)))
    print("Iterations:", cfg["random_search_iterations"])
    print("CV folds:", cfg["search_cv_folds"])
    print(
        "Total CV fits:",
        cfg["random_search_iterations"] * cfg["search_cv_folds"]
    )

    group_cv = GroupKFold(n_splits=cfg["search_cv_folds"])

    estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=cfg["search_xgb_estimators"],
        tree_method="hist",
        scale_pos_weight=1.0,
        random_state=cfg["seed"],
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=cfg["xgb_param_distributions"],
        n_iter=cfg["random_search_iterations"],
        scoring=F2_SCORER,
        cv=group_cv,
        random_state=cfg["seed"],
        n_jobs=cfg["search_n_jobs"],
        verbose=2,
        refit=True,
        return_train_score=False,
    )

    search.fit(X_train, y_train, groups=train_engine_ids)

    print("\n" + "=" * 80)
    print("BEST GROUPKFOLD F2:", f"{search.best_score_:.4f}")
    print("\nBEST PARAMETERS")

    for key, value in search.best_params_.items():
        print(f"{key}: {value}")

    print("=" * 80)

    return search.best_params_, pd.DataFrame(search.cv_results_)


# ================================================================
# FINAL XGBOOST ENSEMBLE
# ================================================================

def train_xgb_ensemble(
    X_train, y_train,
    X_validation, y_validation,
    best_params, cfg
):
    models = []

    seeds = [
        cfg["seed"] + 1009 * k
        for k in range(cfg["ensemble_size"])
    ]

    dtrain = DMatrix(X_train, label=y_train)
    dvalidation = DMatrix(X_validation, label=y_validation)

    print("\n" + "=" * 100)
    print("FINAL XGBOOST ENSEMBLE")
    print("=" * 100)

    for index, seed in enumerate(seeds, start=1):
        print(
            f"\nModel {index}/{cfg['ensemble_size']} | seed={seed}"
        )

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "max_depth": int(best_params["max_depth"]),
            "eta": float(best_params["learning_rate"]),
            "min_child_weight": float(best_params["min_child_weight"]),
            "subsample": float(best_params["subsample"]),
            "colsample_bytree": float(best_params["colsample_bytree"]),
            "alpha": float(best_params["reg_alpha"]),
            "lambda": float(best_params["reg_lambda"]),
            "scale_pos_weight": 1.0,
            "seed": seed,
            "nthread": -1,
        }

        model = xgb_train(
            params,
            dtrain,
            num_boost_round=cfg["xgb_rounds"],
            evals=[(dvalidation, "validation")],
            early_stopping_rounds=cfg["xgb_es_rounds"],
            verbose_eval=False,
        )

        print("Best iteration:", model.best_iteration)
        models.append(model)

    return models


# ================================================================
# ENSEMBLE PREDICTION
# ================================================================

def predict_xgb_ensemble(models, X):
    data = DMatrix(X)
    all_predictions = []

    for model in models:
        best_iteration = getattr(model, "best_iteration", None)

        if best_iteration is not None:
            prediction = model.predict(
                data,
                iteration_range=(0, best_iteration + 1),
            )
        else:
            prediction = model.predict(data)

        all_predictions.append(prediction)

    return np.mean(np.vstack(all_predictions), axis=0)


# ================================================================
# THRESHOLD SELECTION
#
# VALIDATION ONLY
# Precision-Recall candidate thresholds
# Candidates are first restricted to those meeting a minimum
# validation precision (min_precision), then the max-F1 candidate
# among those survivors is selected.
# Tie = higher Recall -> Precision -> Threshold
#
# If NO candidate threshold reaches min_precision, we fall back to
# the full unconstrained candidate pool and flag the fallback, so
# the dataset is never silently mis-thresholded and the fallback
# is visible in logs / saved metadata instead.
#
# F2 is reporting only.
# ================================================================

def select_threshold_by_pr_f1(y_true, probabilities, min_precision=0.0):
    y_true = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(y_true) != len(probabilities):
        raise ValueError(
            "y_true and probabilities must have the same number of samples."
        )

    if len(np.unique(y_true)) < 2:
        raise ValueError(
            "Validation set must contain both classes for PR threshold selection."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError(
            "Validation probabilities contain NaN/Inf."
        )

    precision, recall, thresholds = precision_recall_curve(
        y_true, probabilities
    )

    if len(thresholds) == 0:
        raise RuntimeError(
            "precision_recall_curve produced no candidate thresholds."
        )

    # Last precision/recall point has no associated threshold.
    precision_t = precision[:-1]
    recall_t = recall[:-1]

    f1 = (
        2.0 * precision_t * recall_t
        / (precision_t + recall_t + 1e-12)
    )

    f2 = (
        5.0 * precision_t * recall_t
        / (4.0 * precision_t + recall_t + 1e-12)
    )

    table = pd.DataFrame({
        "Threshold": thresholds.astype(np.float64),
        "Precision": precision_t.astype(np.float64),
        "Recall": recall_t.astype(np.float64),
        "F1": f1.astype(np.float64),
        "F2": f2.astype(np.float64),
    })

    # Restrict candidates to those meeting the precision floor.
    eligible = table[table["Precision"] >= min_precision]

    fallback_used = False
    if len(eligible) == 0:
        # No threshold on this validation split reaches the floor.
        # Fall back to the unconstrained pool rather than raising,
        # but mark it so the caller can log/report the fallback.
        eligible = table
        fallback_used = True

    selected = (
        eligible
        .sort_values(
            ["F1", "Recall", "Precision", "Threshold"],
            ascending=[False, False, False, False],
        )
        .iloc[0]
        .copy()
    )

    selected["PrecisionFloor"] = min_precision
    selected["PrecisionFloorFallback"] = fallback_used

    return selected, table


# ================================================================
# METRICS
# ================================================================

def calculate_metrics(y_true, probabilities, threshold):
    prediction = (probabilities >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true, prediction, labels=[0, 1]
    ).ravel()

    precision = precision_score(
        y_true, prediction, zero_division=0
    )
    recall = recall_score(
        y_true, prediction, zero_division=0
    )
    f1 = f1_score(
        y_true, prediction, zero_division=0
    )
    f2 = fbeta_score(
        y_true, prediction, beta=2, zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    try:
        roc_auc = roc_auc_score(y_true, probabilities)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, probabilities)
    except ValueError:
        pr_auc = np.nan

    return {
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "F2": f2,
        "Specificity": specificity,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


# ================================================================
# TEST ENGINE DIAGNOSTICS
# ================================================================

def build_engine_diagnostics(
    engine_ids, ruls, y_true, probabilities, threshold
):
    predictions = (probabilities >= threshold).astype(np.int64)
    rows = []

    for engine_id, rul, label, probability, prediction in zip(
        engine_ids, ruls, y_true, probabilities, predictions
    ):
        if label == 1 and prediction == 1:
            result = "TP"
        elif label == 1 and prediction == 0:
            result = "FN"
        elif label == 0 and prediction == 1:
            result = "FP"
        else:
            result = "TN"

        rows.append({
            "EngineId": int(engine_id),
            "TrueRUL": float(rul),
            "TrueLabel": int(label),
            "Probability": float(probability),
            "Threshold": float(threshold),
            "Prediction": int(prediction),
            "Result": result,
            "ThresholdMargin": float(probability - threshold),
        })

    return pd.DataFrame(rows)


def calculate_rul_band_metrics(diagnostics):
    positive = diagnostics[
        diagnostics["TrueLabel"] == 1
    ].copy()

    bands = [
        ("RUL <= 10", positive[positive["TrueRUL"] <= 10]),
        (
            "RUL 11-20",
            positive[
                (positive["TrueRUL"] >= 11)
                & (positive["TrueRUL"] <= 20)
            ],
        ),
        (
            "RUL 21-25",
            positive[
                (positive["TrueRUL"] >= 21)
                & (positive["TrueRUL"] <= 25)
            ],
        ),
        (
            "RUL 26-30",
            positive[
                (positive["TrueRUL"] >= 26)
                & (positive["TrueRUL"] <= 30)
            ],
        ),
    ]

    rows = []

    for name, group in bands:
        if len(group) == 0:
            continue

        tp = int((group["Prediction"] == 1).sum())
        fn = int((group["Prediction"] == 0).sum())

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else np.nan
        )

        rows.append({
            "RUL_Band": name,
            "Count": len(group),
            "TP": tp,
            "FN": fn,
            "Recall": recall,
            "MeanProbability": group["Probability"].mean(),
        })

    return pd.DataFrame(rows)


# ================================================================
# RUN ONE DATASET
# ================================================================

def run_dataset(fd, cfg):
    print("\n\n" + "#" * 120)
    print(f"RUNNING {fd}")
    print("#" * 120)

    start_time = time.perf_counter()
    seed_everything(cfg["seed"], cfg["deterministic"])

    full_train_df, official_test_df, official_test_rul = load_cmapss(fd)
    full_train_df = add_train_rul(full_train_df)

    feature_columns = get_feature_columns()

    print("\nInput variables:", len(feature_columns))
    print(feature_columns)
    assert len(feature_columns) == 17

    # 80/20 engine-disjoint split
    train_df, validation_df = split_train_validation(
        full_train_df,
        cfg["val_engine_frac"],
        cfg["seed"],
    )

    # Scaler: fit TRAIN only
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_df[feature_columns].to_numpy(dtype=np.float64)
    )
    validation_scaled = scaler.transform(
        validation_df[feature_columns].to_numpy(dtype=np.float64)
    )
    official_test_scaled = scaler.transform(
        official_test_df[feature_columns].to_numpy(dtype=np.float64)
    )

    check_numeric_array("Scaled TRAIN", train_scaled)
    check_numeric_array("Scaled VALIDATION", validation_scaled)
    check_numeric_array("Scaled OFFICIAL TEST", official_test_scaled)

    train_df.loc[:, feature_columns] = train_scaled
    validation_df.loc[:, feature_columns] = validation_scaled

    official_test_df = official_test_df.copy()
    official_test_df.loc[:, feature_columns] = official_test_scaled

    # Windows
    X_train, y_train, train_ruls, train_engine_ids = make_sliding_windows(
        train_df,
        feature_columns,
        cfg["seq_len"],
        cfg["stride"],
        cfg["rul_threshold"],
    )

    (
        X_validation,
        y_validation,
        validation_ruls,
        validation_engine_ids,
    ) = make_sliding_windows(
        validation_df,
        feature_columns,
        cfg["seq_len"],
        cfg["stride"],
        cfg["rul_threshold"],
    )

    X_test, y_test, test_ruls, test_engine_ids = make_official_test_windows(
        official_test_df,
        official_test_rul,
        feature_columns,
        cfg["seq_len"],
        cfg["rul_threshold"],
    )

    print("\nWINDOW SUMMARY")
    print("-" * 80)
    print(
        "TRAIN:", X_train.shape,
        "| positive rate:", f"{y_train.mean():.4f}"
    )
    print(
        "VALIDATION:", X_validation.shape,
        "| positive rate:", f"{y_validation.mean():.4f}"
    )
    print(
        "OFFICIAL TEST:", X_test.shape,
        "| positive rate:", f"{y_test.mean():.4f}"
    )
    print(
        "Official test engines:",
        len(np.unique(test_engine_ids))
    )

    check_numeric_array("X_train", X_train)
    check_numeric_array("X_validation", X_validation)
    check_numeric_array("X_test", X_test)

    # Encoder
    print("\n" + "=" * 100)
    print("TRAINING TCN-BiGRU-ATTENTION ENCODER")
    print("Checkpoint criterion: VALIDATION F2 @ 0.5")
    print("=" * 100)

    encoder, encoder_history = train_encoder(
        X_train,
        y_train,
        X_validation,
        y_validation,
        cfg,
    )

    # Hybrid features
    print("\n" + "=" * 100)
    print("BUILDING TRAIN HYBRID FEATURES")
    print("=" * 100)
    X_train_hybrid = build_hybrid_features(encoder, X_train)

    print("\n" + "=" * 100)
    print("BUILDING VALIDATION HYBRID FEATURES")
    print("=" * 100)
    X_validation_hybrid = build_hybrid_features(
        encoder, X_validation
    )

    print("\n" + "=" * 100)
    print("BUILDING OFFICIAL TEST HYBRID FEATURES")
    print("=" * 100)
    X_test_hybrid = build_hybrid_features(encoder, X_test)

    # XGB search -- TRAIN only
    best_params, search_results = randomized_xgb_search(
        X_train_hybrid,
        y_train,
        train_engine_ids,
        cfg,
    )

    # Final XGB ensemble -- validation for early stopping
    xgb_models = train_xgb_ensemble(
        X_train_hybrid,
        y_train,
        X_validation_hybrid,
        y_validation,
        best_params,
        cfg,
    )

    # Validation ensemble probabilities
    validation_probabilities = predict_xgb_ensemble(
        xgb_models,
        X_validation_hybrid,
    )

    check_numeric_array(
        "Validation ensemble probabilities",
        validation_probabilities,
    )

    # ------------------------------------------------------------
    # VALIDATION PR CURVE -> MAX F1
    # ------------------------------------------------------------

    selected_threshold, threshold_table = select_threshold_by_pr_f1(
        y_validation,
        validation_probabilities,
        min_precision=cfg["min_precision_floor"],
    )

    threshold = float(selected_threshold["Threshold"])
    precision_floor_fallback = bool(
        selected_threshold["PrecisionFloorFallback"]
    )

    print("\n" + "=" * 100)
    print("SELECTED VALIDATION THRESHOLD")
    print(
        f"Selection criterion: PR CURVE -> MAX F1 "
        f"(precision floor = {cfg['min_precision_floor']:.2f})"
    )
    print("=" * 100)

    if precision_floor_fallback:
        print(
            f"WARNING: no validation threshold for {fd} reached "
            f"precision >= {cfg['min_precision_floor']:.2f}; "
            "fell back to the unconstrained max-F1 candidate."
        )

    print(f"Threshold : {threshold:.6f}")
    print(f"Precision : {selected_threshold['Precision']:.4f}")
    print(f"Recall    : {selected_threshold['Recall']:.4f}")
    print(f"F1        : {selected_threshold['F1']:.4f}")
    print(f"F2        : {selected_threshold['F2']:.4f}")

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
        threshold,
    )

    # ============================================================
    # DEVELOPMENT IS NOW FROZEN
    # ============================================================

    # Official test
    test_probabilities = predict_xgb_ensemble(
        xgb_models,
        X_test_hybrid,
    )

    check_numeric_array(
        "Official test probabilities",
        test_probabilities,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        threshold,
    )

    diagnostics = build_engine_diagnostics(
        test_engine_ids,
        test_ruls,
        y_test,
        test_probabilities,
        threshold,
    )

    rul_bands = calculate_rul_band_metrics(diagnostics)

    # Final results
    print("\n\n" + "#" * 100)
    print(f"{fd} OFFICIAL TEST RESULTS")
    print("#" * 100)

    print(f"Threshold   : {threshold:.6f}")
    print(f"Precision   : {test_metrics['Precision']:.4f}")
    print(f"Recall      : {test_metrics['Recall']:.4f}")
    print(f"F1          : {test_metrics['F1']:.4f}")
    print(f"F2          : {test_metrics['F2']:.4f}")
    print(f"Specificity : {test_metrics['Specificity']:.4f}")
    print(f"ROC-AUC     : {test_metrics['ROC_AUC']:.4f}")
    print(f"PR-AUC      : {test_metrics['PR_AUC']:.4f}")

    print(
        "\n"
        f"TP={test_metrics['TP']} | "
        f"FP={test_metrics['FP']} | "
        f"FN={test_metrics['FN']} | "
        f"TN={test_metrics['TN']}"
    )

    false_negatives = (
        diagnostics[diagnostics["Result"] == "FN"]
        .copy()
        .sort_values("TrueRUL", ascending=False)
    )

    print("\n" + "=" * 100)
    print("FALSE NEGATIVES")
    print("=" * 100)

    if len(false_negatives) == 0:
        print("No false negatives.")
    else:
        print(false_negatives.round(4).to_string(index=False))

    false_positives = (
        diagnostics[diagnostics["Result"] == "FP"]
        .copy()
        .sort_values("Probability", ascending=False)
    )

    print("\n" + "=" * 100)
    print("FALSE POSITIVES")
    print("=" * 100)

    if len(false_positives) == 0:
        print("No false positives.")
    else:
        print(false_positives.round(4).to_string(index=False))

    print("\n" + "=" * 100)
    print("TEST RECALL BY RUL BAND")
    print("=" * 100)

    if len(rul_bands) > 0:
        print(rul_bands.round(4).to_string(index=False))

    # Save
    dataset_dir = ARTIFACT_DIR / fd
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if cfg["save_artifacts"]:
        encoder_history.to_csv(
            dataset_dir / "encoder_history.csv", index=False
        )

        search_results.to_csv(
            dataset_dir / "xgb_random_search.csv", index=False
        )

        threshold_table.to_csv(
            dataset_dir / "validation_pr_thresholds.csv", index=False
        )

        # Save validation probabilities as an audit artifact.
        pd.DataFrame({
            "EngineId": validation_engine_ids,
            "TrueRUL": validation_ruls,
            "TrueLabel": y_validation,
            "Probability": validation_probabilities,
            "Prediction": (
                validation_probabilities >= threshold
            ).astype(np.int64),
        }).to_csv(
            dataset_dir / "validation_predictions.csv",
            index=False,
        )

        diagnostics.to_csv(
            dataset_dir / "official_test_engine_diagnostics.csv",
            index=False,
        )

        rul_bands.to_csv(
            dataset_dir / "official_test_rul_bands.csv",
            index=False,
        )

        torch.save(
            encoder.state_dict(),
            dataset_dir / "encoder.pt",
        )

        with open(dataset_dir / "scaler.pkl", "wb") as file:
            pickle.dump(scaler, file)

        for index, model in enumerate(xgb_models, start=1):
            model.save_model(
                dataset_dir / f"xgb_{index}.json"
            )

        metadata = {
            "dataset": fd,
            "seed": cfg["seed"],
            "train_fraction": 0.80,
            "validation_fraction": 0.20,
            "seq_len": cfg["seq_len"],
            "stride": cfg["stride"],
            "rul_threshold": cfg["rul_threshold"],
            "input_features": feature_columns,

            "encoder_checkpoint":
                "validation_F2_at_fixed_threshold_0.5",

            "xgb_hyperparameter_search":
                "train_only_3_fold_engine_GroupKFold_F2",

            "threshold_selection":
                "validation_precision_recall_curve_max_F1_precision_floor",

            "threshold_grid_used": False,
            "probability_calibration": False,
            "minimum_recall_constraint": False,
            "minimum_precision_floor": cfg["min_precision_floor"],
            "minimum_precision_floor_fallback": precision_floor_fallback,

            "best_xgb_params": {
                key: (
                    value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in best_params.items()
            },

            "selected_threshold": threshold,

            "validation_precision": float(
                validation_metrics["Precision"]
            ),
            "validation_recall": float(
                validation_metrics["Recall"]
            ),
            "validation_f1": float(
                validation_metrics["F1"]
            ),
            "validation_f2": float(
                validation_metrics["F2"]
            ),
        }

        with open(dataset_dir / "metadata.json", "w") as file:
            json.dump(metadata, file, indent=4)

    elapsed = time.perf_counter() - start_time

    print("\nRuntime:", f"{elapsed / 60:.2f} minutes")

    result = {
        "Dataset": fd,
        "Threshold": threshold,
        "PrecisionFloor": cfg["min_precision_floor"],
        "PrecisionFloorFallback": precision_floor_fallback,

        "ValPrecision": validation_metrics["Precision"],
        "ValRecall": validation_metrics["Recall"],
        "ValF1": validation_metrics["F1"],
        "ValF2": validation_metrics["F2"],

        "TestPrecision": test_metrics["Precision"],
        "TestRecall": test_metrics["Recall"],
        "TestF1": test_metrics["F1"],
        "TestF2": test_metrics["F2"],
        "Specificity": test_metrics["Specificity"],
        "ROC_AUC": test_metrics["ROC_AUC"],
        "PR_AUC": test_metrics["PR_AUC"],

        "TP": test_metrics["TP"],
        "FP": test_metrics["FP"],
        "FN": test_metrics["FN"],
        "TN": test_metrics["TN"],
    }

    del encoder
    del xgb_models
    del X_train_hybrid
    del X_validation_hybrid
    del X_test_hybrid

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ================================================================
# RUN ALL DATASETS
# ================================================================

def main():
    all_results = []
    overall_start = time.perf_counter()

    for fd in CFG["datasets"]:
        result = run_dataset(fd, CFG)
        all_results.append(result)

        pd.DataFrame(all_results).to_csv(
            ARTIFACT_DIR / "partial_results.csv",
            index=False,
        )

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        ARTIFACT_DIR / "final_results.csv",
        index=False,
    )

    print("\n\n" + "#" * 130)
    print("FINAL OFFICIAL TEST RESULTS")
    print("#" * 130)

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )

    # Paper table
    paper_table = results_df[
        [
            "Dataset",
            "Threshold",
            "TestPrecision",
            "TestRecall",
            "TestF1",
            "TestF2",
            "Specificity",
            "ROC_AUC",
            "PR_AUC",
        ]
    ].copy()

    paper_table.columns = [
        "Subset",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "F2",
        "Specificity",
        "ROC-AUC",
        "PR-AUC",
    ]

    latex = paper_table.to_latex(
        index=False,
        float_format=lambda x: f"{x:.4f}",
        escape=False,
    )

    with open(
        ARTIFACT_DIR / "final_results_table.tex",
        "w",
    ) as file:
        file.write(latex)

    overall_elapsed = time.perf_counter() - overall_start

    print(
        "\nTotal runtime:",
        f"{overall_elapsed / 60:.2f} minutes"
    )

    print("\nArtifacts saved to:")
    print(ARTIFACT_DIR)


if __name__ == "__main__":
    main()
