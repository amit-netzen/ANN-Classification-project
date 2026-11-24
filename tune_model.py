"""Tune the bank churn ANN, evaluate the winner, and save app-ready artifacts."""

from __future__ import annotations

import itertools
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Churn_Modelling.csv"
SEED = 42
MAX_EPOCHS = 35
PATIENCE = 4


def make_one_hot_encoder():
    """Support both current and older scikit-learn constructor names."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def encode_features(frame, gender_encoder, geography_encoder):
    gender = gender_encoder.transform(frame["Gender"])
    geography = geography_encoder.transform(frame[["Geography"]])
    encoded = frame.drop(columns=["RowNumber", "CustomerId", "Surname", "Geography", "Gender"])
    encoded.insert(1, "Gender", gender)
    geography_frame = pd.DataFrame(
        geography,
        columns=geography_encoder.get_feature_names_out(["Geography"]),
        index=frame.index,
    )
    return pd.concat([encoded, geography_frame], axis=1)


def build_model(input_size, units, depth, dropout, learning_rate):
    model = Sequential([Input(shape=(input_size,))])
    for _ in range(depth):
        model.add(Dense(units, activation="relu"))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc"), "accuracy"],
    )
    return model


def main():
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.keras.backend.clear_session()

    data = pd.read_csv(DATA_PATH)
    target = data["Exited"].astype("int32")
    raw_features = data.drop(columns="Exited")
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        raw_features,
        target,
        test_size=0.2,
        random_state=SEED,
        stratify=target,
    )
    x_train_raw, x_val_raw, y_train, y_val = train_test_split(
        x_train_raw,
        y_train,
        test_size=0.2,
        random_state=SEED,
        stratify=y_train,
    )

    gender_encoder = LabelEncoder().fit(x_train_raw["Gender"])
    geography_encoder = make_one_hot_encoder().fit(x_train_raw[["Geography"]])
    x_train = encode_features(x_train_raw, gender_encoder, geography_encoder)
    x_val = encode_features(x_val_raw, gender_encoder, geography_encoder)
    x_test = encode_features(x_test_raw, gender_encoder, geography_encoder)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # A compact, reproducible grid tests width, depth, dropout, optimizer rate,
    # and batch size. Each trial uses the same training/validation rows.
    configs = list(itertools.product(
        [32, 64, 128],
        [1, 2],
        [0.0, 0.2],
        [0.001, 0.0003],
        [64, 128],
    ))
    rng = np.random.default_rng(SEED)
    chosen = rng.choice(len(configs), size=8, replace=False)
    trials = [configs[i] for i in chosen]
    results = []
    best_auc = -1.0
    best_loss = float("inf")
    best_model = None
    best_config = None

    for trial_number, (units, depth, dropout, learning_rate, batch_size) in enumerate(trials, 1):
        print(
            f"Starting trial {trial_number:02d}/{len(trials)}: "
            f"units={units}, layers={depth}, dropout={dropout}, "
            f"learning_rate={learning_rate}, batch_size={batch_size}",
            flush=True,
        )
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED + trial_number)
        model = build_model(x_train_scaled.shape[1], units, depth, dropout, learning_rate)
        stopper = EarlyStopping(
            monitor="val_auc", mode="max", patience=PATIENCE, restore_best_weights=True
        )
        history = model.fit(
            x_train_scaled,
            y_train,
            validation_data=(x_val_scaled, y_val),
            epochs=MAX_EPOCHS,
            batch_size=batch_size,
            callbacks=[stopper],
            verbose=0,
        )
        evaluation = model.evaluate(x_val_scaled, y_val, verbose=0, return_dict=True)
        config = {
            "units": units,
            "layers": depth,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        result = {
            "trial": trial_number,
            **config,
            "epochs_trained": len(history.history["loss"]),
            "val_auc": float(evaluation["auc"]),
            "val_accuracy": float(evaluation["accuracy"]),
            "val_loss": float(evaluation["loss"]),
        }
        results.append(result)
        print(
            f"Trial {trial_number:02d}/{len(trials)}: "
            f"val_auc={result['val_auc']:.4f}, val_accuracy={result['val_accuracy']:.4f}, "
            f"{config}, epochs={result['epochs_trained']}"
            ,
            flush=True,
        )
        if (result["val_auc"], -result["val_loss"]) > (best_auc, -best_loss):
            best_auc = result["val_auc"]
            best_loss = result["val_loss"]
            best_model = model
            best_config = config

    assert best_model is not None and best_config is not None
    test_probability = np.asarray(best_model.predict(x_test_scaled, verbose=0)).reshape(-1)
    test_prediction = (test_probability >= 0.5).astype("int32")
    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_probability)),
        "accuracy": float(accuracy_score(y_test, test_prediction)),
        "precision": float(precision_score(y_test, test_prediction, zero_division=0)),
        "recall": float(recall_score(y_test, test_prediction, zero_division=0)),
        "f1": float(f1_score(y_test, test_prediction, zero_division=0)),
    }

    # Save in the filenames consumed by app.py and prediction.ipynb.
    best_model.save(ROOT / "churn_model.h5")
    for filename, artifact in (
        ("scaler.pkl", scaler),
        ("label_encoder.pkl", gender_encoder),
        ("onehot_encoder.pkl", geography_encoder),
    ):
        with (ROOT / filename).open("wb") as file:
            pickle.dump(artifact, file)
    pd.DataFrame(results).sort_values("val_auc", ascending=False).to_csv(
        ROOT / "tuning_results.csv", index=False
    )
    report = {
        "selection_metric": "validation ROC AUC",
        "validation_auc": best_auc,
        "best_parameters": best_config,
        "test_metrics_at_threshold_0.5": test_metrics,
        "seed": SEED,
        "trials": len(trials),
    }
    with (ROOT / "tuning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nBest configuration:", json.dumps(best_config, indent=2))
    print(f"Best validation ROC AUC: {best_auc:.4f}")
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))
    print("Saved churn_model.h5, preprocessing artifacts, tuning_results.csv, and tuning_summary.json")


if __name__ == "__main__":
    main()
