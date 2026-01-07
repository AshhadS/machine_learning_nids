# train_kdd_nids.py
# Train a self-adapting NIDS model (online learning) on KDD Cup 1999 data.
# Model: SGDClassifier (logistic regression) + preprocessing (scaling + one-hot encoding)
#
# Run:
#   python train_kdd_nids.py --data "kddcup.data_10_percent"
# or (gz file):
#   python train_kdd_nids.py --data "kddcup.data.gz"
#
# Output:
#   nids_kdd_pipeline.joblib   (full preprocessing + model)
#   nids_label_encoder.joblib  (label encoder for attack labels)

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix


KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate','label'
]

ATTACK_GROUPS = {
    # DoS
    "back":"dos","land":"dos","neptune":"dos","pod":"dos","smurf":"dos","teardrop":"dos",
    "mailbomb":"dos","processtable":"dos","udpstorm":"dos","apache2":"dos","worm":"dos",
    # Probe
    "ipsweep":"probe","nmap":"probe","portsweep":"probe","satan":"probe","mscan":"probe","saint":"probe",
    # R2L
    "guess_passwd":"r2l","ftp_write":"r2l","imap":"r2l","phf":"r2l","multihop":"r2l","warezmaster":"r2l",
    "warezclient":"r2l","spy":"r2l","xlock":"r2l","xsnoop":"r2l","snmpguess":"r2l","snmpgetattack":"r2l",
    "httptunnel":"r2l","sendmail":"r2l","named":"r2l",
    # U2R
    "buffer_overflow":"u2r","loadmodule":"u2r","perl":"u2r","rootkit":"u2r","ps":"u2r",
    "sqlattack":"u2r","xterm":"u2r"
}

def map_to_category(label: str) -> str:
    """
    Convert raw KDD labels into 5 classes: normal, dos, probe, r2l, u2r
    KDD labels often end with '.' (e.g., 'neptune.'). We normalize that.
    """
    lbl = label.strip().lower().rstrip(".")
    if lbl == "normal":
        return "normal"
    return ATTACK_GROUPS.get(lbl, "other")  # "other" catches any unknown labels


def load_kdd(path: str) -> pd.DataFrame:
    compression = "gzip" if path.lower().endswith(".gz") else None
    df = pd.read_csv(path, header=None, names=KDD_COLUMNS, compression=compression)
    return df


def build_pipeline():
    categorical_features = ['protocol_type', 'service', 'flag']
    numeric_features = [c for c in KDD_COLUMNS if c not in categorical_features + ['label']]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )

    # Self-adapting / online-friendly classifier (supports partial_fit when needed later)
    clf = SGDClassifier(
        loss="log_loss",      # logistic regression
        penalty="l2",
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        random_state=42
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to KDD file (e.g., kddcup.data_10_percent or .gz)")
    parser.add_argument("--output_model", default="nids_kdd_pipeline.joblib")
    parser.add_argument("--output_labels", default="nids_label_encoder.joblib")
    parser.add_argument("--use_categories", action="store_true",
                        help="If set, predicts categories: normal/dos/probe/r2l/u2r/other instead of raw labels")
    parser.add_argument("--sample", type=int, default=0,
                        help="Optional: take a random sample of N rows (0 = use all)")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")

    print(f"Loading: {args.data}")
    df = load_kdd(args.data)

    if args.sample and args.sample > 0:
        df = df.sample(args.sample, random_state=42).reset_index(drop=True)
        print(f"Using sample of {args.sample} rows")

    # Optional: convert raw labels to broader categories (recommended for cleaner results)
    if args.use_categories:
        df["label"] = df["label"].astype(str).apply(map_to_category)

    # Split X/y
    X = df.drop("label", axis=1)
    y = df["label"].astype(str)

    # Encode labels
    label_enc = LabelEncoder()
    y_enc = label_enc.fit_transform(y)

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.30, random_state=42, stratify=y_enc
    )

    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("\nEvaluating...")
    preds = pipeline.predict(X_test)
    print(classification_report(y_test, preds, target_names=label_enc.classes_))

    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    print(f"\nSaving model -> {args.output_model}")
    joblib.dump(pipeline, args.output_model)

    print(f"Saving label encoder -> {args.output_labels}")
    joblib.dump(label_enc, args.output_labels)

    print("\nDone ✅")
    print("Next step (realtime): load the saved pipeline + encoder and call pipeline.predict() on new traffic rows.")


if __name__ == "__main__":
    main()
