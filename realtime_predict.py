import joblib
import pandas as pd

# Must match the 41 feature columns used during training (no 'label' here)
KDD_COLUMNS = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
    'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
    'root_shell','su_attempted','num_root','num_file_creations','num_shells',
    'num_access_files','num_outbound_cmds','is_host_login','is_guest_login',
    'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
    'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
    'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
    'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate'
]

# 1) Load saved model + label encoder
pipeline = joblib.load("nids_kdd_pipeline.joblib")
label_enc = joblib.load("nids_label_encoder.joblib")

# 2) Load new traffic rows (must contain the 41 columns)
# Example: incoming.csv should have either headers matching KDD_COLUMNS,
# or be 41 columns without headers.
try:
    incoming = pd.read_csv("incoming.csv")
    if set(KDD_COLUMNS).issubset(incoming.columns):
        X_new = incoming[KDD_COLUMNS]
    else:
        raise ValueError
except Exception:
    incoming = pd.read_csv("incoming.csv", header=None)
    incoming.columns = KDD_COLUMNS
    X_new = incoming

# 3) Predict
pred_ids = pipeline.predict(X_new)                     # numeric class ids
pred_labels = label_enc.inverse_transform(pred_ids)    # readable labels

# 4) Print results
for i, label in enumerate(pred_labels, start=1):
    print(f"Row {i}: {label}")
