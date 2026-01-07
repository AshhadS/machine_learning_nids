import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 41 feature columns (NO label)
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

st.set_page_config(page_title="NIDS Dashboard", layout="wide")
st.title("🛡️ Realtime NIDS Dashboard (KDD + self-trained model)")

# Sidebar controls
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model (.joblib)", "nids_kdd_pipeline.joblib")
labels_path = st.sidebar.text_input("Label encoder (.joblib)", "nids_label_encoder.joblib")
incoming_path = st.sidebar.text_input("Incoming feed (CSV)", "incoming.csv")
refresh_sec = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 2)
show_last_n = st.sidebar.slider("Show last N events", 10, 500, 100)

@st.cache_resource
def load_artifacts(model_p, labels_p):
    pipeline = joblib.load(model_p)
    label_enc = joblib.load(labels_p)
    return pipeline, label_enc

def load_incoming_csv(path: str) -> pd.DataFrame:
    """
    Reads incoming.csv which should contain 41 feature columns.
    Works with or without headers.
    """
    try:
        df = pd.read_csv(path)
        if set(KDD_COLUMNS).issubset(df.columns):
            return df[KDD_COLUMNS].copy()
        # If header exists but doesn't match, fall back to header=None
        raise ValueError
    except Exception:
        df = pd.read_csv(path, header=None)
        if df.shape[1] != len(KDD_COLUMNS):
            raise ValueError(f"incoming.csv must have {len(KDD_COLUMNS)} columns, found {df.shape[1]}")
        df.columns = KDD_COLUMNS
        return df

def predict(pipeline, label_enc, X: pd.DataFrame):
    pred_ids = pipeline.predict(X)
    pred_labels = label_enc.inverse_transform(pred_ids)

    conf = None
    try:
        proba = pipeline.predict_proba(X)
        conf = np.max(proba, axis=1)
    except Exception:
        conf = None

    return pred_labels, conf

# Load model
try:
    pipeline, label_enc = load_artifacts(model_path, labels_path)
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# Main loop (simple realtime refresh)
placeholder = st.empty()

while True:
    with placeholder.container():
        st.subheader("Live predictions")

        try:
            incoming_df = load_incoming_csv(incoming_path)
        except Exception as e:
            st.warning(f"Waiting for valid incoming.csv... ({e})")
            st.info("Create incoming.csv with 41 columns (KDD feature order).")
            time.sleep(refresh_sec)
            st.rerun()

        if len(incoming_df) == 0:
            st.info("incoming.csv is empty. Add rows to see predictions.")
            time.sleep(refresh_sec)
            st.rerun()

        # Only show last N rows
        recent = incoming_df.tail(show_last_n).reset_index(drop=True)

        labels, conf = predict(pipeline, label_enc, recent)

        out = recent.copy()
        out["predicted_type"] = labels
        if conf is not None:
            out["confidence"] = np.round(conf, 3)

        # Add a simple timestamp column (dashboard time)
        out["dashboard_time"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(out.tail(30), use_container_width=True)

        with col2:
            st.metric("Total rows seen", len(incoming_df))
            counts = pd.Series(labels).value_counts()
            st.bar_chart(counts)

            st.write("Attack type distribution (last N)")
            st.table(counts.rename_axis("type").to_frame("count"))

        st.caption("Tip: Append new rows to incoming.csv to simulate live traffic.")

    time.sleep(refresh_sec)
    st.rerun()
