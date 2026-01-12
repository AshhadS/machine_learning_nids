# dashboard.py
# Streamlit Realtime NIDS Dashboard (KDD + saved .joblib model)
#
# What this dashboard does (understaff-friendly):
# - Shows a prioritized "Top Alerts" queue with severity icons
# - Provides plain-English meaning + recommended action
# - Lets staff take action: ⛔ Block or ✅ Bypass (bypass requires a reason)
# - Logs every action to actions_audit.csv
# - Writes Blocked sources/events to blocked_list.csv (demo enforcement list)
# - Writes Bypasses with expiry to bypass_list.csv (demo allow list)
#
# Run:
#   python -m streamlit run dashboard.py
#
# Notes:
# - incoming.csv must contain ONLY the 41 KDD feature columns (no label)
# - This is a demo-style realtime feed (CSV). In production, you'd stream logs/events.

import os
import time
import hashlib
from datetime import datetime, timedelta
import json
import shutil

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# KDD feature columns (41) - NO label column here
# -----------------------------
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

# -----------------------------
# Understaff-friendly severity + messaging
# -----------------------------
SEVERITY = {
    "u2r": ("CRITICAL", "🔴", 4, "Privilege escalation risk (insider or compromised account)."),
    "r2l": ("CRITICAL", "🔴", 4, "Unauthorized access attempt (credential abuse / suspicious login)."),
    "dos": ("HIGH", "🟠", 3, "Service disruption risk (traffic flood / availability impact)."),
    "probe": ("MEDIUM", "🟡", 2, "Recon/scanning activity (may precede an attack)."),
    "normal": ("INFO", "✅", 1, "Normal-looking activity."),
    "other": ("MEDIUM", "🟡", 2, "Suspicious / unknown pattern.")
}

RECOMMENDED = {
    "u2r": "Block + escalate immediately",
    "r2l": "Block + verify account activity",
    "dos": "Rate-limit or block source",
    "probe": "Monitor + consider temporary block",
    "normal": "Allow",
    "other": "Review + monitor"
}

# Demo action files
AUDIT_FILE = "actions_audit.csv"
BLOCKED_FILE = "blocked_list.csv"
BYPASS_FILE = "bypass_list.csv"

# Human-review learning queue (best-practice: collect -> verify -> batch retrain)
PENDING_FILE = "feedback_pending.csv"
VERIFIED_FILE = "feedback_verified.csv"
REJECTED_FILE = "feedback_rejected.csv"
MODEL_DIR = "models"
TRAIN_LOG = "model_training_log.csv"


# -----------------------------
# Helpers: filesystem + logging
# -----------------------------
def _ensure_csv(path: str, header: str):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(header + "\n")


def ensure_action_files():
    _ensure_csv(AUDIT_FILE, "time,event_id,predicted_type,severity,action,reason,confidence")
    _ensure_csv(BLOCKED_FILE, "time,event_id,predicted_type,severity,confidence,notes")
    _ensure_csv(BYPASS_FILE, "time,event_id,predicted_type,severity,confidence,expires_at,reason")

    # Learning queue files
    queue_header = (
        "time,event_id,action,proposed_label,predicted_type,severity,confidence,reason,"
        "status,reviewed_time,verified_label,review_notes,"
        + ",".join(KDD_COLUMNS)
    )
    _ensure_csv(PENDING_FILE, queue_header)
    _ensure_csv(VERIFIED_FILE, queue_header)
    _ensure_csv(REJECTED_FILE, queue_header)

    _ensure_csv(
        TRAIN_LOG,
        "time,model_in,model_out,verified_used,updated_method,notes,metrics_json",
    )
    os.makedirs(MODEL_DIR, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_csv_field(s: str) -> str:
    return (s or "").replace(",", " ").replace("\n", " ").strip()


def make_event_id(row_dict: dict) -> str:
    """
    Create a stable fingerprint for an event.
    Because incoming.csv doesn't have an IP/user, we fingerprint the full row.
    """
    keys = sorted([k for k in row_dict.keys() if k in KDD_COLUMNS] + ["predicted_type", "severity", "confidence"])
    s = "|".join(str(row_dict.get(k, "")) for k in keys)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def log_action(event_id: str, pred_type: str, severity: str, action: str, reason: str, confidence):
    ensure_action_files()
    ts = utc_now_iso()
    conf_str = "" if confidence is None else f"{float(confidence):.3f}"
    with open(AUDIT_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts},{event_id},{pred_type},{severity},{action},{safe_csv_field(reason)},{conf_str}\n")


def add_block(event_id: str, pred_type: str, severity: str, confidence, notes: str = ""):
    ensure_action_files()
    ts = utc_now_iso()
    conf_str = "" if confidence is None else f"{float(confidence):.3f}"
    with open(BLOCKED_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts},{event_id},{pred_type},{severity},{conf_str},{safe_csv_field(notes)}\n")


def add_bypass(event_id: str, pred_type: str, severity: str, confidence, expires_at_iso: str, reason: str):
    ensure_action_files()
    ts = utc_now_iso()
    conf_str = "" if confidence is None else f"{float(confidence):.3f}"
    with open(BYPASS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{ts},{event_id},{pred_type},{severity},{conf_str},{expires_at_iso},{safe_csv_field(reason)}\n")


def log_to_learning_queue(row_dict: dict, action: str, proposed_label: str, reason: str, status: str = "PENDING"):
    """Append one row (features + metadata) into the learning queue."""
    ensure_action_files()
    ts = utc_now_iso()
    confv = row_dict.get("confidence", None)
    conf_str = "" if confv is None else f"{float(confv):.3f}"

    fields = [
        ts,
        str(row_dict.get("event_id", "")),
        action,
        proposed_label,
        str(row_dict.get("predicted_type", "")),
        str(row_dict.get("severity", "")),
        conf_str,
        safe_csv_field(reason),
        status,
        "",  # reviewed_time
        "",  # verified_label
        "",  # review_notes
    ]
    for c in KDD_COLUMNS:
        fields.append(str(row_dict.get(c, "")))

    with open(PENDING_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")


def _read_queue(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _append_queue_rows(path: str, rows_df: pd.DataFrame):
    if rows_df is None or rows_df.empty:
        return
    ensure_action_files()
    # Always write in the exact file column order
    header_df = _read_queue(path)
    if header_df.empty:
        # If file exists but was unreadable, just write columns from our queue header
        cols = [
            "time","event_id","action","proposed_label","predicted_type","severity","confidence","reason",
            "status","reviewed_time","verified_label","review_notes",
        ] + KDD_COLUMNS
    else:
        cols = list(header_df.columns)
    rows_df = rows_df.copy()
    for c in cols:
        if c not in rows_df.columns:
            rows_df[c] = ""
    rows_df = rows_df[cols]
    rows_df.to_csv(path, mode="a", header=False, index=False)


def move_queue_item(event_id: str, decision: str, verified_label: str = "", review_notes: str = "") -> bool:
    """Move one item from pending -> verified/rejected."""
    ensure_action_files()
    pending = _read_queue(PENDING_FILE)
    if pending.empty or "event_id" not in pending.columns:
        return False
    mask = pending["event_id"].astype(str) == str(event_id)
    if not mask.any():
        return False

    item = pending[mask].copy()
    pending = pending[~mask].copy()
    # Update review fields
    item.loc[:, "reviewed_time"] = utc_now_iso()
    item.loc[:, "verified_label"] = verified_label
    item.loc[:, "review_notes"] = safe_csv_field(review_notes)
    item.loc[:, "status"] = "VERIFIED" if decision == "VERIFY" else "REJECTED"

    target = VERIFIED_FILE if decision == "VERIFY" else REJECTED_FILE
    _append_queue_rows(target, item)

    # Rewrite pending without the moved item
    pending.to_csv(PENDING_FILE, index=False)
    return True


def load_bypass_active() -> pd.DataFrame:
    """
    Load bypasses and filter only currently active ones (expires_at in future).
    """
    if not os.path.exists(BYPASS_FILE):
        return pd.DataFrame(columns=["event_id", "expires_at"])
    df = pd.read_csv(BYPASS_FILE)
    if df.empty or "expires_at" not in df.columns:
        return pd.DataFrame(columns=["event_id", "expires_at"])

    # Parse expires_at
    def parse_expires(x):
        try:
            return datetime.fromisoformat(str(x).replace("Z", ""))
        except Exception:
            return None

    now = datetime.utcnow()
    df["expires_dt"] = df["expires_at"].apply(parse_expires)
    df = df[df["expires_dt"].notna()]
    df = df[df["expires_dt"] > now]
    return df


def is_event_bypassed(event_id: str, bypass_df: pd.DataFrame) -> bool:
    if bypass_df.empty:
        return False
    return (bypass_df["event_id"] == event_id).any()


# -----------------------------
# Model + incoming loading
# -----------------------------
@st.cache_resource
def load_artifacts(model_path: str, labels_path: str):
    pipeline = joblib.load(model_path)
    label_enc = joblib.load(labels_path)
    return pipeline, label_enc


def load_incoming_csv(path: str) -> pd.DataFrame:
    """
    Reads incoming.csv which should contain 41 feature columns.
    Works with or without headers.
    """
    # With headers
    try:
        df = pd.read_csv(path)
        if set(KDD_COLUMNS).issubset(df.columns):
            return df[KDD_COLUMNS].copy()
        raise ValueError
    except Exception:
        # Without headers
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


def try_batch_update_from_verified(pipeline, label_enc, verified_df: pd.DataFrame):
    """
    Best-effort weekly update using VERIFIED feedback.
    - If the pipeline (or its final classifier step) supports partial_fit, we do a batched partial_fit.
    - If it does not, we return (False, "not_supported").
    """
    if verified_df is None or verified_df.empty:
        return False, "no_verified_data", {}

    # Resolve labels to use
    if "verified_label" in verified_df.columns:
        y_labels = verified_df["verified_label"].fillna("")
    else:
        y_labels = pd.Series([""] * len(verified_df))
    if (y_labels == "").any():
        # Fallback to proposed label if reviewer left it blank
        if "proposed_label" in verified_df.columns:
            y_labels = y_labels.mask(y_labels == "", verified_df["proposed_label"].fillna(""))

    if (y_labels == "").any():
        return False, "missing_labels", {}

    X = verified_df[KDD_COLUMNS].copy()
    y_ids = label_enc.transform(y_labels.astype(str).tolist())
    classes = np.arange(len(label_enc.classes_))

    updated_method = ""
    try:
        if hasattr(pipeline, "partial_fit"):
            pipeline.partial_fit(X, y_ids, classes=classes)
            updated_method = "pipeline.partial_fit"
        elif hasattr(pipeline, "named_steps"):
            # Find a step that supports partial_fit (commonly clf)
            for step_name, step_obj in pipeline.named_steps.items():
                if hasattr(step_obj, "partial_fit"):
                    # Transform X through steps before the classifier step
                    try:
                        Xt = X
                        names = list(pipeline.named_steps.keys())
                        clf_idx = names.index(step_name)
                        for n in names[:clf_idx]:
                            Xt = pipeline.named_steps[n].transform(Xt)
                        step_obj.partial_fit(Xt, y_ids, classes=classes)
                        updated_method = f"named_steps.{step_name}.partial_fit"
                        break
                    except Exception:
                        continue

        if not updated_method:
            return False, "not_supported", {}

        # Basic self-check metrics on verified set (training-set metrics; useful for regression smoke-test)
        y_pred = pipeline.predict(X)
        y_pred_labels = label_enc.inverse_transform(y_pred)
        acc = float(np.mean(y_pred_labels == y_labels.astype(str).values))
        metrics = {"verified_accuracy": round(acc, 4), "verified_samples": int(len(verified_df))}
        return True, updated_method, metrics
    except Exception as e:
        return False, f"error:{e}", {}


def save_model_versioned(pipeline, base_name: str = "nids_kdd_pipeline") -> str:
    """Save a timestamped model under MODEL_DIR and return its path."""
    ensure_action_files()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(MODEL_DIR, f"{base_name}_{ts}.joblib")
    joblib.dump(pipeline, out_path)
    return out_path


def log_training_event(model_in: str, model_out: str, verified_used: int, updated_method: str, notes: str, metrics: dict):
    ensure_action_files()
    ts = utc_now_iso()
    metrics_json = json.dumps(metrics or {}, ensure_ascii=False)
    with open(TRAIN_LOG, "a", encoding="utf-8") as f:
        f.write(
            f"{ts},{safe_csv_field(model_in)},{safe_csv_field(model_out)},{int(verified_used)},{safe_csv_field(updated_method)},{safe_csv_field(notes)},{safe_csv_field(metrics_json)}\n"
        )


def build_triage(df_features: pd.DataFrame, labels: np.ndarray, conf: np.ndarray | None) -> pd.DataFrame:
    triage = df_features.copy()
    triage["predicted_type"] = labels

    sev_name, sev_icon, sev_rank, meaning, recommend = [], [], [], [], []
    for t in labels:
        name, icon, rank, msg = SEVERITY.get(str(t), ("MEDIUM", "🟡", 2, "Suspicious pattern."))
        sev_name.append(name)
        sev_icon.append(icon)
        sev_rank.append(rank)
        meaning.append(msg)
        recommend.append(RECOMMENDED.get(str(t), "Review"))

    triage["severity"] = sev_name
    triage["severity_icon"] = sev_icon
    triage["severity_rank"] = sev_rank
    triage["meaning"] = meaning
    triage["recommended_action"] = recommend

    if conf is not None:
        triage["confidence"] = np.round(conf, 3)

    triage["seen_time_utc"] = utc_now_iso()
    triage["event_id"] = triage.apply(lambda r: make_event_id(r.to_dict()), axis=1)

    # Sort for priority: highest severity first, then confidence (if available)
    sort_cols = ["severity_rank"] + (["confidence"] if "confidence" in triage.columns else [])
    triage = triage.sort_values(by=sort_cols, ascending=False).reset_index(drop=True)
    return triage


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="NIDS Dashboard", layout="wide")

st.title("🛡️ Realtime NIDS Dashboard")
st.caption("Designed for understaffed teams: prioritized alerts, simple meaning, and quick actions (Block / Bypass).")

# Sidebar settings
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Model (.joblib)", "nids_kdd_pipeline.joblib")
labels_path = st.sidebar.text_input("Labels (.joblib)", "nids_label_encoder.joblib")
incoming_path = st.sidebar.text_input("Incoming feed (CSV)", "incoming.csv")

refresh_sec = st.sidebar.slider("Auto-refresh interval (seconds)", 1, 10, 2)
auto_refresh = st.sidebar.toggle("Auto-refresh", value=True)

show_last_n = st.sidebar.slider("Process last N incoming rows", 50, 5000, 500, step=50)
display_keep = st.sidebar.slider("Keep last N events in dashboard", 50, 2000, 500, step=50)

st.sidebar.divider()
bypass_minutes_default = st.sidebar.slider("Bypass expiry (minutes)", 5, 240, 60, step=5)
top_n = st.sidebar.slider("Top Alerts to show", 5, 50, 10)

st.sidebar.divider()
st.sidebar.caption("Action logs written to:")
st.sidebar.code(
    f"{AUDIT_FILE}\n{BLOCKED_FILE}\n{BYPASS_FILE}\n\n{PENDING_FILE}\n{VERIFIED_FILE}\n{REJECTED_FILE}\n{TRAIN_LOG}",
    language="text",
)

# Session state init
if "last_seen_rows" not in st.session_state:
    st.session_state.last_seen_rows = 0
if "events" not in st.session_state:
    st.session_state.events = pd.DataFrame()
if "last_error" not in st.session_state:
    st.session_state.last_error = ""

# Load model
try:
    pipeline, label_enc = load_artifacts(model_path, labels_path)
    st.success("Model loaded ✅")
except Exception as e:
    st.error(f"Failed to load model files: {e}")
    st.stop()

# Load incoming
if not os.path.exists(incoming_path):
    st.warning(f"Waiting for `{incoming_path}` to appear...")
    st.info("Tip: run `python traffic_simulator.py` to start feeding incoming traffic.")
    st.stop()

try:
    incoming_df = load_incoming_csv(incoming_path)
    st.session_state.last_error = ""
except Exception as e:
    st.session_state.last_error = str(e)
    st.error(f"Incoming file error: {e}")
    st.stop()

if incoming_df.empty:
    st.info("`incoming.csv` is empty. Add rows (or run the traffic simulator) to see predictions.")
    st.stop()

# Only process the last show_last_n rows to keep it responsive
incoming_df = incoming_df.tail(show_last_n).reset_index(drop=True)

# Detect new rows since last run (within the window)
# If the window shrinks or resets, just reprocess from 0.
if st.session_state.last_seen_rows > len(incoming_df):
    st.session_state.last_seen_rows = 0

new_start = st.session_state.last_seen_rows
new_end = len(incoming_df)
new_rows = incoming_df.iloc[new_start:new_end].copy()
st.session_state.last_seen_rows = new_end

# Predict only on new rows
if not new_rows.empty:
    labels, conf = predict(pipeline, label_enc, new_rows)
    triage_new = build_triage(new_rows, labels, conf)

    # Append to stored events (keep last display_keep)
    if st.session_state.events.empty:
        st.session_state.events = triage_new
    else:
        st.session_state.events = pd.concat([st.session_state.events, triage_new], ignore_index=True)

    if len(st.session_state.events) > display_keep:
        st.session_state.events = st.session_state.events.tail(display_keep).reset_index(drop=True)

# Active bypass list (to visually mark bypassed events)
bypass_active = load_bypass_active()

# -----------------------------
# Summary KPIs
# -----------------------------
events = st.session_state.events.copy()
if events.empty:
    st.info("No events processed yet.")
    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()
    st.stop()

colA, colB, colC, colD = st.columns(4)
colA.metric("Events in view", len(events))
colB.metric("CRITICAL", int((events["severity"] == "CRITICAL").sum()))
colC.metric("HIGH", int((events["severity"] == "HIGH").sum()))
colD.metric("Last update (UTC)", utc_now_iso())

tabs = st.tabs(["🚨 Alerts", "🧑‍⚖️ Review Queue", "🧠 Model Training", "🗂️ Logs"])

# -----------------------------
# TAB 1: Prioritized Alerts + Actions
# -----------------------------
with tabs[0]:
    st.subheader("🚨 Top Alerts (Prioritized)")
    st.caption(
        "Focus on CRITICAL/HIGH first. "
        "Actions are logged. All decisions are added to a weekly human-review queue."
    )

    # Build a working triage list from current stored events
    triage_view = events.copy()
    triage_view["bypassed"] = triage_view["event_id"].apply(lambda eid: is_event_bypassed(eid, bypass_active))

    # For display priority: severity_rank desc, confidence desc, bypassed last
    sort_cols = ["severity_rank"]
    asc = [False]
    if "confidence" in triage_view.columns:
        sort_cols.append("confidence")
        asc.append(False)
    sort_cols.append("bypassed")
    asc.append(True)  # bypassed=True should go later

    triage_view = triage_view.sort_values(by=sort_cols, ascending=asc).reset_index(drop=True)

    label_options = list(getattr(label_enc, "classes_", []))
    if "other" not in label_options:
        label_options = label_options + ["other"]

    for i in range(min(top_n, len(triage_view))):
        row = triage_view.iloc[i].to_dict()
        pred = str(row["predicted_type"])
        sev = str(row["severity"])
        icon = str(row["severity_icon"])
        confv = row.get("confidence", None)
        event_id = str(row["event_id"])
        bypassed = bool(row.get("bypassed", False))

        with st.container(border=True):
            left, right = st.columns([3, 2])

            with left:
                title = f"{icon} {sev} — {pred.upper()}"
                if confv is not None:
                    title += f" (confidence {float(confv):.3f})"
                if bypassed:
                    title += "  ✅ BYPASSED (active)"

                st.markdown(f"### {title}")
                st.write(row["meaning"])
                st.write(f"**Recommended:** {row['recommended_action']}")
                st.caption(f"Event ID: `{event_id}`")

            with right:
                proposed_label = st.selectbox(
                    "Proposed label (for weekly review)",
                    options=label_options,
                    index=label_options.index(pred) if pred in label_options else 0,
                    key=f"proposed_{event_id}",
                )

                reason = st.text_input(
                    "Reason (required for bypass)",
                    key=f"reason_{event_id}",
                    placeholder="e.g., Verified internal maintenance scan / approved test",
                )

                bypass_minutes = st.number_input(
                    "Bypass duration (minutes)",
                    min_value=5,
                    max_value=240,
                    value=int(bypass_minutes_default),
                    step=5,
                    key=f"bypass_minutes_{event_id}",
                )

                st.divider()
                c1, c2 = st.columns(2)

                with c1:
                    if st.button("⛔ Block", key=f"block_{event_id}"):
                        log_action(event_id, pred, sev, "BLOCK", "", confv)
                        add_block(event_id, pred, sev, confv, notes="User chose block from dashboard")
                        log_to_learning_queue(row, "BLOCK", proposed_label, "")
                        st.success("Recorded: BLOCK ✅ (added to weekly review queue)")

                with c2:
                    bypass_disabled = (reason.strip() == "")
                    bypass_label = "✅ Bypass (Allow)"
                    if sev in ("CRITICAL", "HIGH"):
                        bypass_label = "⚠️ Bypass (Allow) — confirm"

                    if st.button(bypass_label, key=f"bypass_{event_id}", disabled=bypass_disabled):
                        expires_at = (datetime.utcnow() + timedelta(minutes=int(bypass_minutes)))
                        expires_iso = expires_at.replace(microsecond=0).isoformat() + "Z"

                        log_action(event_id, pred, sev, "BYPASS", reason, confv)
                        add_bypass(event_id, pred, sev, confv, expires_iso, reason)
                        log_to_learning_queue(row, "BYPASS", proposed_label, reason)
                        st.warning(f"Recorded: BYPASS ⚠️ (expires at {expires_iso}) — added to weekly review queue")


# -----------------------------
# TAB 2: Human Review Queue (Pending -> Verified/Rejected)
# -----------------------------
with tabs[1]:
    ensure_action_files()
    st.subheader("🧑‍⚖️ Weekly Human Review Queue")
    st.caption("Items here are collected during the week. A reviewer verifies labels before model retraining.")

    pending_df = _read_queue(PENDING_FILE)
    verified_df = _read_queue(VERIFIED_FILE)
    rejected_df = _read_queue(REJECTED_FILE)

    cA, cB, cC = st.columns(3)
    cA.metric("Pending", int(0 if pending_df.empty else len(pending_df)))
    cB.metric("Verified", int(0 if verified_df.empty else len(verified_df)))
    cC.metric("Rejected", int(0 if rejected_df.empty else len(rejected_df)))

    st.divider()
    if pending_df.empty:
        st.info("No pending items right now.")
    else:
        show_n_pending = st.slider("Show first N pending items", 5, 200, 50, step=5)
        label_options = list(getattr(label_enc, "classes_", []))
        if "other" not in label_options:
            label_options = label_options + ["other"]

        view = pending_df.head(show_n_pending).copy()
        for _, item in view.iterrows():
            eid = str(item.get("event_id", ""))
            sev = str(item.get("severity", ""))
            pred = str(item.get("predicted_type", ""))
            conf = item.get("confidence", "")
            proposed = str(item.get("proposed_label", ""))
            action = str(item.get("action", ""))

            header = f"{sev} — {pred.upper()} | action: {action} | event_id: {eid}"
            if str(conf) not in ("", "nan"):
                header += f" | conf: {conf}"

            with st.expander(header, expanded=False):
                st.write(f"**Reason:** {item.get('reason', '')}")
                st.write(f"**Proposed label:** `{proposed}`")
                st.caption(f"Seen: {item.get('time', '')}")

                verified_label = st.selectbox(
                    "Verified label",
                    options=label_options,
                    index=label_options.index(proposed) if proposed in label_options else 0,
                    key=f"verify_label_{eid}",
                )
                notes = st.text_input("Review notes (optional)", key=f"review_notes_{eid}")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("✅ Verify", key=f"verify_btn_{eid}"):
                        ok = move_queue_item(eid, "VERIFY", verified_label=verified_label, review_notes=notes)
                        if ok:
                            st.success("Moved to VERIFIED")
                            st.rerun()
                        else:
                            st.error("Could not move item (it may have already been processed).")
                with b2:
                    if st.button("❌ Reject", key=f"reject_btn_{eid}"):
                        ok = move_queue_item(eid, "REJECT", verified_label="", review_notes=notes)
                        if ok:
                            st.warning("Moved to REJECTED")
                            st.rerun()
                        else:
                            st.error("Could not move item (it may have already been processed).")

        st.divider()
        if st.button("📤 Export pending items (CSV)"):
            st.download_button(
                "Download feedback_pending.csv",
                data=open(PENDING_FILE, "rb").read(),
                file_name="feedback_pending.csv",
            )


# -----------------------------
# TAB 3: Model Training (Weekly batch update from VERIFIED)
# -----------------------------
with tabs[2]:
    ensure_action_files()
    st.subheader("🧠 Weekly Model Retraining")
    st.caption(
        "Best-practice demo workflow: collect decisions → human verify → retrain from VERIFIED only. "
        "This performs a batch update if your pipeline supports partial_fit."
    )

    verified_df = _read_queue(VERIFIED_FILE)
    if verified_df.empty:
        st.info("No VERIFIED feedback yet. Go to ‘Review Queue’ and verify some items first.")
    else:
        use_last_7_days = st.toggle("Use only last 7 days of VERIFIED data", value=True)
        min_samples = st.slider("Minimum VERIFIED samples required", 10, 500, 50, step=10)

        train_df = verified_df.copy()
        if use_last_7_days and "time" in train_df.columns:
            def _parse_time(x):
                try:
                    return datetime.fromisoformat(str(x).replace("Z", ""))
                except Exception:
                    return None
            train_df["_t"] = train_df["time"].apply(_parse_time)
            cutoff = datetime.utcnow() - timedelta(days=7)
            train_df = train_df[train_df["_t"].notna()]
            train_df = train_df[train_df["_t"] >= cutoff].copy()
            train_df.drop(columns=["_t"], inplace=True, errors="ignore")

        st.write(f"VERIFIED rows selected for training: **{len(train_df)}**")
        if len(train_df) < min_samples:
            st.warning("Not enough VERIFIED samples yet. Keep collecting/verification until you reach the threshold.")

        if "last_candidate_model" not in st.session_state:
            st.session_state.last_candidate_model = ""

        if st.button("🧪 Train candidate model from VERIFIED", disabled=(len(train_df) < min_samples)):
            # Reload the current model from disk so we don't mutate the in-memory cached model
            base_model = joblib.load(model_path)
            ok, method, metrics = try_batch_update_from_verified(base_model, label_enc, train_df)
            if ok:
                candidate_path = save_model_versioned(base_model, base_name="candidate_model")
                st.session_state.last_candidate_model = candidate_path
                log_training_event(
                    model_in=model_path,
                    model_out=candidate_path,
                    verified_used=len(train_df),
                    updated_method=method,
                    notes="Weekly verified batch update",
                    metrics=metrics,
                )
                st.success(f"Candidate model created: {candidate_path}")
                st.json(metrics)
            else:
                st.error(f"Could not train candidate model: {method}")

        candidate = st.session_state.get("last_candidate_model", "")
        if candidate:
            st.divider()
            st.write(f"**Last candidate:** `{candidate}`")
            st.caption("Promote replaces the active model file. A backup of the previous model is kept in /models.")
            if st.button("🚀 Promote candidate to active model"):
                # Backup current
                backup_path = save_model_versioned(joblib.load(model_path), base_name="backup_model")
                shutil.copyfile(candidate, model_path)
                st.success(f"Promoted ✅ Active model updated. Backup saved at: {backup_path}")
                st.info("Restart the dashboard (or clear cache) to ensure the new model is loaded everywhere.")

        st.divider()
        st.subheader("Training history")
        try:
            tdf = pd.read_csv(TRAIN_LOG).tail(20)
            st.dataframe(tdf, use_container_width=True)
        except Exception:
            st.info("No training events logged yet.")


# -----------------------------
# TAB 4: Logs + Live Events
# -----------------------------
with tabs[3]:
    st.subheader("📊 Live Events (Recent)")
    st.caption("This table shows the most recent events seen by the dashboard (not necessarily prioritized).")

    table_cols = [
        "seen_time_utc", "event_id", "severity_icon", "severity", "predicted_type"
    ]
    if "confidence" in events.columns:
        table_cols.append("confidence")

    table_df = events[table_cols].copy()
    table_df = table_df.tail(50).reset_index(drop=True)
    st.dataframe(table_df, use_container_width=True)

    st.subheader("📈 Attack Type Distribution (Current View)")
    counts = events["predicted_type"].value_counts()
    st.bar_chart(counts)

    st.subheader("🧭 Severity Breakdown (Current View)")
    sev_counts = events["severity"].value_counts()
    st.bar_chart(sev_counts)

    st.subheader("🗂️ Recent Actions (Audit Preview)")
    ensure_action_files()
    try:
        audit_df = pd.read_csv(AUDIT_FILE).tail(20)
        st.dataframe(audit_df, use_container_width=True)
    except Exception as e:
        st.info(f"Audit log will appear after first action. ({e})")

# -----------------------------
# Auto-refresh
# -----------------------------
if auto_refresh:
    time.sleep(refresh_sec)
    st.rerun()
