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
import uuid
from datetime import datetime, timedelta

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

# Weekly human-review feedback workflow files
FEEDBACK_PENDING_FILE = "feedback_pending.csv"
FEEDBACK_VERIFIED_FILE = "feedback_verified.csv"
FEEDBACK_REJECTED_FILE = "feedback_rejected.csv"
TRAIN_LOG_FILE = "model_training_log.csv"
MODEL_DIR = "models"


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

    # Weekly review workflow
    pending_header = (
        "time,feedback_id,event_id,action,proposed_label,predicted_type,severity,confidence,reason,"
        + ",".join(KDD_COLUMNS)
    )
    verified_header = (
        "time,feedback_id,event_id,action,verified_label,predicted_type,severity,confidence,reason,"
        + ",".join(KDD_COLUMNS)
    )
    rejected_header = (
        "time,feedback_id,event_id,action,reject_reason,predicted_type,severity,confidence,reason,"
        + ",".join(KDD_COLUMNS)
    )
    _ensure_csv(FEEDBACK_PENDING_FILE, pending_header)
    _ensure_csv(FEEDBACK_VERIFIED_FILE, verified_header)
    _ensure_csv(FEEDBACK_REJECTED_FILE, rejected_header)
    _ensure_csv(TRAIN_LOG_FILE, "time,action,model_path,notes,num_samples")

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



def log_pending_feedback(event_row: dict, action: str, proposed_label: str, reason: str, confidence):
    """Append a PENDING feedback item for weekly human review."""
    ensure_action_files()
    ts = utc_now_iso()
    feedback_id = uuid.uuid4().hex
    conf_str = "" if confidence is None else f"{float(confidence):.3f}"
    fields = [
        ts, feedback_id, str(event_row.get("event_id", "")), action, proposed_label,
        str(event_row.get("predicted_type", "")), str(event_row.get("severity", "")), conf_str, safe_csv_field(reason)
    ]
    for c in KDD_COLUMNS:
        fields.append(str(event_row.get(c, "")))
    with open(FEEDBACK_PENDING_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
    return feedback_id


def _read_csv_safe(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _rewrite_csv(path: str, df: pd.DataFrame):
    df.to_csv(path, index=False)


def move_pending_to_verified(feedback_id: str, verified_label: str):
    ensure_action_files()
    pending = _read_csv_safe(FEEDBACK_PENDING_FILE)
    if pending.empty or "feedback_id" not in pending.columns:
        return False
    row = pending[pending["feedback_id"].astype(str) == str(feedback_id)]
    if row.empty:
        return False
    row = row.iloc[0].to_dict()
    ts = utc_now_iso()
    conf = row.get("confidence", "")
    fields = [
        ts, str(row.get("feedback_id")), str(row.get("event_id")), str(row.get("action")), verified_label,
        str(row.get("predicted_type")), str(row.get("severity")), str(conf), safe_csv_field(str(row.get("reason", "")))
    ]
    for c in KDD_COLUMNS:
        fields.append(str(row.get(c, "")))
    with open(FEEDBACK_VERIFIED_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
    pending = pending[pending["feedback_id"].astype(str) != str(feedback_id)]
    _rewrite_csv(FEEDBACK_PENDING_FILE, pending)
    return True


def move_pending_to_rejected(feedback_id: str, reject_reason: str):
    ensure_action_files()
    pending = _read_csv_safe(FEEDBACK_PENDING_FILE)
    if pending.empty or "feedback_id" not in pending.columns:
        return False
    row = pending[pending["feedback_id"].astype(str) == str(feedback_id)]
    if row.empty:
        return False
    row = row.iloc[0].to_dict()
    ts = utc_now_iso()
    conf = row.get("confidence", "")
    fields = [
        ts, str(row.get("feedback_id")), str(row.get("event_id")), str(row.get("action")), safe_csv_field(reject_reason),
        str(row.get("predicted_type")), str(row.get("severity")), str(conf), safe_csv_field(str(row.get("reason", "")))
    ]
    for c in KDD_COLUMNS:
        fields.append(str(row.get(c, "")))
    with open(FEEDBACK_REJECTED_FILE, "a", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
    pending = pending[pending["feedback_id"].astype(str) != str(feedback_id)]
    _rewrite_csv(FEEDBACK_PENDING_FILE, pending)
    return True

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
st.sidebar.code(f"{AUDIT_FILE}\n{BLOCKED_FILE}\n{BYPASS_FILE}", language="text")

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
def render_live(bypass_active, top_n, auto_refresh, refresh_sec):
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

    # -----------------------------
    # Prioritized Alerts + Actions
    # -----------------------------
    st.subheader("🚨 Top Alerts (Prioritized)")
    st.caption(
        "Focus on CRITICAL/HIGH first. "
        "Block/BYPASS decisions are logged. Bypass requires a reason and expires automatically."
    )

    # Build a working triage list from current stored events
    triage_view = events.copy()

    # Mark bypassed items (active)
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

    # Show top N
    for i in range(min(top_n, len(triage_view))):
        row = triage_view.iloc[i].to_dict()
        pred = str(row["predicted_type"])
        sev = str(row["severity"])
        icon = str(row["severity_icon"])
        confv = row.get("confidence", None)
        event_id = str(row["event_id"])
        bypassed = bool(row.get("bypassed", False))

        border_color = "#ffcccc" if sev == "CRITICAL" else ("#ffe6cc" if sev == "HIGH" else "#f3f3f3")

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
                    # Proposed label for weekly review (unique key includes index)
                    label_options = ["normal", "dos", "probe", "r2l", "u2r", "other"]
                    default_label = pred if pred in label_options else ("normal" if sev == "INFO" else "other")
                    proposed_label = st.selectbox(
                        "Proposed label (for weekly review)",
                        options=label_options,
                        index=label_options.index(default_label) if default_label in label_options else 0,
                        key=f"proposed_{event_id}_{i}",
                    )

                    # Staff inputs
                    reason = st.text_input(
                        "Reason (required for bypass)",
                        key=f"alerts_reason_{event_id}_{i}",
                        placeholder="e.g., Verified internal maintenance scan / approved test",
                    )

                    bypass_minutes = st.number_input(
                        "Bypass duration (minutes)",
                        min_value=5,
                        max_value=240,
                        value=int(bypass_minutes_default),
                        step=5,
                        key=f"alerts_bypass_minutes_{event_id}_{i}",
                    )

                    st.divider()
                    c1, c2 = st.columns(2)

                    with c1:
                        if st.button("⛔ Block", key=f"alerts_block_{event_id}_{i}"):
                            log_action(event_id, pred, sev, "BLOCK", "", confv)
                            add_block(event_id, pred, sev, confv, notes="User chose block from dashboard")
                            st.success("Recorded: BLOCK ✅")

                    with c2:
                        # Guardrails: require reason; extra warning label for severe cases
                        bypass_disabled = (reason.strip() == "")
                        bypass_label = "✅ Bypass (Allow)"
                        if sev in ("CRITICAL", "HIGH"):
                            bypass_label = "⚠️ Bypass (Allow) — confirm"

                        if st.button(bypass_label, key=f"alerts_bypass_{event_id}_{i}", disabled=bypass_disabled):
                            expires_at = (datetime.utcnow() + timedelta(minutes=int(bypass_minutes)))
                            expires_iso = expires_at.replace(microsecond=0).isoformat() + "Z"

                            log_action(event_id, pred, sev, "BYPASS", reason, confv)
                            add_bypass(event_id, pred, sev, confv, expires_iso, reason)
                            _row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
                            log_pending_feedback(_row_dict, "BYPASS", proposed_label, reason, confv)
                            st.warning(f"Recorded: BYPASS ⚠️ (expires at {expires_iso})")

    # -----------------------------
    # Live Events Table + Charts
    # -----------------------------
    st.subheader("📊 Live Events (Recent)")
    st.caption("This table shows the most recent events seen by the dashboard (not necessarily prioritized).")

    # Show a smaller subset for readability
    table_cols = [
        "seen_time_utc", "event_id", "severity_icon", "severity", "predicted_type"
    ]
    if "confidence" in events.columns:
        table_cols.append("confidence")

    table_df = events[table_cols].copy()
    table_df = table_df.tail(50).reset_index(drop=True)

    st.dataframe(table_df, use_container_width=True)

    # Distribution chart
    st.subheader("📈 Attack Type Distribution (Current View)")
    counts = events["predicted_type"].value_counts()
    st.bar_chart(counts)

    # Severity breakdown
    st.subheader("🧭 Severity Breakdown (Current View)")
    sev_counts = events["severity"].value_counts()
    st.bar_chart(sev_counts)

    # -----------------------------
    # Action Log Preview
    # -----------------------------
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

def render_review_queue():
    st.subheader("🧑‍⚖️ Review Queue (Weekly Human Check)")
    st.caption("Review PENDING items created from Block/BYPASS actions. Verify or Reject them.")
    pending = _read_csv_safe(FEEDBACK_PENDING_FILE)
    if pending.empty:
        st.info("No pending items.")
        return

    cols = ["time","feedback_id","event_id","action","proposed_label","predicted_type","severity","confidence","reason"]
    show_cols = [c for c in cols if c in pending.columns]
    st.dataframe(pending[show_cols].head(200), use_container_width=True)
    st.divider()

    for idx, r in pending.head(50).iterrows():
        fid = str(r.get("feedback_id"))
        eid = str(r.get("event_id"))
        sev = str(r.get("severity",""))
        pred = str(r.get("predicted_type",""))
        proposed = str(r.get("proposed_label",""))

        st.markdown(f"**{sev} — {pred.upper()}** · event_id `{eid}` · feedback_id `{fid}`")
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            label_options = ["normal","dos","probe","r2l","u2r","other"]
            default = proposed if proposed in label_options else (pred if pred in label_options else "other")
            verified_label = st.selectbox(
                "Verified label",
                options=label_options,
                index=label_options.index(default) if default in label_options else 0,
                key=f"verified_{fid}_{idx}",
            )
            reject_reason = st.text_input("Reject reason", key=f"rej_{fid}_{idx}")
        with c2:
            if st.button("✅ Verify", key=f"verify_btn_{fid}_{idx}"):
                ok = move_pending_to_verified(fid, verified_label)
                st.success("Verified ✅" if ok else "Item not found.")
                st.rerun()
        with c3:
            if st.button("❌ Reject", key=f"reject_btn_{fid}_{idx}"):
                ok = move_pending_to_rejected(fid, reject_reason or "Rejected by reviewer")
                st.warning("Rejected ❌" if ok else "Item not found.")
                st.rerun()

        with st.expander("Show features"):
            feats = {c: r.get(c) for c in KDD_COLUMNS if c in pending.columns}
            st.json(feats)
        st.divider()


def render_model_training(model_path: str, labels_path: str):
    st.subheader("🧠 Model Training (from VERIFIED items)")
    st.caption("Trains a candidate model from VERIFIED feedback and lets you promote it with backup.")
    verified = _read_csv_safe(FEEDBACK_VERIFIED_FILE)
    if verified.empty:
        st.info("No verified items yet.")
        return

    st.write(f"Verified samples: **{len(verified)}**")
    candidate_path = os.path.join(MODEL_DIR, "candidate_model.joblib")

    if st.button("🧪 Train Candidate (incremental)", key="train_candidate"):
        try:
            pipeline, label_enc = load_artifacts(model_path, labels_path)
            X = verified[KDD_COLUMNS].copy()
            y = verified["verified_label"].astype(str).values
            y_ids = label_enc.transform(y)
            classes = np.arange(len(label_enc.classes_))

            if hasattr(pipeline, "partial_fit"):
                pipeline.partial_fit(X, y_ids, classes=classes)
            else:
                pipeline.fit(X, y_ids)

            joblib.dump(pipeline, candidate_path)
            with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{utc_now_iso()},TRAIN,{candidate_path},trained from verified,{len(verified)}\n")
            st.success(f"Candidate saved: {candidate_path}")
        except Exception as e:
            st.error(f"Training failed: {e}")

    if st.button("🚀 Promote Candidate", key="promote_candidate"):
        if not os.path.exists(candidate_path):
            st.error("Train a candidate first.")
            return
        try:
            backup_path = os.path.join(MODEL_DIR, f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.joblib")
            if os.path.exists(model_path):
                joblib.dump(joblib.load(model_path), backup_path)
            os.replace(candidate_path, model_path)
            with open(TRAIN_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"{utc_now_iso()},PROMOTE,{model_path},backup={backup_path},{len(verified)}\n")
            st.success(f"Promoted ✅ Backup: {backup_path}")
        except Exception as e:
            st.error(f"Promotion failed: {e}")


def render_logs():
    st.subheader("🗂️ Logs")
    ensure_action_files()
    audit = _read_csv_safe(AUDIT_FILE)
    pending = _read_csv_safe(FEEDBACK_PENDING_FILE)
    verified = _read_csv_safe(FEEDBACK_VERIFIED_FILE)
    rejected = _read_csv_safe(FEEDBACK_REJECTED_FILE)
    tlog = _read_csv_safe(TRAIN_LOG_FILE)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pending", 0 if pending.empty else len(pending))
    col2.metric("Verified", 0 if verified.empty else len(verified))
    col3.metric("Rejected", 0 if rejected.empty else len(rejected))

    st.write("### Audit (latest 100)")
    st.dataframe(audit.tail(100) if not audit.empty else pd.DataFrame(), use_container_width=True)

    st.write("### Training log")
    st.dataframe(tlog.tail(200) if not tlog.empty else pd.DataFrame(), use_container_width=True)

# -----------------------------
# Tabs (always render)
# -----------------------------
tab_live, tab_review, tab_train, tab_logs = st.tabs([
    "🚨 Live Alerts",
    "🧑‍⚖️ Review Queue",
    "🧠 Model Training",
    "🗂️ Logs",
])


with tab_live:
    render_live(bypass_active, top_n, auto_refresh, refresh_sec)

with tab_review:
    render_review_queue()

with tab_train:
    render_model_training(model_path, labels_path)

with tab_logs:
    render_logs()