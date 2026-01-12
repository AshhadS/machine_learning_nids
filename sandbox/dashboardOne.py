# dashboard.py
# NIDS AI - Streamlit Dashboard (2-column layout)
# - Left: collapsible “compact cards” (expanders) for each attack (no select list)
# - Right (top): tables for (1) Attack severity & type summary, (2) Recent actions
# - Right (below): filters + settings
#
# Run:
#   streamlit run dashboard.py
#
# Data:
#   - Reads from incoming.csv by default
# Decisions:
#   - Saved to decisions.csv (Block / Bypass + optional note)

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="NIDS Dashboard", layout="wide", initial_sidebar_state="collapsed")

DEFAULT_DATA_FILE = "incoming.csv"
DECISIONS_FILE = "decisions.csv"

COLUMN_ALIASES = {
    "timestamp": ["timestamp", "time", "datetime", "date"],
    "attack_type": ["attack_type", "prediction", "label", "attack", "class"],
    "severity": ["severity", "risk", "level"],
    "confidence": ["confidence", "probability", "score", "proba"],
    "src_ip": ["src_ip", "source_ip", "src", "source"],
    "dst_ip": ["dst_ip", "dest_ip", "destination_ip", "dst", "dest", "destination"],
    "protocol": ["protocol", "proto"],
    "service": ["service", "dst_service", "app", "application"],
    "src_port": ["src_port", "source_port", "sport"],
    "dst_port": ["dst_port", "dest_port", "destination_port", "dport", "port"],
}


# ----------------------------
# Helpers
# ----------------------------
def pick_col(df: pd.DataFrame, key: str) -> str | None:
    for c in COLUMN_ALIASES.get(key, []):
        if c in df.columns:
            return c
    return None


def safe_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x)


def parse_timestamp(val) -> str:
    s = safe_str(val).strip()
    if not s:
        return ""
    try:
        dt = pd.to_datetime(s, errors="coerce")
        if pd.isna(dt):
            return s
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return s


def map_severity(attack_type: str, severity_val: str) -> str:
    sev = safe_str(severity_val).strip().upper()
    if sev:
        return sev

    a = safe_str(attack_type).lower()
    if any(k in a for k in ["u2r", "r2l", "root", "shell", "buffer", "warez", "sql", "xss", "injection"]):
        return "CRITICAL"
    if any(k in a for k in ["dos", "ddos", "flood", "syn", "smurf", "neptune"]):
        return "HIGH"
    if any(k in a for k in ["probe", "scan", "portsweep", "nmap", "satan", "ipsweep"]):
        return "MEDIUM"
    if any(k in a for k in ["normal", "benign"]):
        return "LOW"
    return "MEDIUM"


def severity_badge(sev: str) -> str:
    sev = safe_str(sev).upper()
    if sev == "CRITICAL":
        return "🟥 CRITICAL"
    if sev == "HIGH":
        return "🟧 HIGH"
    if sev == "MEDIUM":
        return "🟨 MEDIUM"
    return "🟩 LOW"


def format_conf(val) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    try:
        f = float(val)
        if 0 <= f <= 1:
            return f"{f*100:.1f}%"
        return f"{f:.3f}"
    except Exception:
        return safe_str(val)


def compute_event_id(row: dict) -> str:
    for k in ["event_id", "id", "uuid"]:
        if k in row and safe_str(row[k]).strip():
            return safe_str(row[k]).strip()

    basis = "|".join(
        [
            safe_str(row.get("_ts", "")),
            safe_str(row.get("_src_ip", "")),
            safe_str(row.get("_dst_ip", "")),
            safe_str(row.get("_attack_type", "")),
            safe_str(row.get("_protocol", "")),
            safe_str(row.get("_dst_port", "")),
        ]
    )
    return str(abs(hash(basis)))


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin-1")
        except Exception:
            return pd.DataFrame()


def ensure_decisions_file():
    if not os.path.exists(DECISIONS_FILE):
        pd.DataFrame(
            columns=[
                "event_id",
                "decision",
                "note",
                "decided_at",
                "attack_type",
                "severity",
                "src_ip",
                "dst_ip",
                "protocol",
                "service",
            ]
        ).to_csv(DECISIONS_FILE, index=False)


def read_decisions() -> pd.DataFrame:
    if not os.path.exists(DECISIONS_FILE):
        return pd.DataFrame(
            columns=[
                "event_id",
                "decision",
                "note",
                "decided_at",
                "attack_type",
                "severity",
                "src_ip",
                "dst_ip",
                "protocol",
                "service",
            ]
        )
    try:
        return pd.read_csv(DECISIONS_FILE)
    except Exception:
        return pd.DataFrame()


def save_decision(selected: dict, decision: str, note: str = ""):
    ensure_decisions_file()

    out = {
        "event_id": selected.get("_event_id", ""),
        "decision": decision,
        "note": note,
        "decided_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "attack_type": selected.get("_attack_type", ""),
        "severity": selected.get("_severity", ""),
        "src_ip": selected.get("_src_ip", ""),
        "dst_ip": selected.get("_dst_ip", ""),
        "protocol": selected.get("_protocol", ""),
        "service": selected.get("_service", ""),
    }

    existing = read_decisions()
    if not existing.empty and "event_id" in existing.columns:
        existing = existing[existing["event_id"] != out["event_id"]]

    updated = pd.concat([existing, pd.DataFrame([out])], ignore_index=True)
    updated.to_csv(DECISIONS_FILE, index=False)


def apply_filters(df: pd.DataFrame, severities: list[str], search: str) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()

    if severities:
        out = out[out["_severity"].isin(severities)]

    s = (search or "").strip().lower()
    if s:
        cols = [c for c in ["_ts", "_src_ip", "_dst_ip", "_attack_type", "_protocol", "_service", "_dst_port"] if c in out.columns]
        if cols:
            mask = False
            for c in cols:
                mask = mask | out[c].astype(str).str.lower().str.contains(s, na=False)
            out = out[mask]

    return out


def attack_header(row: dict) -> str:
    sev = severity_badge(row.get("_severity", ""))
    ts = safe_str(row.get("_ts", ""))
    src = safe_str(row.get("_src_ip", ""))
    dst = safe_str(row.get("_dst_ip", ""))
    proto = safe_str(row.get("_protocol", ""))
    svc = safe_str(row.get("_service", ""))
    dport = safe_str(row.get("_dst_port", ""))
    attack = safe_str(row.get("_attack_type", ""))

    svc_part = f"{proto}/{svc}" if svc else proto
    port_part = f":{dport}" if dport else ""
    return f"{sev} | {ts} | {src} → {dst} | {svc_part}{port_part} | {attack}"


def render_attack_details(row: dict):
    d1, d2, d3 = st.columns(3)
    with d1:
        st.write("**Time:**", row.get("_ts", ""))
        st.write("**Attack Type:**", row.get("_attack_type", ""))
        st.write("**Severity:**", severity_badge(row.get("_severity", "")))
    with d2:
        st.write("**Source IP:**", row.get("_src_ip", ""))
        st.write("**Destination IP:**", row.get("_dst_ip", ""))
        st.write("**Protocol:**", row.get("_protocol", ""))
    with d3:
        st.write("**Service:**", row.get("_service", ""))
        st.write("**Dest Port:**", row.get("_dst_port", ""))
        st.write("**Confidence:**", format_conf(row.get("_confidence", "")))

    with st.expander("Extra / Raw Fields", expanded=False):
        raw = {k: v for k, v in row.items() if not k.startswith("_")}
        st.json(raw)


# ----------------------------
# Header
# ----------------------------
st.title("🛡️ NIDS Dashboard")
st.caption("Collapsible alerts with criticality. Actions + summaries on the right.")


# ----------------------------
# Settings (inline on right, but keep minimal defaults here)
# ----------------------------
# We keep these in session for stability during reruns
if "data_file" not in st.session_state:
    st.session_state["data_file"] = DEFAULT_DATA_FILE
if "auto_refresh" not in st.session_state:
    st.session_state["auto_refresh"] = True
if "refresh_seconds" not in st.session_state:
    st.session_state["refresh_seconds"] = 3


# ----------------------------
# Load + normalize data
# ----------------------------
df = load_csv(st.session_state["data_file"])
if df.empty:
    st.warning(f"Couldn't load data from `{st.session_state['data_file']}`. Make sure the file exists and has rows.")
    st.stop()

col_ts = pick_col(df, "timestamp")
col_attack = pick_col(df, "attack_type")
col_sev = pick_col(df, "severity")
col_conf = pick_col(df, "confidence")
col_src_ip = pick_col(df, "src_ip")
col_dst_ip = pick_col(df, "dst_ip")
col_proto = pick_col(df, "protocol")
col_service = pick_col(df, "service")
col_dst_port = pick_col(df, "dst_port")

df["_ts"] = df[col_ts].apply(parse_timestamp) if col_ts else ""
df["_attack_type"] = df[col_attack].apply(safe_str) if col_attack else ""
df["_severity"] = [
    map_severity(a, s if col_sev else "")
    for a, s in zip(df["_attack_type"].tolist(), (df[col_sev].tolist() if col_sev else [""] * len(df)))
]
df["_confidence"] = df[col_conf] if col_conf else ""
df["_src_ip"] = df[col_src_ip].apply(safe_str) if col_src_ip else ""
df["_dst_ip"] = df[col_dst_ip].apply(safe_str) if col_dst_ip else ""
df["_protocol"] = df[col_proto].apply(safe_str) if col_proto else ""
df["_service"] = df[col_service].apply(safe_str) if col_service else ""
df["_dst_port"] = df[col_dst_port].apply(safe_str) if col_dst_port else ""

df["_event_id"] = df.apply(lambda r: compute_event_id(r.to_dict()), axis=1)

# Sort newest first (best effort)
if col_ts:
    try:
        df["_ts_sort"] = pd.to_datetime(df[col_ts], errors="coerce")
        df = df.sort_values("_ts_sort", ascending=False, na_position="last").drop(columns=["_ts_sort"])
    except Exception:
        pass


# ----------------------------
# Two-column layout
# ----------------------------
left, right = st.columns([2.2, 1.2], gap="large")

# ----------------------------
# Right column: TOP TABLES (required)
# - Attack severity/type summary
# - Recent actions
# Then filters/settings
# ----------------------------
with right:
    st.subheader("Summary")

    # --- Attack severity & type table (top-right)
    # Keep it compact: group by severity + attack type
    summary = (
        df.groupby(["_severity", "_attack_type"], dropna=False)
        .size()
        .reset_index(name="Count")
        .sort_values(["_severity", "Count"], ascending=[True, False])
    )

    # Make severity ordering nicer
    sev_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    summary["_sev_rank"] = summary["_severity"].map(lambda x: sev_order.get(safe_str(x).upper(), 99))
    summary = summary.sort_values(["_sev_rank", "Count"], ascending=[True, False]).drop(columns=["_sev_rank"])

    st.markdown("**Attack severity & type**")
    st.dataframe(
        summary.rename(columns={"_severity": "Severity", "_attack_type": "Attack Type"}),
        use_container_width=True,
        height=220,
    )

    # --- Recent actions table (top-right)
    decisions_df = read_decisions()
    st.markdown("**Recent actions**")
    if decisions_df.empty:
        st.info("No actions yet. (Block/Bypass decisions will appear here.)")
    else:
        # newest first
        try:
            decisions_df["_t"] = pd.to_datetime(decisions_df["decided_at"], errors="coerce")
            decisions_df = decisions_df.sort_values("_t", ascending=False).drop(columns=["_t"])
        except Exception:
            pass

        recent = decisions_df.head(12).copy()
        show_cols = ["decided_at", "decision", "severity", "attack_type", "src_ip", "dst_ip", "note"]
        recent = recent[[c for c in show_cols if c in recent.columns]]
        recent.rename(
            columns={
                "decided_at": "Time",
                "decision": "Action",
                "severity": "Severity",
                "attack_type": "Attack",
                "src_ip": "Src",
                "dst_ip": "Dst",
                "note": "Note",
            },
            inplace=True,
        )
        st.dataframe(recent, use_container_width=True, height=220)

    st.divider()

    # --- Filters/settings (below top tables)
    st.subheader("Filters & Settings")

    st.session_state["data_file"] = st.text_input("CSV file path", st.session_state["data_file"])

    sev_options = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    available_sev = sorted(list(set([s for s in df["_severity"].unique().tolist() if safe_str(s).strip()])))
    sev_choices = [s for s in sev_options if s in available_sev] + [s for s in available_sev if s not in sev_options]
    selected_sev = st.multiselect("Severity", options=sev_choices, default=sev_choices)

    search = st.text_input("Search (IP / attack / service / port)", "")

    st.session_state["auto_refresh"] = st.toggle("Auto refresh", value=st.session_state["auto_refresh"])
    st.session_state["refresh_seconds"] = st.slider("Refresh every (seconds)", 1, 15, st.session_state["refresh_seconds"])

    st.caption("Actions are saved to decisions.csv")


# Apply filters once
df_view = apply_filters(df, selected_sev, search)

# ----------------------------
# Left column: Collapsible cards (expanders)
# ----------------------------
with left:
    st.subheader("Live Alerts")

    # Small KPI row at top-left for staff friendliness
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total", int(len(df_view)))
    with k2:
        st.metric("Critical", int((df_view["_severity"] == "CRITICAL").sum()))
    with k3:
        st.metric("High", int((df_view["_severity"] == "HIGH").sum()))
    with k4:
        st.metric("Medium", int((df_view["_severity"] == "MEDIUM").sum()))

    if df_view.empty:
        st.info("No alerts match your filters.")
    else:
        # Don’t render too many expanders (keeps UI fast)
        MAX_ALERTS = 50
        view_slice = df_view.head(MAX_ALERTS)

        st.caption(f"Showing latest {len(view_slice)} alerts (max {MAX_ALERTS}).")

        for i, (_, r) in enumerate(view_slice.iterrows()):
            row = r.to_dict()

            # Make a truly-unique UI key for this rendered row
            ui_key = f"{row.get('_event_id', 'noid')}_{i}"

            header = attack_header(row)
            with st.expander(header, expanded=False):

                render_attack_details(row)
                st.divider()

                a1, a2, a3 = st.columns([1, 1, 2])

                with a1:
                    if st.button("🚫 Block", key=f"block_{ui_key}"):
                        save_decision(row, "Block", note="")
                        st.success("Saved: Block")
                        st.rerun()

                with a2:
                    if st.button("✅ Bypass", key=f"bypass_{ui_key}"):
                        save_decision(row, "Bypass", note="")
                        st.success("Saved: Bypass")
                        st.rerun()

                with a3:
                    note = st.text_input("Note (optional)", key=f"note_{ui_key}")
                    if st.button("💾 Save note", key=f"savenote_{ui_key}"):
                        save_decision(row, "Note", note=note)
                        st.success("Saved: Note")
                        st.rerun()



# ----------------------------
# Auto refresh at end
# ----------------------------
if st.session_state["auto_refresh"]:
    time.sleep(st.session_state["refresh_seconds"])
    st.rerun()
