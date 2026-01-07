import time
import pandas as pd

# This expects you have your original KDD file available
# We will copy random rows (features only) and append them to incoming.csv
KDD_COLUMNS_WITH_LABEL = [
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

FEATURE_COLS = KDD_COLUMNS_WITH_LABEL[:-1]

kdd_path = "kddcup.data_10_percent"
df = pd.read_csv(kdd_path, header=None, names=KDD_COLUMNS_WITH_LABEL)

# Write headers once (optional)
out_path = "incoming.csv"
if not pd.io.common.file_exists(out_path):
    pd.DataFrame(columns=FEATURE_COLS).to_csv(out_path, index=False)

print("Appending rows to incoming.csv every 1 second. Ctrl+C to stop.")
while True:
    row = df.sample(1)[FEATURE_COLS]
    row.to_csv(out_path, mode="a", header=False, index=False)
    time.sleep(1)
