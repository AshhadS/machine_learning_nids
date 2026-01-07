# Network Intrusion Detection System (NIDS) – KDD Cup 1999

This project implements a **machine-learning-based Network Intrusion Detection System (NIDS)** using the **KDD Cup 1999 (10%) dataset**.  
The system is capable of:

- Training an intrusion detection model offline
- Detecting network intrusions in near-realtime
- Classifying traffic into attack categories
- Displaying live predictions in a **Streamlit dashboard**
- Simulating live network traffic for demonstration purposes

The model predicts **high-level attack categories**:

- `normal`
- `dos`
- `probe`
- `r2l`
- `u2r`

---

## Project Structure

```
.
├── dashboard.py
├── incoming.csv
├── incoming copy.csv
├── kddcup.data_10_percent
├── nids_kdd_pipeline.joblib
├── nids_label_encoder.joblib
├── nids_training.py
├── realtime_predict.py
├── traffic_simulator.py
└── README.md
```

---

## File Descriptions

### `kddcup.data_10_percent`
The **KDD Cup 1999 (10%) dataset** used for training and simulation.

- Each row = one network connection
- 41 traffic features + 1 label
- Used for model training and traffic simulation

---

### `nids_training.py`
Trains the intrusion detection model.

- Loads the KDD dataset
- Groups attack labels into categories
- Applies preprocessing (scaling + one-hot encoding)
- Trains a machine-learning classifier
- Saves trained artifacts

Outputs:
- `nids_kdd_pipeline.joblib`
- `nids_label_encoder.joblib`

---

### `nids_kdd_pipeline.joblib`
Serialized trained model pipeline (preprocessing + classifier).

---

### `nids_label_encoder.joblib`
Maps numeric class IDs to readable labels (`normal`, `dos`, `probe`, `r2l`, `u2r`).

---

### `incoming.csv`
Live input file for realtime detection.
- 41 feature columns only
- No label column
- Continuously updated by the traffic simulator

---

### `incoming copy.csv`
Backup/reference copy of `incoming.csv`.

---

### `traffic_simulator.py`
Simulates realtime network traffic by appending rows to `incoming.csv`.

---

### `realtime_predict.py`
Terminal-based realtime intrusion detection script.

---

### `dashboard.py`
Streamlit-based GUI dashboard for realtime intrusion detection.

---

## Requirements

### Python Version
- Python 3.9 or higher recommended

### Required Python Modules

```bash
python -m pip install pandas numpy scikit-learn joblib streamlit
```

---

## How to Run the Project (Step-by-Step)

### Step 1 — Navigate to Project Folder
```bash
cd path\to\your\project\folder
```

---

### Step 2 — Verify Python
```bash
python --version
```

---

### Step 3 — Train the Model (Run Once)
```bash
python nids_training.py --data "kddcup.data_10_percent" --use_categories
```

---

### Step 4 — Prepare Incoming Traffic File
```bash
copy "incoming copy.csv" "incoming.csv"
```

---

### Step 5 — Start Traffic Simulator
```bash
python traffic_simulator.py
```

---

### Step 6 — (Optional) Terminal Realtime Detection
```bash
python realtime_predict.py
```

---

### Step 7 — Start Streamlit Dashboard
```bash
python -m streamlit run dashboard.py
```

---

## Execution Flow Summary

```
KDD Dataset
     ↓
Model Training
     ↓
Saved Model (.joblib)
     ↓
Traffic Simulator → incoming.csv
     ↓
Realtime Prediction
     ↓
Streamlit Dashboard
```

---

## Notes

- Always run Streamlit using `python -m streamlit run dashboard.py`
- Do not add a label column to `incoming.csv`
- Keep the traffic simulator running during demos

---

## Outcome

This project demonstrates realtime intrusion detection, attack classification, and live visualization using machine learning and Streamlit.
