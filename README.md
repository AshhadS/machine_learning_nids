# NIDS (KDD’99) – Training, Realtime Detection, and Dashboard

This project demonstrates a simple **Network Intrusion Detection System (NIDS)** using the **KDD Cup 1999 (10%)** dataset.  
It includes:

- **Model training** (offline)
- **Realtime-style prediction** on incoming connections
- **A dashboard GUI** to display live predictions
- **A traffic simulator** that feeds the dashboard/predictor

The model predicts **attack categories** (because training was done with `--use_categories`):

- `normal`
- `dos`
- `probe`
- `r2l`
- `u2r`
- (optionally `other` if used during mapping)

---

## Project Files Explained

### `kddcup.data_10_percent`
The raw **KDD Cup 1999** dataset file (10% subset).  
- Each row = one network connection record
- Contains **41 features + 1 label** (42 columns total)
- Used for **training** and for generating simulated live traffic

---

### `nids_training.py`
Trains the NIDS model using the KDD dataset and saves the trained artifacts.

What it does:
- Loads `kddcup.data_10_percent`
- Applies preprocessing:
  - scales numeric features
  - one-hot encodes categorical features (`protocol_type`, `service`, `flag`)
- Trains an ML model (SGDClassifier with `log_loss`)
- Evaluates performance
- Saves:
  - `nids_kdd_pipeline.joblib`
  - `nids_label_encoder.joblib`

> ✅ Use this file when you want to retrain the model or experiment with different settings.

---

### `nids_kdd_pipeline.joblib`
Saved trained model **pipeline** (preprocessing + classifier).  
This includes:
- Scaling + OneHotEncoding configuration learned from training data
- The trained classifier weights

Used by:
- `realtime_predict.py`
- `dashboard.py`

---

### `nids_label_encoder.joblib`
Saved label encoder that maps between:
- numeric class IDs (e.g., 0,1,2…)
- label strings (e.g., `normal`, `dos`, `probe`, `r2l`, `u2r`)

Used by:
- `realtime_predict.py`
- `dashboard.py`

---

### `incoming.csv`
A CSV file containing **incoming traffic records** for realtime detection.

Important:
- It should contain **41 feature columns only**
- **NO label column**
- Can be:
  - with headers (recommended), or
  - without headers (must be in correct feature order)

Used by:
- `realtime_predict.py`
- `dashboard.py`

---

### `incoming copy.csv`
A backup or alternate input file (same format as `incoming.csv`).  
You can use this for testing without overwriting your main `incoming.csv`.

---

### `traffic_simulator.py`
A script that **simulates realtime network traffic** by appending rows to `incoming.csv`.

What it does:
- Reads `kddcup.data_10_percent`
- Randomly samples rows
- Writes only the **41 features** (no label) to `incoming.csv`
- Appends a new row every second (or per configured delay)

Used for:
- Demonstrating realtime detection without needing a real network capture system.

---

### `realtime_predict.py`
A simple realtime prediction script.

What it does:
- Loads:
  - `nids_kdd_pipeline.joblib`
  - `nids_label_encoder.joblib`
- Reads `incoming.csv`
- Predicts attack type for each row using `pipeline.predict()`
- Prints results to the terminal

Use this when you want:
- quick command-line predictions
- debugging model output

---

### `dashboard.py`
A realtime dashboard (GUI) that displays predictions and summary stats.

What it does:
- Loads:
  - `nids_kdd_pipeline.joblib`
  - `nids_label_encoder.joblib`
- Continuously reads `incoming.csv`
- Predicts attack categories
- Displays:
  - live table of recent predictions
  - counts of each attack type
  - simple charts (depending on the dashboard implementation)

Run this while `traffic_simulator.py` is feeding new rows to `incoming.csv`.

---

## Requirements (Modules to Install)

Install all required Python packages:

```bash
python -m pip install pandas numpy scikit-learn joblib
