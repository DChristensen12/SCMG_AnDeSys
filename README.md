
# Strawberry Creek Monitoring Group Anomaly Detection System

<p align="center">
  <img src="assets/SCMGlogo.jpg" width="400">
</p>

## Overview
This repository contains a Temporal Graph Neural Network (GNN) designed to monitor Strawberry Creek in real-time.

## How to use the Anomaly Detection System:

#### 1. Environment Setup
Clone the repo and install the dependencies. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

#### 2. Configuration (`.env`)
Create a `.env` file in the root directory based on `.env.example`. You will need:
* **SCMG_API_TOKEN**: Your access key for the Strawberry Creek API.
* **SMTP Credentials**: A Gmail App Password is required if you want to receive email alerts for detected spills.

#### 3. Initialization
Before starting the live monitor, the model needs a baseline. Run a fresh training cycle:
```bash
python main.py --mode train
```
This pulls the last 30 days of data, builds the graph, and saves the initial weights to `models/gnn_weights.pt`.


### Operation Modes

The system is controlled via the `--mode` flag in `main.py`:

| Mode | Description |
| :--- | :--- |
| `train` | Wipes existing weights and trains the GNN from scratch. |
| `update` | Loads existing weights and performs a fine-tuning pass on new data. |
| `inference` | Skips training. Rapidly evaluates the most recent 48 hours for spills. |

**To start the 24/7 monitoring loop:**
```bash
python run_live.py
```

---

### Outputs & Monitoring
* **Reports:** Every 15 minutes, a snapshot of the creek's health is saved to `reports/latest_report.png`.
* **Alerts:** If a spill is confirmed (anomaly score > rain-adjusted threshold), an email is dispatched to the receiver listed in your `.env`.
* **Logs:** Standard output tracks API pings and model evaluation scores.

---
