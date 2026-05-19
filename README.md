
# Strawberry Creek Monitoring Group Anomaly Detection System

<p align="center">
  <img src="assets/SCMGlogo.jpg" width="400">
</p>

## Overview
This repository contains the Strawberry Creek anomaly detection pipeline. It uses a Temporal Graph Neural Network that models the creek as a directed flow graph, 5 sensor nodes connected by the physical direction of water, and learns what normal conditions look like via reconstruction. Anomalies are flagged when the model's prediction error spikes past a threshold that adjusts upward during rain events to reduce false positives from natural runoff.

The live model being used right now in the registry is **DuskCrayfish**: a Graph Convolutional Network spatial layer followed by an Long Short-Term Memory temporal layer, named after the crayfish that live in Strawberry Creek.

## How to use the Anomaly Detection System:

#### 1. Environment Setup
Clone the repo and install the dependencies. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

#### 2. Configuration (`.env`)
Create a `.env` file in the root directory based on `.env.example`. Fill in what applies to your setup:

* **SCMG_API_TOKEN**: Optional. The public API currently requires no token, but SCMG may re-enable auth — leave blank unless you have one.
* **MySQL credentials** (`MYSQL_HOST`, `MYSQL_DATABASE_USER`, etc.): Only needed if running with `--data-source sql`. This connects directly to Night Heron's production database and uses the same credentials, so most users should stick with the default API source.
* **NWS weather** (`NWS_USER_AGENT`): Add a contact email to the User-Agent string as the NWS API requires. The station is LBNL1 (Lawrence Berkeley National Lab), which provides air temperature, dewpoint, humidity, wind, and pressure. Set `USE_NWS_WEATHER=false` to disable.
* **SMTP credentials**: A Gmail App Password is required if you want email alerts for detected spills.

#### 3. Initialization
Before starting the live monitor, the model needs a baseline. Run a fresh training cycle:
```bash
python main.py --mode train
```
This pulls the last 30 days of creek and NWS weather data, builds the flow graph, and saves the initial weights to `models/dusk_crayfish_weights.pt`.

---

### Operation Modes

The system is controlled via `--mode` and `--model` flags in `main.py`:

| Mode | Description |
| :--- | :--- |
| `train` | Trains the model from scratch on the last 30 days of data. |
| `update` | Loads existing weights and fine-tunes on fresh data. |
| `inference` | Skips training. Evaluates the last 48 hours for anomalies. |

The `--model` flag selects which architecture to use (default: `dusk_crayfish`). The `--data-source` flag switches between the public REST API (`api`, default) and the MySQL database (`sql`).

**To start the 24/7 monitoring loop:**
```bash
python run_live.py
```
This runs inference every 15 minutes via a subprocess call to `main.py --mode inference`.

---

### Outputs & Monitoring
* **Reports:** Each inference cycle saves a dashboard to `reports/latest_report.png` and a timestamped copy alongside it.
* **Alerts:** If a spill is confirmed (anomaly score exceeds the rain-adjusted threshold), an email is sent to the address in your `.env`.
* **Logs:** Standard output tracks data pulls, model evaluation scores, and anomaly counts each cycle.

---

<p align="center">
  <img src="assets/SCMGBacklogo.png" width="400">
</p>