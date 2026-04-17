# TCLab Industrial Digital Twin & Anomaly Detection

This project implements a **Digital Twin** of the Temperature Control Lab (TCLab) hardware. It runs a real-time physical simulation side-by-side with the actual device and uses statistical methods (CUSUM) to detect operational anomalies.

## Project Features
- **Physics-Based Digital Twin:** A real-time ODE simulation using a dual-heater energy balance model.
- **Dynamic Calibration:** Uses **GEKKO (Moving Horizon Estimation)** to automatically tune physical parameters ($U$, $\alpha_1$, $\alpha_2$) for your specific hardware.
- **CUSUM Anomaly Detection:** A robust statistical filter that detects persistent shifts in the residual ($T_{actual} - T_{twin}$) while ignoring sensor noise.
- **Real-Time Monitoring:** Integrated dashboard with temperature tracking, heater power, and fault diagnosis.

## Workflow: Getting Started

### 1. Installation
Choose your preferred environment manager:

**Using Conda (Recommended):**
```bash
conda env create -f environment.yml
conda activate tclab_twin
```

**Using Pip:**
```bash
pip install -r requirements.txt
```

### 2. Collect Data (Hardware Calibration)
Run the step-test script to capture 10 minutes of real-world data from your device:
```bash
python src/collect_data.py
```

### 3. Calibrate Parameters
Use GEKKO to find the optimal physical parameters for your unique hardware:
```bash
python src/calibrate_parameters.py
```
*This updates `models/physics_parameters.json` with your device's specific $U$ and $\alpha$ factors.*

### 4. Run the Digital Twin
Start the real-time simulation and anomaly detector:
```bash
python src/digital_twin.py
```

## Directory Structure
- `models/`: Stores `physics_parameters.json` (The "DNA" of your calibrated model).
- `src/`: Core logic for data collection, GEKKO estimation, and the Digital Twin engine.
- `data/`: (Ignored by Git) Stores logs and generated plots.

## Anomaly Detection Logic
The system monitors for three primary fault types:
- **Heater Degradation:** Actual temperature is lower than predicted while heating.
- **External Cooling (Fan):** Rapid temperature drop beyond physical cooling rates.
- **External Heat Source:** Temperature rise while the heater is OFF.
- **Sensor Drift:** Persistent small offsets between reality and the twin.

---
*Note: This project was developed as part of the TCLab Digital Twin & Anomaly Detection series.*
