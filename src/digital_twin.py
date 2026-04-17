import tclab
import numpy as np
import time
import json
import csv
import os
import matplotlib.pyplot as plt
from anomaly_detector import CUSUMDetector

# 1. Load Pre-Estimated Physical Parameters
# Use the script's location to find the project root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
params_path = os.path.join(ROOT_DIR, 'models/physics_parameters.json')

# Default parameters
params = {
    "U": 10.0, 
    "alpha1": 0.0100, 
    "alpha2": 0.0075, 
    "A": 0.001, 
    "As": 0.0002, 
    "mCp": 2.0, 
    "epsilon": 0.9, 
    "sigma": 5.67e-8
}

if os.path.exists(params_path):
    try:
        with open(params_path, 'r') as f:
            file_params = json.load(f)
            # Update only keys that exist in the file
            params.update(file_params)
    except Exception as e:
        print(f"Warning: Could not load {params_path}, using defaults. Error: {e}")

U = params['U']
alpha1 = params['alpha1']
alpha2 = params['alpha2']
A = params['A']
As = params['As']
mCp = params['mCp']
epsilon = params['epsilon']
sigma = params['sigma']

# 2. Dual-Heater Physics Engine (from PDF)
def tclab_dual_ode(T1, T2, Q1, Q2, Ta):
    T1_k = T1 + 273.15
    T2_k = T2 + 273.15
    Ta_k = Ta + 273.15
    
    # Common terms
    rad_const = epsilon * sigma
    
    # Heater 1 derivatives
    conv1 = U * A * (Ta_k - T1_k)
    rad1 = rad_const * A * (Ta_k**4 - T1_k**4)
    conv12 = U * As * (T2_k - T1_k)
    rad12 = rad_const * As * (T2_k**4 - T1_k**4)
    heat1 = alpha1 * Q1
    dT1dt = (conv1 + rad1 + conv12 + rad12 + heat1) / mCp
    
    # Heater 2 derivatives
    conv2 = U * A * (Ta_k - T2_k)
    rad2 = rad_const * A * (Ta_k**4 - T2_k**4)
    conv21 = U * As * (T1_k - T2_k)
    rad21 = rad_const * As * (T1_k**4 - T2_k**4)
    heat2 = alpha2 * Q2
    dT2dt = (conv2 + rad2 + conv21 + rad21 + heat2) / mCp
    
    return dT1dt, dT2dt

# 3. Setup Monitoring and Anomaly Detector
# threshold: higher = less sensitive
# drift: higher = ignores larger small-scale errors
detector = CUSUMDetector(threshold=50.0, drift=5.0)

# 3.1. Auto-select Hardware or Simulation (Digital Twin Mode)
try:
    lab = tclab.TCLab()
    print("Connected to TCLab hardware.")
except RuntimeError:
    print("No Arduino device found. Initializing TCLabModel (Digital Twin Mode)...")
    lab = tclab.TCLabModel()

n_steps = 1200
dt = 1.0

# Setup Data Logging
log_file = os.path.join(ROOT_DIR, 'data/tclab_log.csv')
os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
csv_file = open(log_file, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'Q1', 'Q2', 'T1_Actual', 'T1_Twin', 'T2_Actual', 'T2_Twin', 'Residual', 'Alarm', 'FaultType'])

with lab:
    print(f"Digital Twin and Anomaly Detector started. Logging to {log_file}...")
    T1_actual = lab.T1
    T2_actual = lab.T2
    T1_twin = T1_actual
    T2_twin = T2_actual
    Ta = T1_actual
    
    history = {
        'time': [], 
        'actual1': [], 'twin1': [], 
        'actual2': [], 'twin2': [],
        'residual': [], 'alarm': [], 'Q1': []
    }
    
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))
    
    for i in tclab.clock(n_steps, dt):
        # Programmed heating sequence
        if i < 30: Q1 = 0
        elif i < 600: Q1 = 80
        elif i < 900: Q1 = 0
        else: Q1 = 50
        
        Q2 = 0 # Keep second heater off
        
        lab.Q1(Q1)
        lab.Q2(Q2)
        T1_actual = lab.T1
        T2_actual = lab.T2
        
        # Calculate residual for Heater 1
        residual = T1_actual - T1_twin
        
        # 4. Update Anomaly Detector
        is_anomaly, fault_type, s_high, s_low = detector.update(residual, Q1)
        
        if is_anomaly:
            print(f"[{i}s] *** ALARM: {fault_type}! *** (Res: {residual:.2f}C)")
        
        # 5. Update Digital Twin (Dual Heater)
        dT1dt, dT2dt = tclab_dual_ode(T1_twin, T2_twin, Q1, Q2, Ta)
        T1_twin += dT1dt * dt
        T2_twin += dT2dt * dt
        
        # 6. Log and Save
        history['time'].append(i)
        history['actual1'].append(T1_actual)
        history['twin1'].append(T1_twin)
        history['actual2'].append(T2_actual)
        history['twin2'].append(T2_twin)
        history['residual'].append(residual)
        history['alarm'].append(is_anomaly)
        history['Q1'].append(Q1)
        
        csv_writer.writerow([i, Q1, Q2, T1_actual, T1_twin, T2_actual, T2_twin, residual, is_anomaly, fault_type])
        csv_file.flush()
        
        if i % 5 == 0:
            ax1.clear()
            ax1.plot(history['time'], history['actual1'], 'r-', label='T1 Reality')
            ax1.plot(history['time'], history['twin1'], 'r--', label='T1 Twin')
            ax1.set_title('Temperature: Digital Twin (Heater 1)')
            ax1.set_ylabel('Temp (C)')
            ax1.legend()
            
            ax2.clear()
            ax2.step(history['time'], history['Q1'], 'k-', label='Heater 1 Power')
            ax2.set_title('Heater Output (Q1)')
            ax2.set_ylabel('Power (%)')
            ax2.legend()
            
            ax3.clear()
            ax3.plot(history['time'], history['residual'], 'g-', label='Residual (T1)')
            if is_anomaly:
                ax3.set_facecolor((1.0, 0.9, 0.9))
            else:
                ax3.set_facecolor((1.0, 1.0, 1.0))
            ax3.axhline(10.0, color='r', linestyle='--')
            ax3.axhline(-10.0, color='r', linestyle='--')
            ax3.set_title('Fault Detection (CUSUM Monitoring)')
            ax3.set_ylabel('Residual (C)')
            ax3.set_xlabel('Time (s)')
            ax3.legend()
            fig.tight_layout()
            plt.pause(0.01)

# Save the final plot
plot_path = os.path.join(ROOT_DIR, 'data/tclab_plot.png')
plt.savefig(plot_path)
print(f"Final plot saved to {plot_path}")

csv_file.close()
print("Monitoring complete.")
