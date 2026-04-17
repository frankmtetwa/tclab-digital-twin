import numpy as np
import pandas as pd
import tclab
import time
import os
import matplotlib.pyplot as plt

# 1. Project Root and Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(ROOT_DIR, 'data/tclab_dyn_data2.csv')
os.makedirs(os.path.dirname(data_path), exist_ok=True)
n_steps = 601 

# 2. Define Heater Step Sequences
Q1d = np.zeros(n_steps)
Q1d[10:200] = 80
Q1d[200:280] = 20
Q1d[280:400] = 70
Q1d[400:] = 50

Q2d = np.zeros(n_steps)
Q2d[120:320] = 100
Q2d[320:520] = 10
Q2d[520:] = 80

# 3. Connect to TCLab
try:
    lab = tclab.TCLab()
    print("Connected to TCLab hardware.")
except Exception as e:
    print(f"Could not connect to hardware: {e}")
    print("Switching to TCLabModel (Simulation) for testing...")
    lab = tclab.TCLabModel()

# 4. Data Collection Loop
print(f"Starting 10-minute data collection. Saving to {data_path}...")
with lab:
    with open(data_path, 'w') as f:
        f.write('Time,H1,H2,T1,T2\n')
    
    start_time = time.time()
    try:
        for i in range(n_steps):
            lab.Q1(Q1d[i])
            lab.Q2(Q2d[i])
            t1, t2 = lab.T1, lab.T2
            
            if i % 10 == 0:
                print(f"Time: {i:3d}s | H1: {Q1d[i]:3.0f}% | H2: {Q2d[i]:3.0f}% | T1: {t1:5.2f}°C | T2: {t2:5.2f}°C")
            
            with open(data_path, 'a') as f:
                f.write(f"{i},{Q1d[i]},{Q2d[i]},{t1},{t2}\n")
            
            # Precise timing sync
            sleep_time = (i + 1) - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\nData collection interrupted. Saving partial data...")

# 5. Plotting Results
print("Data collection complete. Generating plot...")
data = pd.read_csv(data_path)

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(data['Time'], data['H1'], 'r-', label='Heater 1 (Q1)')
plt.plot(data['Time'], data['H2'], 'b--', label='Heater 2 (Q2)')
plt.ylabel('Heater Power (%)')
plt.title('Step Test: Input Power')
plt.legend(loc='best')
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(data['Time'], data['T1'], 'r.', label='Temperature 1 (T1)')
plt.plot(data['Time'], data['T2'], 'b.', label='Temperature 2 (T2)')
plt.ylabel('Temperature (°C)')
plt.xlabel('Time (s)')
plt.title('Step Test: Measured Response')
plt.legend(loc='best')
plt.grid(True)

plt.tight_layout()
plot_path = os.path.join(ROOT_DIR, 'data/tclab_dyn_meas2.png')
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
