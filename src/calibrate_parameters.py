import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gekko import GEKKO
import json
import os

# 1. Project Root and Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(ROOT_DIR, 'data/tclab_dyn_data2.csv')
params_path = os.path.join(ROOT_DIR, 'models/physics_parameters.json')

# 2. Import or generate data
if not os.path.exists(data_path):
    print(f"Data file {data_path} not found. Downloading from APMonitor...")
    url = 'https://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
    data = pd.read_csv(url)
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    data.to_csv(data_path, index=False)
else:
    data = pd.read_csv(data_path)

# 3. Create GEKKO Model
m = GEKKO(remote=False) # Use local solver
m.time = data['Time'].values

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

# Update with existing parameters if available
if os.path.exists(params_path):
    try:
        with open(params_path, 'r') as f:
            params.update(json.load(f))
    except Exception as e:
        print(f"Warning: Error loading {params_path}: {e}")

U = m.FV(value=params['U'], lb=1, ub=20)
alpha1 = m.FV(value=params['alpha1'], lb=0.003, ub=0.03)
alpha2 = m.FV(value=params['alpha2'], lb=0.002, ub=0.02)

U.STATUS = 1  
alpha1.STATUS = 1
alpha2.STATUS = 1

# Measured inputs
Q1 = m.MV(value=data['H1'].values)
Q2 = m.MV(value=data['H2'].values)

# State variables
TC1 = m.CV(value=data['T1'].values)
TC1.FSTATUS = 1    # minimize fstatus * (meas-pred)^2
TC2 = m.CV(value=data['T2'].values)
TC2.FSTATUS = 1    # minimize fstatus * (meas-pred)^2

Ta = m.Param(value=data['T1'].values[0] + 273.15)
mCp = m.Param(value=params['mCp']) 
A = m.Param(value=params['A'])
As = m.Param(value=params['As'])
eps = m.Param(value=params['epsilon'])
sigma = m.Const(params['sigma'])

# Heater temperatures in Kelvin
T1 = m.Intermediate(TC1 + 273.15)
T2 = m.Intermediate(TC2 + 273.15)

# Heat transfer between two heaters
Q_C12 = m.Intermediate(U * As * (T2 - T1))
Q_R12 = m.Intermediate(eps * sigma * As * (T2**4 - T1**4))

# Energy balances
m.Equation(TC1.dt() == (1.0/mCp) * (U * A * (Ta - T1) \
                    + eps * sigma * A * (Ta**4 - T1**4) \
                    + Q_C12 + Q_R12 \
                    + alpha1 * Q1))

m.Equation(TC2.dt() == (1.0/mCp) * (U * A * (Ta - T2) \
                    + eps * sigma * A * (Ta**4 - T2**4) \
                    - Q_C12 - Q_R12 \
                    + alpha2 * Q2))

# Options
m.options.IMODE   = 5
m.options.EV_TYPE = 2
m.options.NODES   = 2
m.options.SOLVER  = 3

# Solve
print("Starting GEKKO Parameter Estimation...")
m.solve(disp=False)

# 4. Save results to models/physics_parameters.json
params['U'] = float(U.value[0])
params['alpha1'] = float(alpha1.value[0])
params['alpha2'] = float(alpha2.value[0])
params['note'] = "Parameters re-estimated using GEKKO from latest dataset."

os.makedirs(os.path.dirname(params_path), exist_ok=True)
with open(params_path, 'w') as f:
    json.dump(params, f, indent=4)
print(f"Updated {params_path} with new values.")

# 5. Create validation plot
plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.plot(data['Time'], data['T1'], 'ro', label='T1 Measured', markersize=2)
plt.plot(m.time, TC1.value, 'r-', label='T1 Predicted', linewidth=2)
plt.plot(data['Time'], data['T2'], 'bo', label='T2 Measured', markersize=2)
plt.plot(m.time, TC2.value, 'b-', label='T2 Predicted', linewidth=2)
plt.ylabel('Temperature (°C)')
plt.legend()
plt.title('Parameter Estimation Results')

plt.subplot(2,1,2)
plt.plot(data['Time'], data['H1'], 'r-', label='Q1 (Heater 1)')
plt.plot(data['Time'], data['H2'], 'b-', label='Q2 (Heater 2)')
plt.ylabel('Heater Power (%)')
plt.xlabel('Time (s)')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(ROOT_DIR, 'data/calibration_plot.png'))
print(f"Validation plot saved to {os.path.join(ROOT_DIR, 'data/calibration_plot.png')}")
