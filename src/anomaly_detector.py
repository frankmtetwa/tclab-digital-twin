import numpy as np

class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) Anomaly Detector.
    Detects small, persistent shifts in the mean of the residual.
    """
    def __init__(self, threshold=5.0, drift=0.5):
        """
        threshold: The value at which an alarm is triggered (H).
        drift: The slack or 'allowance' parameter (k). 
               Changes smaller than 'drift' are ignored.
        """
        self.threshold = threshold
        self.drift = drift
        self.s_high = 0.0  # Cumulative sum for positive shifts
        self.s_low = 0.0   # Cumulative sum for negative shifts
        self.alarm = False
        self.alarm_type = None

    def update(self, residual, Q):
        """
        Update the CUSUM sums with a new residual value.
        Includes basic fault diagnosis and auto-reset logic.
        """
        # Calculate high and low CUSUM
        self.s_high = max(0, self.s_high + residual - self.drift)
        self.s_low = min(0, self.s_low + residual + self.drift)

        # AUTO-RESET: If the current residual is within the drift zone, 
        # slowly bleed the sums toward zero or reset them to prevent "sticky" alarms.
        if abs(residual) < self.drift:
            self.s_high = 0.0
            self.s_low = 0.0

        # Check for threshold crossing
        if self.s_high > self.threshold:

            self.alarm = True
            if Q > 10:
                self.alarm_type = "Potential Sensor Drift (High) or Model Mismatch"
            else:
                self.alarm_type = "External Heat Source Detected"
        elif self.s_low < -self.threshold:
            self.alarm = True
            if Q > 10:
                self.alarm_type = "Heater Degradation or External Cooling"
            else:
                self.alarm_type = "Sensor Drift (Low) or External Cooling (Fan)"
        else:
            self.alarm = False
            self.alarm_type = None

        return self.alarm, self.alarm_type, self.s_high, self.s_low

    def reset(self):
        """Reset the cumulative sums."""
        self.s_high = 0.0
        self.s_low = 0.0
        self.alarm = False
        self.alarm_type = None
