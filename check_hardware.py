import tclab
import sys

try:
    lab = tclab.TCLab()
    print("Connected to TCLab hardware.")
    lab.close()
except Exception as e:
    print(f"Caught exception: {type(e).__name__}: {e}")
    sys.exit(1)
