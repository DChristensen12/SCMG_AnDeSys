import time
import subprocess
import sys
from datetime import datetime

def run_monitor():
    """
    Infinite loop that triggers the GNN inference every 15 minutes.
    """
    print("--- SCMG Live Monitoring Service Started ---")
    
    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{now}] Starting detection cycle :)")
        
        try:
            # This calls main.py using the 'inference' mode
            # We use subprocess so each run starts with a clean memory state
            subprocess.run([sys.executable, "main.py", "--mode", "inference"], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"[{now}] Error during detection cycle: {e}")
        
        print(f"[{now}] Cycle complete. Sleeping for 15 minutes n^n")
        
        # 900 seconds = 15 minutes
        time.sleep(900)

if __name__ == "__main__":
    try:
        run_monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        sys.exit(0)