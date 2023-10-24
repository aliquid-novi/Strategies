import time
import subprocess
from datetime import datetime, timedelta
import pytz

def execute_inter_lagged_script():
    print("Running Inter-Lagged.py")
    subprocess.run(["python", "Inter-Lagged.py"])


def check_time_and_execute():
    current_utc_time = datetime.utcnow()
    current_utc_hour = current_utc_time.hour
    current_utc_minute = current_utc_time.minute

    # Check if it's one minute past a new hour
    if current_utc_minute == 1:
        # Daily script should run at 22:01 UTC
        if current_utc_hour == 22:
            execute_daily_script()

        # 4-hourly script should run at 00:01, 04:01, 08:01, 12:01, 16:01, 20:01 UTC
        if current_utc_hour in [0, 4, 8, 12, 16, 20]:
            execute_4hr_script()

while True:
    check_time_and_execute()
    execute_inter_lagged_script()  # This will run every loop iteration

    now = datetime.now(pytz.timezone('Europe/Helsinki'))  # use the appropriate timezone
    print("Pending...")
    print(f"Local time is {now}.")

    # Sleep for 240 seconds (4 minutes) before checking again
    time.sleep(240)