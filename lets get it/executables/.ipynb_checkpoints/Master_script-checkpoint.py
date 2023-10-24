import time
import subprocess
from datetime import datetime, timedelta

current_utc_time = datetime.utcnow()

def execute_daily_script():
    print("Running Execute_Script.py")
    subprocess.run(["python", "Execute_Script.py"])

def execute_4hr_script():
    print("Running Execute_Script_4hr.py")
    subprocess.run(["python", "Execute_Script_4hr.py"])

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
    print("Pending...")
    print(f"UTC time is {current_utc_time}.")
    # Sleep for 120 seconds before checking again
    time.sleep(120)