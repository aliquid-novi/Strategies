import time
import subprocess
from datetime import datetime, timedelta

def execute_granger_arima():
    print("Running Granger Arima")
    subprocess.run(["python", "granger_arima.py"])

def execute_daily_granger():
    print("Running Daily Granger")
    subprocess.run(["python", "granger_daily.py"])

def check_time_and_execute():
    current_utc_time = datetime.utcnow()

    # Convert to GMT+2
    current_gmt3_time = current_utc_time + timedelta(hours=2)
    
    current_gmt3_hour = current_gmt3_time.hour
    current_gmt3_minute = current_gmt3_time.minute

    # Check if it's one minute past a new hour
    if current_gmt3_minute == 1:

        # Daily script should run at 00:01 GMT+3
        if current_gmt3_hour == 0:
         execute_daily_granger()

        # 4-hourly script should run at 00:01, 04:01, 08:01, 12:01, 16:01, 20:01 GMT+3
        if current_gmt3_hour in [0, 4, 8, 12, 16, 20]:
            execute_granger_arima()

while True:
    check_time_and_execute()
    print("Pending...")
    current_utc_time = datetime.utcnow()
    # Convert to GMT+3 for displaying
    current_gmt3_time = current_utc_time + timedelta(hours=2)
    print(f"GMT+2 time is {current_gmt3_time}.")
    # Sleep for 60 seconds before checking again
    time.sleep(60)