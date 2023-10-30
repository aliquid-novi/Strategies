import time
import subprocess
from datetime import datetime, timedelta
import pytz

def execute_correl_reversion():
    print("Running correl_reversion")
    subprocess.run(["python", "Correl_Reversion_h1.py"])


def check_time_and_execute():
    current_utc_time = datetime.utcnow()
    current_utc_hour = current_utc_time.hour
    current_utc_minute = current_utc_time.minute


while True:
    # Execute your scripts
    check_time_and_execute()
    execute_correl_reversion()

    # Get the current time in the 'Europe/Helsinki' timezone
    now = datetime.now(pytz.timezone('Europe/Helsinki'))

    # Calculate the time difference until the next hour
    next_hour = now + timedelta(hours=1)
    next_hour = next_hour.replace(minute=0, second=0, microsecond=0)
    sleep_time = (next_hour - now).seconds
    
    print("Pending...")
    print(f"Local time is {now}.")
    print(f"Sleeping for {sleep_time} seconds until the next hour.")

    # Sleep until the start of the next hour
    time.sleep(sleep_time)