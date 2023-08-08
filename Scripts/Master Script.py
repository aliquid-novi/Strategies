import schedule
import time 
import pytz
import threading
import datetime
import Interlagged
import MetaTrader5 as mt5
mt5.initialize(path="C:\Program Files\ICMarkets - MetaTrader 5\terminal64.exe")
account=51127988
password="Aar2frM7"
server = 'ICMarkets-Demo'

# Print some info for verification
print(mt5.terminal_info())
print(mt5.version())

aest = pytz.timezone('Australia/Sydney')
now = datetime.datetime.now(tz = aest).strftime("%I:%M%p %Z")
print(f"Master Script executed at {now}")

    
def func():
    print("Running 15m Pair Strat...")
    Interlagged.run()

# func()

schedule.every(2).seconds.do(func)

while True:
    aest = pytz.timezone('Australia/Sydney')
    now = datetime.datetime.now(tz = aest).strftime("%I:%M%p %Z")
    print(f"Checking for scheduled tasks... at {now}...")
    schedule.run_pending()
    time.sleep(30)