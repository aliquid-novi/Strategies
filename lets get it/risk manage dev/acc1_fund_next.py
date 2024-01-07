import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
from risk_manage_main import ProfitLossControl
import pytz 
import time

mt5.initialize()
# Replace following with your MT5 Account Login
account = 533690
password = 'gywNZ76##' 
server = 'GrowthNext-Server'

def close_position(position):

    tick = mt5.symbol_info_tick(position.symbol)

    request = {
        "action" : mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_BUY if position.type == 1 else mt5.ORDER_TYPE_SELL,
        "price": tick.ask if position.type == 1 else tick.bid,
        "deviation": 20,
        "magic": 100,
        "comment": 'Regres Close',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling':mt5.ORDER_FILLING_IOC,

        }
    result = mt5.order_send(request)
    
def close_all():
    close_positions = []
    open_positions = mt5.positions_get()
    open_positions
    for i in open_positions:
        close_positions.append(i)
        
    for pos in close_positions:
        close_position(pos)
        
def run_script():
    obj = ProfitLossControl(-0.05, 0.05, 0.00125, 25000)

    print("Running script")
    start_equity = obj.start_equity() # Reset the start equity everyday at 9am AEST / 12:00AM GMT+2

    if start_equity != None:
        days_pnl = obj.day_pnl(start_equity)

        obj.daily_dd_control(days_pnl)

    else:
        print("No value for starting equity yet")

def wait_for_next_half_hour():
    now = datetime.now()

    # Check if the current minute is before or after the 30-minute mark
    if now.minute < 30:
        # Set next half-hour mark to the same hour but 30 minutes
        next_half_hour = now.replace(minute=30, second=0, microsecond=0)
    else:
        # Set next half-hour mark to the next hour and 0 minutes
        next_half_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

    wait_time = (next_half_hour - now).total_seconds()
    print(f"Current time is {now}")
    print(f"Waiting for {wait_time} seconds until the next half hour.")
    time.sleep(wait_time)

while True:
    run_script()
    wait_for_next_half_hour()
    