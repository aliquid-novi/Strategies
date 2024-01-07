import pytz
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5

class ProfitLossControl:
    

    def __init__(self, max_daily_dd, profit_threshold, hedging_threshold, starting_equity):
        self.max_daily_dd = max_daily_dd
        self.profit_threshold = profit_threshold
        self.hedging_threshold = hedging_threshold
        self.starting_equity = starting_equity
    
    def start_equity(self):
        gmt_plus_2 = pytz.timezone('EET')  # Eastern European Time is typically GMT+2

        # Get the current time in UTC and convert it to GMT+2
        now_utc = datetime.now(timezone.utc)
        now_gmt_plus_2 = now_utc.astimezone(gmt_plus_2)

        # Create a datetime object for the reset time (00:00:01) on the same day in GMT+2
        reset_time_today = now_gmt_plus_2.replace(hour=0, minute=0, second=0)
        
        # Check if the current time is past today's reset time
        if now_gmt_plus_2 <= reset_time_today:

            # Set reset time to the start of the next day
            reset_time_tomorrow = reset_time_today + timedelta(days=1)
            countdown = reset_time_tomorrow - now_gmt_plus_2

            info = mt5.account_info()
            start_equity = info.equity

            print(f"Starting equity is {start_equity}")
            return start_equity

        else:
            # Calculate the countdown to today's reset time
            reset_time_tomorrow = reset_time_today + timedelta(days=1)
            countdown = reset_time_tomorrow - now_gmt_plus_2
            
            # Print the countdown
            print(f"{countdown} remaining until equity reset")

    def day_pnl(self, start_equity): # Return current days equity drawdow / profit as a percentage from the start of the day. 
        
        # Define the GMT+2 timezone
        gmt_plus_2 = pytz.timezone('EET')  # Eastern European Time is typically GMT+2

        # Get the current time in UTC and convert it to GMT+2
        now_utc = datetime.now(timezone.utc)
        time = now_utc.astimezone(gmt_plus_2)
        now_gmt_plus_2 = now_utc.astimezone(gmt_plus_2)

        # Create a datetime object for the reset time (00:00:01) on the same day in GMT+2
        reset_time_today = now_gmt_plus_2.replace(hour=0, minute=0, second = 0)

        # Calculate the countdown to the reset time
        if now_gmt_plus_2 >= reset_time_today:
            # If the current time is after the reset time, calculate the countdown to the next day's reset time
            reset_time_tomorrow = reset_time_today + timedelta(days=1)
            countdown = reset_time_tomorrow - now_gmt_plus_2
        else:
            # If the current time is before today's reset time, calculate the countdown to today's reset time
            countdown = reset_time_today - now_gmt_plus_2
        
        account_info = mt5.account_info()
        equity = account_info.balance + account_info.profit
        
        day_pnl = equity - start_equity

        print(day_pnl)
        print(f"Time until next reset: {countdown}")
        
        return int(day_pnl)
        
    def daily_dd_control(self, day_pnl): 
        
        if day_pnl <= (self.max_daily_dd / 1.35):
            print('Closing all positions for the day')
            close_all()
        else:
            print(f'Threshold not reached. Current day pnl percent: {day_pnl}')