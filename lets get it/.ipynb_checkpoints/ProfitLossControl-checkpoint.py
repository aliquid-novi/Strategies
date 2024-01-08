class ProfitLossControl:
    
    def __init__(self, max_daily_dd, profit_threshold, hedging_threshold, starting_equity):
        self.max_daily_dd = max_daily_dd
        self.profit_threshold = profit_threshold
        self.hedging_threshold = hedging_threshold
        self.starting_equity = starting_equity
    
    def equity_track(self):
        equity = pd.DataFrame()
        # Define the GMT+2 timezone
        gmt_plus_2 = pytz.timezone('EET')  # Eastern European Time is typically GMT+2

        # Get the current time in UTC and convert it to GMT+2
        now_utc = datetime.datetime.now(datetime.timezone.utc)
        time = now_utc.astimezone(gmt_plus_2)

        # Create a datetime object for the reset time (00:00:01) on the same day in GMT+2
        reset_time_today = now_gmt_plus_2.replace(hour=0, minute=0, second=1, microsecond=0)

        # Calculate the countdown to the reset time
        if now_gmt_plus_2 >= reset_time_today:
            # If the current time is after the reset time, calculate the countdown to the next day's reset time
            reset_time_tomorrow = reset_time_today + datetime.timedelta(days=1)
            countdown = reset_time_tomorrow - now_gmt_plus_2
        else:
            # If the current time is before today's reset time, calculate the countdown to today's reset time
            countdown = reset_time_today - now_gmt_plus_2
        
        account_info = mt5.account_info()
        equity = account_info.balance + account_info.profit

        print('Equity Recorded')
        # Output the countdown
        print(f"Time until next reset: {countdown}")
            
    # def cur_pnl(self):
        # days_start_equity = 
        # print('Current PNL:')