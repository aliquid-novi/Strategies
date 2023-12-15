# Strategy Notes 

## T1

Multiple Linear Regression Model didn't work as nearly as well as previously thought - in fact didn't work nearly at all, with 90% + days losing money regardless of position sizing. It was however trading on the 4hr, when testing etc was done on the daily.


## BIS Statistics Hypothesis

EER Values published by the BIS is able to forecast Foreign Exchange Prices. By conducting statistical tests such as the Granger Causality test, a p-value of less than 0.05 to represent a Granger-Causality relationship will be present. 

Therefore, further study upon the relationship of EER Values will be able to assist in forecasting FX Prices on both the Daily and Weely timeframe.

### Relationships Found 
**Weekly** 

EER Correlation with AU is: 0.7901864448535375
EER Correlation with CA is: -0.8028662382263347
EER Correlation with CH is: -0.6972064531265869
EER Correlation with GB is: 0.6872038876573432
EER Correlation with JP is: -0.8162789646117149
EER Correlation with NZ is: 0.7759010905793639
EER Correlation with XM is: 0.75461984964802


## Correlation Mean Reversion

Descriptive statistics, such as correlation, has mean reverting properties in which alpha can be found across another instrument that demonstrates a Granger-Casuality relationship.

Relationships found so far:

EURUSD x GBPUSD
- Rolling variance (Period of 3) and spread of returns ('diff') at lags 1, 2 for EURUSD and AUDUSD. Other results:
    - EURUSD x GBPUSD:
        - Lags 2,3,4 5, 
- Rolling_corr_returns and diff (for eurusd x gbpusd) has a relationship with diff on lags 4 onwards 
- MA_Ratio x spread (all lags up to 10)
- rolling_corr_returns x spread lags 5+
- rolling_var x spread lag 3-4

- For its yen-equivalent (EURJPY/GBPJPY):
    - CSS_spread (lag 3-5, 7-15)
    - CSS_rolling_var (lag 2, 4 onwards)
    - CSS_MA_Ratio (All lags)
    - CSS_diff (All lags)
    
EURUSD
- rolling_var (lag 5 onwards)
- rolling_corr_returns (lag 5 onwards)
- spread (lag 7) 

- CSS_rolling_corr_returns(lag 6-8)
- CSS_rolling_var (lag 2:)
- CSS_MA_Ratio (lag 1 onwards)
- CSS_spread (lag 5)

