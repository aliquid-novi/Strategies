# Strategy Notes 

## T1

Multiple Linear Regression Model didn't work as nearly as well as previously thought - in fact didn't work nearly at all, with 90% + days losing money regardless of position sizing. It was however trading on the 4hr, when testing etc was done on the daily.


## BIS Statistics Hypothesis

EER Values published by the BIS is able to forecast Foreign Exchange Prices. By conducting statistical tests such as the Granger Causality test, a p-value of less than 0.05 to represent a Granger-Causality relationship will be present. 

Therefore, further study upon the relationship of EER Values will be able to assist in forecasting FX Prices on both the Daily and Weely timeframe.

## Correlation Mean Reversion

Descriptive statistics, such as correlation, has mean reverting properties in which alpha can be found across another instrument that demonstrates a Granger-Casuality relationship.

Relationships found so far:
- Rolling variance (Period of 3) and spread of returns ('diff') at lags 1, 2 for EURUSD and AUDUSD. Other results:
    - EURUSD x GBPUSD:
        - Lags 2,3,4 5, 
- Rolling_corr_returns and diff (for eurusd x gbpusd) has a relationship with diff on lags 4 onwards 