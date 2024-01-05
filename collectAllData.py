import yfinance as yf
import pandas as pd

# Define the start date for fetching the historical data
start_date = "2020-01-01"
end_date ="2023-10-27"

# Fetching the data for ES=F (S&P 500 futures)
es_data = yf.download("ES=F", start=start_date, end=end_date, interval="1d")

# Fetching the data for NQ=F (NASDAQ 100 futures)
nq_data = yf.download("NQ=F", start=start_date, end=end_date, interval="1d")

# Fetching the data for ^VIX
vix = yf.download("^VIX", start=start_date, end=end_date, interval="1d")

# Resampling the ES=F data to 4-hour intervals
es_resampled = es_data.resample('24H').agg({'Open': 'first', 'Close': 'last', 'Volume': 'sum'}).ffill()

# Resampling the NQ=F data to 4-hour intervals
nq_resampled = nq_data.resample('24H').agg({'Open': 'first', 'Close': 'last', 'Volume': 'sum'}).ffill()

# Resample VIX data to 4-hour intervals
vix_4h = vix.resample('24H').agg({'Open': 'first', 'Close': 'last'}).ffill()

    
# # Aligning with the VIX data
# aligned_es = es_resampled.reindex(vix_4h.index, method='ffill')
# aligned_nq = nq_resampled.reindex(vix_4h.index, method='ffill')

# Creating a combined DataFrame
combined_data = pd.DataFrame({
    'ES_Open': es_data['Open'],
    'ES_Close': es_data['Close'],
    'ES_Volume': es_data['Volume'],
    'NQ_Open': nq_data['Open'],
    'NQ_Close': nq_data['Close'],
    'NQ_Volume': nq_data['Volume'],
    'VIX_Open': vix['Open'],
    'VIX_Close': vix['Close']
    
})

# Save Data to CSV file
combined_data.to_csv('NQESVIX_financial_data.csv')
