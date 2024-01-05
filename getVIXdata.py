import yfinance as yf
import pandas as pd

def fetch_vix_data():
    # Fetch VIX data
    vix = yf.download("^VIX", start="2021-04-21", end="2023-10-26", interval="1d")
    
    # Resample to 4-hour data
    vix_4h = vix['Close'].resample('4H').ffill()
    
    # Calculate 32-week moving average
    vix_4h_moving_avg = vix_4h.rolling(window=32*5*4).mean()
    
    # Save to CSV
    vix_4h_moving_avg.to_csv("vix_4h_moving_avg.csv")
    
    print("VIX data fetched and saved to 'vix_4h_moving_avg.csv'")

if __name__ == "__main__":
    fetch_vix_data()
