import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import find_peaks

# Fetch VIX data
def fetch_vix_data():
    vix = yf.download("^VIX", start="2020-01-01", end="2023-01-01", interval="1d")
    vix_4h = vix['Close'].resample('4H').ffill()
    return vix_4h

# Fetch S&P 500 futures volume data
def fetch_es_data():
    es = yf.download("ES=F", start="2020-01-01", end="2023-01-01", interval="1d")
    es_4h = es['Volume'].resample('4H').ffill()
    return es_4h

# Calculate additional parameters
def calculate_parameters(vix_data, es_data):
    aligned_data = pd.concat([vix_data, es_data], axis=1, join='inner')
    aligned_data.columns = ['VIX', 'Volume']
    
    aligned_data['Amplitude'] = aligned_data['VIX']
    
    peaks, _ = find_peaks(aligned_data['VIX'])
    troughs, _ = find_peaks(-aligned_data['VIX'])
    extrema = np.sort(np.concatenate((peaks, troughs)))
    lengths = np.diff(extrema) * 4  # converting index differences to hours
    aligned_data['Length'] = np.nan
    aligned_data.loc[aligned_data.index[extrema[1:]], 'Length'] = lengths
    aligned_data['Length'] = aligned_data['Length'].fillna(method='ffill')
    
    aligned_data['Wave Number'] = 1 / aligned_data['Length']
    aligned_data['Phase'] = (aligned_data.index - aligned_data.index[0]).total_seconds() / (aligned_data['Length'] * 3600) * 2 * np.pi
    aligned_data['Phase'] = aligned_data['Phase'] % (2 * np.pi)
    
    water_depth = aligned_data['VIX'].mode().iloc[0]
    mean_volume = aligned_data['Volume'].mean()
    aligned_data['Currents/Wind'] = np.where(aligned_data['Volume'] > mean_volume, 'Above Mean', 'Below Mean')
    
    return aligned_data, water_depth

# Implement Wave Prediction Model
def wave_prediction_model(aligned_data):
    aligned_data['SMA'] = aligned_data['VIX'].rolling(window=20).mean()
    aligned_data['Cyclic Component'] = np.sin(aligned_data['Phase'])
    aligned_data['Returns'] = aligned_data['VIX'].pct_change()
    aligned_data['Volatility'] = aligned_data['Returns'].rolling(window=20).std()
    aligned_data['Prediction'] = aligned_data['SMA'] + aligned_data['Cyclic Component'] * aligned_data['Volatility']
    return aligned_data

# Main function to run the script
def main():
    vix_data = fetch_vix_data()
    es_data = fetch_es_data()
    aligned_data, water_depth = calculate_parameters(vix_data, es_data)
    results = wave_prediction_model(aligned_data)
    print("Water Depth (Mode of VIX):", water_depth)
    print(results[['VIX', 'SMA', 'Cyclic Component', 'Volatility', 'Prediction']].tail())

if __name__ == "__main__":
    main()
