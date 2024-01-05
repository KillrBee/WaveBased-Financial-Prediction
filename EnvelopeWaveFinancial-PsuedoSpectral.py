import numpy as np
import cupy as cp
import pandas as pd
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
import requests
from bs4 import BeautifulSoup
#from textblob import TextBlob

# Load the financial data
file_path = 'NQESVIX_financial_data.csv'  # Update this path
financial_data = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

# Manually calculate the start date for the 32-week period
last_date = financial_data.index[-1]
start_date = last_date - pd.DateOffset(weeks=72)

# Select the financial data within this 32-week range
selected_financial_data = financial_data.loc[start_date:last_date]

# # Show the first few rows of the selected financial data
# print(selected_financial_data.head())

# Step 1: Set Temporal Resolution
num_steps = len(selected_financial_data)
print (f"Number of steps: {num_steps}")

dt = 24 * 60 * 60  # 4 hours in seconds

# Step 2: Adjust Spatial Grid
num_points = 2**8  # A power of 2 for efficient FFT
L = 10  # Size of the spatial domain
dx = L / num_points
x = np.linspace(0, L, num_points)

# Print the settings to confirm
print("Number of Steps:", num_steps)
print("Time Step (dt):", dt, "seconds")
print("Number of Spatial Points:", num_points)
print("Size of Spatial Domain (L):", L)
print("Spatial Resolution (dx):", dx)


# Preprocess the financial data (from Script 1)
vix_returns = financial_data['VIX_Close'].pct_change().fillna(0)
nq_volume_mean = financial_data['NQ_Volume'].mean()
nq_volume_std = financial_data['NQ_Volume'].std()
es_volume_mean = financial_data['ES_Volume'].mean()
es_volume_std = financial_data['ES_Volume'].std()

# Define coefficients P and Q (from Script 1)
P = vix_returns.rolling(window=21).std() * (252 ** 0.5)  # Annualized volatility
P = P.fillna(method='bfill')
Q = ((financial_data['NQ_Volume'] > (nq_volume_mean + 1.5 * nq_volume_std)) |
     (financial_data['ES_Volume'] > (es_volume_mean + 1.5 * es_volume_std))).astype(int)

# Normalize time and space dimensions (from Script 1)
time_index = financial_data.index
time_values = (time_index - time_index[0]).total_seconds()
time_max = time_values.max()
time_normalized = time_values / time_max
# space_normalized = np.linspace(0, 1, len(time_normalized))

# Calculate the returns of VIX
vix_returns = financial_data['VIX_Close'].pct_change().fillna(0)

# Calculate the amplitude and phase from VIX returns
amplitude = np.abs(vix_returns) / np.abs(vix_returns).max()
phase = np.cumsum(vix_returns) / np.cumsum(np.abs(vix_returns)).max()

# Define the spatial dimension
price_levels = np.linspace(financial_data['VIX_Close'].min(), financial_data['VIX_Close'].max(), num_points)

# Initialize the wave function A
A = np.zeros((num_steps, num_points), dtype=complex)
for i in range(num_steps):
    # Find the closest price level to the VIX close price at time i
    idx = np.argmin(np.abs(price_levels - financial_data['VIX_Close'].iloc[i]))
    # Set the amplitude and phase at the corresponding price level
    A[i, idx] = amplitude[i] * np.exp(1j * phase[i])
    
    
## Initialize the wave function A based on VIX returns
amplitude = np.abs(vix_returns) / np.abs(vix_returns).max()
phase = np.cumsum(vix_returns) / np.cumsum(np.abs(vix_returns)).max()
A = amplitude * np.exp(1j * phase)

# Pseudo-spectral method parameters
k = 2 * np.pi * np.fft.fftfreq(num_points, dx)

# Select the last 1344 data points to match the number of steps in the simulation
A_subset = A.iloc[-num_steps:]
P_subset = P.iloc[-num_steps:]
Q_subset = Q.iloc[-num_steps:]

# Ensure that P and Q have the correct shape
A_preadjust = A_subset.values.reshape(-1, 1)
P_adjusted = P_subset.values.reshape(-1, 1)
Q_adjusted = Q_subset.values.reshape(-1, 1)

# Broadcast P and Q across the spatial dimension
P_broadcasted = P_adjusted * np.ones((num_steps, num_points))
Q_broadcasted = Q_adjusted * np.ones((num_steps, num_points))

# Ensure A has the correct shape
A_adjusted = A_preadjust * np.ones((num_steps, num_points))

# Pseudo-spectral method for solving the NLS equation
def solve_nls(A, P, Q, dt, num_steps, k):
    A = A.copy()
    A_history = [A.copy()]
    k = cp.asarray(k)  # Convert k to a CuPy array
    
    for _ in range(num_steps):
        A_k = cp.fft.fft(A)
        A_k = A_k * cp.exp(-1j * P * k**2 * dt / 2)
        A = cp.fft.ifft(A_k)
        A = A * cp.exp(-1j * Q * cp.abs(A)**2 * dt)
        A_k = cp.fft.fft(A)
        A_k = A_k * cp.exp(-1j * P * k**2 * dt / 2)
        A = cp.fft.ifft(A_k)
        A_history.append(A.copy())
    
    return cp.array(A_history)

# Solve the NLS equation using CuPy
A_gpu = cp.asarray(A_adjusted)  # Transfer A to GPU
P_gpu = cp.asarray(P_broadcasted)  # Transfer P to GPU
Q_gpu = cp.asarray(Q_broadcasted)  # Transfer Q to GPU
k_gpu = cp.asarray(k)  # Transfer k to GPU

# Solve the NLS equation
A_history = solve_nls(A_gpu, P_gpu, Q_gpu, dt, num_steps, k)

#A_history = cp.asnumpy(A_history)

# Define parameters for the Nonlinear Schrodinger Equation (from Script 2)
#num_points = 256
#num_steps = 500
#L = 10
#dx = L / num_points
dt = 0.02
x = cp.linspace(0, L, num_points)
t = cp.linspace(0, num_steps*dt, num_steps)
V0 = 1
P, Q = 1, 1  # You can adjust these values as needed

# Define the Nonlinear Schrödinger Equation (from Script 2)
def NLS(t, psi):
    psi = cp.asarray(psi).reshape((2, -1))
    A = psi[0] + 1j * psi[1]
    dAdt = 1j * (-0.5 * cp.gradient(cp.gradient(A, dx), dx) + V0 * cp.abs(A)**2 * A + P * A + 1j * Q * A)
    result = cp.vstack((cp.real(dAdt), cp.imag(dAdt))).flatten()
    return cp.asnumpy(result)
    
# Initial condition (from Script 2)
A0 = cp.exp(-0.5 * (x - L/2)**2)
psi0 = cp.vstack((cp.real(A0), cp.imag(A0))).flatten()

A0 = cp.asnumpy(A0)
psi0 = cp.asnumpy(psi0)

# Solve the Nonlinear Schrödinger Equation (from Script 2)
sol = solve_ivp(NLS, [cp.asnumpy(t[0]), cp.asnumpy(t[-1])], psi0, t_eval=cp.asnumpy(t), method='RK45')

# Extract the amplitude of the wave function (from Script 2)
A = sol.y[0] + 1j * sol.y[1]
#A = A.reshape((num_steps, num_points))
envelope = np.abs(A)

# Debug code
print("Shape of envelope:", envelope.shape)
print("Shape of A_history:", A_history.shape)
# Time Series Analysis: Plotting the Envelope for Selected Points in Space (from Script 2)
space_points = [0, num_points // 4, num_points // 2, 3 * num_points // 4, -1]
# plt.figure(figsize=(12, 6))
# for sp in space_points:
    # plt.plot(cp.asnumpy(cp.abs(A_history[:,:,0])), label=f'Wave Function History : Space Point {sp/num_points:.2f}')

# plt.title('Time Series Analysis of the Envelope')
# plt.xlabel('Time Step')
# plt.ylabel('Amplitude')
# #plt.legend()
# plt.show()
# Creating a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(t, x, np.array([cp.asnumpy(cp.abs(A_history)), np.abs(envelope)]), cmap='viridis', linewidth=0, antialiased=False)

# Adding labels and title
ax.set_xlabel('Time Steps')
ax.set_ylabel('Space Points')
ax.set_zlabel('Amplitude')
ax.set_title('3D Surface Plot of Wave Function History')

# Adding a color bar
fig.colorbar(surf)

plt.show()
# # Web Scraping and Sentiment Analysis (from Script 1)
# def scrape_financial_news(date):
    # url = f"https://example-finance-news-website.com/{date}"
    # response = requests.get(url)
    # news_texts = []
    # if response.status_code == 200:
        # soup = BeautifulSoup(response.content, 'html.parser')
        # articles = soup.find_all('article', class_='news-article')
        # news_texts = [article.get_text() for article in articles]
    # else:
        # print("Failed to retrieve news articles")
    # return news_texts

# def perform_sentiment_analysis(news_texts):
    # sentiments = [TextBlob(article).sentiment.polarity for article in news_texts]
    # average_sentiment = np.mean(sentiments)
    # return average_sentiment

# def analyze_financial_data(financial_data):
    # # Implement your financial data analysis and anomaly detection here
    # # ...
    # # Return the significant dates as a list of strings
    # significant_dates = ["2023-10-05", "2023-10-12"]  # Example dates
    # return significant_dates

# # Main analysis workflow (from Script 1)
# financial_data = None  # Load your financial data here
# significant_dates = analyze_financial_data(financial_data)

# for date in significant_dates:
    # news_texts = scrape_financial_news(date)
    # average_sentiment = perform_sentiment_analysis(news_texts)
    # print(f"Date: {date}, Average Sentiment: {average_sentiment}")
