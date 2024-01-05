# Import some libraries
import numpy as np
import pandas as pd
import scipy.optimize as opt
import regime_switch_model as rsm

# Create a regime-switching Heston model to accurately simulate VIX and market volatility

# Define the model dynamics

# Define the model dynamics and parameters. 
# The regime-switching Heston model assumes that there are two possible regimes for the volatility process: a low-volatility regime and a high-volatility regime. 
# The transition between the regimes is governed by a Markov chain with transition probabilities that depend on the current regime. 
# The model dynamics under the risk-neutral measure are given by:

def rs_heston_model (S0, V0, K, T, r, kappa1, theta1, eta1, rho1, kappa2, theta2, eta2, rho2, p11, p22):
    # S0: initial stock price
    # V0: initial variance
        # The initial variance should be calculated as the historical volatility and placed in the csv file as an initial value. One formula that can be used is the standard deviation of the natural log of the consecutive closing prices over a fixed historical timeframe.
    # K: strike price
    # T: maturity time
    # r: risk-free interest rate
    # kappa1: mean-reversion speed in regime 1
    # theta1: long-term mean in regime 1
    # eta1: volatility of volatility in regime 1
    # rho1: correlation between stock and variance in regime 1
    # kappa2: mean-reversion speed in regime 2
    # theta2: long-term mean in regime 2
    # eta2: volatility of volatility in regime 2
    # rho2: correlation between stock and variance in regime 2
    # p11: probability of staying in regime 1 given current regime is 1
    # p22: probability of staying in regime 2 given current regime is 2

    # Create a regime switching model object
    model = rsm.RegimeSwitchingModel ()

    # Set the initial values for the state variables
    model.set_initial_values (S0=S0, V0=V0)

    # Set the parameters for the model dynamics
    model.set_model_parameters (K=K, T=T, r=r,
                                kappa1=kappa1, theta1=theta1, eta1=eta1, rho1=rho1,
                                kappa2=kappa2, theta2=theta2, eta2=eta2, rho2=rho2,
                                p11=p11, p22=p22)

    # Return the model object
    return model


# To calibrate the regime-switching Heston model, we would need market data that reflects the option prices and their implied volatilities for both the S&P 500 and the VIX indices. 
# We would need the initial stock price, the initial variance, the strike price, the maturity time, and the risk-free interest rate for each option.

# One possible source of market data is Yahoo Finance, to download historical data for the S&P 500 and the VIX indices, as well as their options. 
# We can use Python libraries, such as pandas_datareader or yfinance, to access and manipulate the data programmatically.

    # Another possible source of market data is OptionMetrics, which is a commercial database that provides high-quality and comprehensive data on options and implied volatility for many underlying assets, including the S&P 500 and the VIX indices. 
    # We can access this database through some academic institutions or research centers, or purchase a subscription.


# Define the objective function

# Calibrate the model parameters to match the market option prices. 
# This can be done by minimizing an objective function that measures the difference between the model-implied option prices and the market-observed option prices. 
# The objective function can be defined as:

def objective_function (params, market_data):
    # params: a vector of model parameters to be calibrated
    # market_data: a pandas dataframe of market option prices with columns ['S0', 'V0', 'K', 'T', 'r', 'C']

    # Unpack the parameters from the vector
    kappa1, theta1, eta1, rho1, kappa2, theta2, eta2, rho2, p11, p22 = params

    # Initialize an empty list to store the model prices
    model_prices = []

    # Loop over each row of market data
    for i in range (len (market_data)):
        # Get the market data for each option
        S0 = market_data.loc[i,'S0']
        V0 = market_data.loc[i,'V0']
        K = market_data.loc[i,'K']
        T = market_data.loc[i,'T']
        r = market_data.loc[i,'r']

        # Create a regime switching Heston model object with the given parameters and market data
        model = rs_heston_model (S0=S0,V0=V0,K=K,T=T,r=r,
                                 kappa1=kappa1,theta1=theta1,
                                 eta1=eta1,rho1=rho1,
                                 kappa2=kappa2,
                                 theta2=theta2,
                                 eta2=eta2,rho2=rho2,
                                 p11=p11,p22=p22)

        # Calculate the model-implied option price using the analytic formula
        model_price = model.analytic_price ()

        # Append the model price to the list
        model_prices.append (model_price)

    # Convert the list of model prices to a numpy array
    model_prices = np.array (model_prices)

    # Get the market option prices from the dataframe
    market_prices = market_data['C'].values

    # Calculate the mean squared error between the model prices and the market prices
    mse = np.mean ((model_prices - market_prices)**2)

    # Return the mean squared error as the objective function value
    return mse


# Optimize the objective function using some numerical optimization method, such as Levenberg-Marquardt algorithm.


# Define some initial guess for the parameters:

# How were these initial values determined?
# I looked for some papers and articles that discussed the calibration and simulation of the Heston model or its variants, such as the regime-switching Heston model. 
# I also looked for some examples and implementations of the model in Python or other programming languages.

# From these sources, I found some typical ranges and values for the parameters that are commonly used or reported in the literature. 
# For example, I found that the long-term mean of the variance (theta) is usually between 0 and 0.1, the volatility of volatility (eta) is usually between 0 and 1, and the correlation between the stock and variance (rho) is usually between -1 and 1. 
# I also found that some parameters have some constraints or conditions, such as the Feller condition that requires 2*kappa*theta > eta^2 for each regime.



params0 = [1, 0.04, 0.2, -0.5, 2, 0.08, 0.4, 0.5, 0.9, 0.9]

# Define some bounds for the parameters
bounds = [(0, None), (0, None), (0, None), (-1, 1), (0, None), (0, None), (0, None), (-1, 1), (0, 1), (0, 1)]

# Define some constraints for the parameters
# For example, we can enforce the Feller condition for each regime: 2*kappa*theta > eta**2
def constraint1 (params):
    kappa1, theta1, eta1, rho1, kappa2, theta2, eta2, rho2, p11, p22 = params
    return 2*kappa1*theta1 - eta1**2

def constraint2 (params):
    kappa1, theta1, eta1, rho1, kappa2, theta2, eta2, rho2, p11, p22 = params
    return 2*kappa2*theta2 - eta2**2

constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2}]

# Load some market data from a csv file
    # NOTE: Obtain this data programmatically by executing the calculateVIX.py application which mimics the CBOE VIX algorithm
market_data = pd.read_csv ('market_data.csv')

# Optimize the objective function using Levenberg-Marquardt algorithm
result = opt.least_squares (objective_function,
                            x0=params0,
                            bounds=bounds,
                            args=(market_data,),
                            method='lm',
                            verbose=1)

# Print the optimal parameters and the objective function value
print ('Optimal parameters:', result.x)
print ('Objective function value:', result.cost)


# Define a function to simulate the regime switching Heston model dynamics

# Simulate the model dynamics using some numerical scheme that can handle the regime switching and the stochastic volatility features

def rs_heston_simulation (model,N):
    # model: a regime switching Heston model object
    # N: number of simulation paths

    # Get the model parameters from the model object
    S0 = model.S0
    V0 = model.V0
    K = model.K
    T = model.T
    r = model.r
    kappa1 = model.kappa1
    theta1 = model.theta1
    eta1 = model.eta1
    rho1 = model.rho1
    kappa2 = model.kappa2
    theta2 = model.theta2
    eta2 = model.eta2
    rho2 = model.rho2
    p11 = model.p11
    p22 = model.p22

    # Define some simulation parameters
    M = 100 # number of time steps
    dt = T/M # time step size

    # Initialize some arrays to store the simulation results
    S = np.zeros ((N,M+1)) # stock price paths
    V = np.zeros ((N,M+1)) # variance paths
    R = np.zeros ((N,M+1)) # regime paths

    # Set the initial values for each path
    S[:,0] = S0 # initial stock price
    V[:,0] = V0 # initial variance
    R[:,0] = np.random.choice ([1, 2], size=N) # initial regime

    # Generate some standard normal random numbers for simulation
    Z_S = np.random.randn (N,M) # for stock price process
    Z_V = np.random.randn (N,M) # for variance process

    # Loop over each time step
    for i in range (M):
        # Get the current values for each path
        S_t = S[:,i]
        V_t = V[:,i]
        R_t = R[:,i]

        # Get the regime-dependent parameters for each path
        kappa = np.where (R_t == 1, kappa1, kappa2)
        theta = np.where (R_t == 1, theta1, theta2)
        eta = np.where (R_t == 1, eta1, eta2)
        rho = np.where (R_t == 1, rho1, rho2)

        # Calculate the drift and diffusion terms for the stock price process
        drift_S = (r - 0.5 * V_t) * dt
        diff_S = np.sqrt (V_t) * np.sqrt (dt) * Z_S[:,i]

        # Calculate the drift and diffusion terms for the variance process
        drift_V = kappa * (theta - V_t) * dt
        diff_V = eta * np.sqrt (V_t) * np.sqrt (dt) * (rho * Z_S[:,i] + np.sqrt (1 - rho**2) * Z_V[:,i])

        # Update the stock price and variance for each path
        S[:,i+1] = S_t + drift_S + diff_S
        V[:,i+1] = V_t + drift_V + diff_V

        # Apply the Feller condition to avoid negative variance
        V[:,i+1] = np.maximum (V[:,i+1], 0)

        # Update the regime for each path using a binomial distribution
        R[:,i+1] = np.where (R_t == 1,
                             np.random.binomial (n=1, p=p11, size=N),
                             2 - np.random.binomial (n=1, p=p22, size=N))

        # Return the simulation results as a dictionary
        return {'S': S, 'V': V, 'R': R}

        