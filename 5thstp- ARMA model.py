# Importing necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

# Read data from csv
dataset = pd.read_csv('TRVDATAFILTERED-300POINTS.csv')


# Loop to fit and predict multiple times
for i in range(1, 6):
    
    AR_order = i
    MA_order = 3
    
    order = (AR_order, 0, MA_order)  # ARIMA order: (p, d, q) where for ARMA d = 0

    # Create an ARMA model with order (p, q)
    arma_model = sm.tsa.ARIMA(dataset, order=order,trend= 'n')

    # Fit the model to the data
    arma_model_fit = arma_model.fit()

    # Predict using the model
    y_predicted = arma_model_fit.predict(start=300, end=400)

    # Plot the data and the predicted values
    plt.plot(dataset, color='b', label='Original Data')
    plt.plot(y_predicted, color='r', label='ARMA Predicted AVG')
    plt.legend()
    #plt.title(f'ARMA Model Iteration: {i}')
    plt.title(f'AR_order = {i}, MA_order = 3')
    plt.show()

    # Print a summary of the model
    print(arma_model_fit.summary())
