#RNN Demo

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the training set
dataset_train = pd.read_csv('C:/Users/Mashrur/Desktop/PERSONAL/Programming/Deep_Learning_A_Z/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/Recurrent_Neural_Networks/EUR_USD_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

#Setting up the data structure
X_train = []
y_train = []

for i in range (5, 1565):
    X_train.append(training_set_scaled[i-5:i, 0])
    y_train.append(training_set_scaled[i,0])

print(X_train)
X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping the arrays
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Importing keras
from keras.models import Sequential
from keras. layers import Dense
from keras. layers import LSTM
from keras. layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True, input_shape= (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM layer and dropout regularisation
regressor.add(LSTM(units = 50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units = 1))

#Compiling the neural Network
regressor.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['mse'])

#fitting the rnn to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size=16)

#Visualizing the results
dataset_test  = pd.read_csv('C:/Users/Mashrur/Desktop/PERSONAL/Programming/Deep_Learning_A_Z/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 3 - Recurrent Neural Networks (RNN)/Section 12 - Building a RNN/Recurrent_Neural_Networks/EUR_USD_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

#predicting next month
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 5:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []

#predicting each financial day in January
for i in range(5, 275):
    X_test.append(inputs[i - 5:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real EURUSD Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted EURUSD Stock Price')
plt.title('EURUSD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('EURUSD Stock Price')
plt.legend()
plt.show()
