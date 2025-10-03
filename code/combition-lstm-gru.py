import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

address = 'crypto-data/yahoo-dataset/ETH-USD.csv'
df = pd.read_csv(address)
# coin_name = address.split("/")[1].split(".")[0].split("_")[1]
coin_name = address.split("/")[1].split("-")[0]
df = df.sort_values('Date').reset_index(drop=True)

df.head()

# df = df.drop(columns=['SNo', 'Name', 'Symbol'])

df.shape

# for date in df['Date']:
#     df = df.replace(date,date[:10])
    # print(date)
# df.replace(df['Date'][0],'sfsf')
# print(df['Date'])

df['Close'] = df['Close'].astype(float)

# plt.figure(figsize=(20,7))
# plt.plot(df['Date'].values, df['Close'].values, label = 'Aave Stock Price', color = 'red')
# plt.xticks(np.arange(0,df.shape[0],20))
# plt.xlabel('Date')
# plt.ylabel('Close ($)')
# plt.legend()
# plt.show()

# limmit count of rows
num_shape = round((len(df.index))*0.5)
# if num_shape%2 == 0:
#     num_shape = round((len(df.index))*0.5) - 1
# print(num_shape)

# train = df.iloc[:num_shape, 1:2].values
# test = df.iloc[num_shape:, 1:2].values

# split to odd & even for scattering
train = df.iloc[1::2, 1:2].values
test = df.iloc[::2, 1:2].values

# print(train, '---\n', test)

sc = MinMaxScaler(feature_range = (0, 1))
train_scaled = sc.fit_transform(train)

X_train = []

#Price on next day
y_train = []

window = 60

for i in range(window, num_shape):
    X_train_ = np.reshape(train_scaled[i-window:i, 0], (window, 1))
    X_train.append(X_train_)
    try:
        y_train.append(train_scaled[i, 0])
    except:
        y_train.append(train_scaled[i-1, 0])
X_train = np.stack(X_train)
y_train = np.stack(y_train)

################# LSTM ##################

# Initializing the Recurrent Neural Network
model = Sequential()
#Adding the first LSTM layer with a sigmoid activation function and some Dropout regularization
#Units - dimensionality of the output space

model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))


model.add(LSTM(units = 50))
model.add(Dropout(0.2))

# Adding the output layer
model.add(Dense(units = 1))
model.summary()


model.compile(optimizer = 'adam', loss = 'mean_squared_error')
model.fit(X_train, y_train, epochs = 20, batch_size = 128);


df_volume = np.vstack((train, test))

inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

num_2 = df_volume.shape[0] - num_shape + window

X_test = []

for i in range(window, num_2):
    X_test_ = np.reshape(inputs[i-window:i, 0], (window, 1))
    X_test.append(X_test_)
    
X_test = np.stack(X_test)


predict = model.predict(X_test)
predict = sc.inverse_transform(predict)

predict_length = len(predict)
test_length = len(test)

# print(type(test))
# print(type(predict))


if predict_length == test_length:
    diff = predict - test
elif predict_length > test_length:
    # predict.pop()
    predict = predict[:predict_length-1]
    diff = predict - test
else:
    # test.pop()
    test = test[:test_length-1]
    diff = predict - test

# print(len(test))
# print(len(predict))


# diff = predict - test


print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))



# plt.figure(figsize=(20,7))
# # print(df_volume[200:])
# plt.plot(df['Date'].values[num_shape:], df_volume[num_shape:], color = 'red', label = 'Real Aave Price')
# plt.plot(df['Date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted Aave Price')
# plt.xticks(np.arange(100,df[num_shape+(round(num_shape*0.1)):].shape[0],50))
# plt.title('Bitcoin Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.show()

#predict n days
pred_ = predict[-1].copy()
prediction_full = []
window = 60
df_copy = df.iloc[:, 1:2][1:].values

predict_days = int(input("predict_days (max:30 D):"))

for j in range(predict_days):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:num_shape]
    test_ = df_[num_shape:]
    
    df_volume_ = np.vstack((train_, test_))

    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
    inputs_ = inputs_.reshape(-1,1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(window, num_2):
        X_test_3 = np.reshape(inputs_[k-window:k, 0], (window, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2)
    predict_ = model.predict(X_test_)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]


prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))


df_date = df[['Date']]
# print(df_date)
for h in range(predict_days):
    df_date_add = pd.to_datetime(df_date['Date'].iloc[-1]) + pd.DateOffset(days=1)
    df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['Date'])
    df_date = pd.concat([df_date, df_date_add])
df_date = df_date.reset_index(drop=True)


# plt.figure(figsize=(20,7))
# plt.plot(df['Date'].values[num_shape:], df_volume[num_shape:], color = 'red', label = f'Real {coin_name} Price')
# plt.plot(df_date['Date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = f'Predicted {coin_name} Price')
# plt.xticks(np.arange(0,df[num_shape:].shape[0],7))
# plt.title(f'{coin_name} Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.show()


########### GRU ################

# The GRU architecture
modelGRU = Sequential()

modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
modelGRU.add(Dropout(0.2))

modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
modelGRU.add(Dropout(0.2))

modelGRU.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
modelGRU.add(Dropout(0.2))

modelGRU.add(GRU(units=50))
modelGRU.add(Dropout(0.2))

modelGRU.add(Dense(units=1))
modelGRU.summary()

modelGRU.compile(optimizer='adam', loss='mean_squared_error')
modelGRU.fit(X_train, y_train, epochs=50, batch_size=128)

predict = modelGRU.predict(X_test)
predict = sc.inverse_transform(predict)

predict_length = len(predict)
test_length = len(test)

if predict_length == test_length:
    diff = predict - test
elif predict_length > test_length:
    # predict.pop()
    predict = predict[:predict_length-1]
    diff = predict - test
else:
    # test.pop()
    test = test[:test_length-1]
    diff = predict - test
print("MSE:", np.mean(diff**2))
print("MAE:", np.mean(abs(diff)))
print("RMSE:", np.sqrt(np.mean(diff**2)))


# plt.figure(figsize=(20,7))
# # print(df_volume[200:])
# plt.plot(df['Date'].values[num_shape:], df_volume[num_shape:], color = 'red', label = 'Real Aave Price')
# plt.plot(df['Date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted Aave Price')
# plt.xticks(np.arange(0,df[num_shape+round(num_shape*0.1):].shape[0],90))
# plt.title('Bitcoin Price Prediction')
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.legend()
# plt.show()


# 10 day predict
pred_ = predict[-1].copy()
prediction_full = []
window = 60
df_copy = df.iloc[:, 1:2][1:].values

for j in range(predict_days):
    df_ = np.vstack((df_copy, pred_))
    train_ = df_[:num_shape]
    test_ = df_[num_shape:]
    
    df_volume_ = np.vstack((train_, test_))

    inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
    inputs_ = inputs_.reshape(-1,1)
    inputs_ = sc.transform(inputs_)

    X_test_2 = []

    for k in range(window, num_2):
        X_test_3 = np.reshape(inputs_[k-window:k, 0], (window, 1))
        X_test_2.append(X_test_3)

    X_test_ = np.stack(X_test_2)
    predict_ = modelGRU.predict(X_test_)
    pred_ = sc.inverse_transform(predict_)
    prediction_full.append(pred_[-1][0])
    df_copy = df_[j:]


prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))

df_date = df[['Date']]

for h in range(20):
    df_date = pd.concat([df_date, df_date_add])
df_date = df_date.reset_index(drop=True)



plt.figure(figsize=(20,7))
plt.plot(df['Date'].values[num_shape:], df_volume[num_shape:], color = 'red', label = 'Real Bitcoin Price')
plt.plot(df_date['Date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = 'Predicted Bitcoin Price')
plt.xticks(np.arange(0,df_date[num_shape:].shape[0],7))
plt.title('Bitcoin Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.show()