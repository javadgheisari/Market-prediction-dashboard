from django.shortcuts import render
from django.views import View
from dashboard.forms import NameForm
from io import BytesIO
import base64

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD

class DashboardView(View):
    template_name = 'dashboard.html'

    def get(self, request):
        return render(request, self.template_name)

class PredictView(View):
    template_name = 'predict.html'
    form_class = NameForm

    def get(self, request, symbol_name):

        context = {
            'symbol_name': symbol_name,
            'form': self.form_class()
        }
        return render(request, self.template_name, context)


    def post(self, request, symbol_name):
        form = self.form_class(request.POST)
        if form.is_valid():
            model_name = form.cleaned_data['model_name']
            predict_days = form.cleaned_data['predict_days']

            address = 'D:/umz/دروس/ترم8/Project/mysite/dashboard/data/new/'+symbol_name+'.csv'
            df = pd.read_csv(address)
            coin_name = symbol_name
            df = df.sort_values('Date').reset_index(drop=True)

            # df.shape

            df['Close'] = df['Close'].astype(float)
 
            # limmit count of rows
            num_shape = round((len(df.index))*0.5)

            # split to odd & even for scattering
            train = df.iloc[1::2, 1:2].values
            test = df.iloc[::2, 1:2].values


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
            
            try:
                model = load_model(f"D:/umz/دروس/ترم8/Project/mysite/dashboard/models/{model_name}/{symbol_name}/")

            except OSError:

                if model_name == "lstm":

                    # Initializing the Recurrent Neural Network
                    model = Sequential()

                    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
                    # model.add(Attention(return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 50, return_sequences = True))
                    # model.add(Attention(return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 50, return_sequences = True))
                    # model.add(Attention(return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 50, return_sequences = True))
                    # model.add(Attention(return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 50))
                    model.add(Dropout(0.2))

                    # Adding the output layer
                    model.add(Dense(units = 1))
                    model.summary()


                    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
                    model.fit(X_train, y_train, epochs = 10, batch_size = 128)
                
                elif model_name == "gru":

                    # The GRU architecture
                    model = Sequential()

                    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
                    model.add(Dropout(0.2))

                    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
                    model.add(Dropout(0.2))

                    model.add(GRU(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
                    model.add(Dropout(0.2))

                    model.add(GRU(units=50))
                    model.add(Dropout(0.2))

                    model.add(Dense(units=1))
                    model.summary()

                    model.compile(optimizer='adam', loss='mean_squared_error')
                    model.fit(X_train, y_train, epochs=15, batch_size=128)

                model.save(f"D:/umz/دروس/ترم8/Project/mysite/dashboard/models/{model_name}/{symbol_name}/")



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

            # n day predict
            pred_ = predict[-1].copy()
            prediction_full = []
            window = 60
            df_copy = df.iloc[:, 1:2][1:].values

            predict_days = round(predict_days/2)

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
                df_date_add = pd.to_datetime(df_date['Date'].iloc[-1]) + pd.DateOffset(hours=2)
                df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d %H:%M")], columns=['Date'])
                df_date = pd.concat([df_date, df_date_add])
            df_date = df_date.reset_index(drop=True)


            # plt.figure(figsize=(20,7))
            plt.plot(df['Date'].values[num_shape+50:], df_volume[num_shape+50:], color = 'red', label = f'Real {coin_name} Price')
            plt.plot(df_date['Date'][-prediction_full_new.shape[0]+50:].values, prediction_full_new[50:], color = 'blue', label = f'Predicted {coin_name} Price')
            plt.xticks(np.arange(0,df[num_shape:].shape[0],12))
            plt.title(f'{coin_name} Price Prediction')
            plt.xlabel('Date')
            plt.ylabel('Price ($)')
            # plt.legend()
            # plt.show()

            # # Generate the plot
            # x = [1, 2, 3, 4, 5]
            # y = [3, 5, 2, 6, 1]
            # plt.plot(x, y)
            # plt.xlabel('X')
            # plt.ylabel('Y')
            # plt.title('Sample Plot')
            
            # Save the plot to a BytesIO object
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            # Convert the plot image to base64
            buffer.seek(0)
            plot_data = base64.b64encode(buffer.read()).decode('utf-8')

            context = {
                'symbol_name': symbol_name,
                'model_name': model_name,
                'predict_days': predict_days,
                'form': form,
                'plot_data': plot_data
            }
        else:
            context = {
                'symbol_name': symbol_name,
                'form': form
            }
        return render(request, self.template_name, context)
