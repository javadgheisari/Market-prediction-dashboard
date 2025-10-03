import pandas as pd 
from datetime import datetime
import os

# file_lists = os.listdir('data/')
# for address in file_lists:
#     file_name = address.split("USD")[0]
#     # addr = 'forex-data/'+address
#     new_address = "data/"+address
#     print(file_name)
#     df = pd.read_csv(new_address)
#     df.rename(columns={"Local time": "Date"}, inplace=True)


#     for date in df['Date']:
#         # df = df.replace(date,date[:10])
#         df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f GMT%z').strftime('%Y-%m-%d %H:%M'))

#     df = df.sort_values('Date').reset_index(drop=True)
#     # df = df[::-1]

#     # print(df)

#     # df.to_csv('crypto-data/duckascopy/BTCUSD_3h_new.csv', index=False)
#     save_address = "data/new/"+file_name+".csv"
#     df.to_csv(save_address, index=False)






address = 'EURUSD_Candlestick_1_Hour_BID_11.10.2023-11.11.2023.csv'
file_name = address.split("USD")[0]
# new_address = "data/"+address
new_address = "C:/Users/Asus/Downloads/"+address
df = pd.read_csv(new_address)
# df.rename(columns={"Local time": "Date"}, inplace=True)
df.rename(columns={"Gmt time": "Date"}, inplace=True)


for date in df['Date']:
    # df = df.replace(date,date[:10])
    # df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f GMT%z').strftime('%Y-%m-%d %H:%M'))
    df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M'))

df = df.sort_values('Date').reset_index(drop=True)
# df = df[::-1] #reverse roes

# print(df)

# save_address = "live-data/crypto/"+file_name+".csv"
save_address = "live-data/forex/"+file_name+".csv"

df.to_csv(save_address, index=False)