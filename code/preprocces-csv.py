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






address = 'XLMUSD_Candlestick_2_Hour_BID_01.08.2022-05.08.2023.csv'
file_name = address.split("USD")[0]
new_address = "data/"+address
df = pd.read_csv(new_address)
df.rename(columns={"Local time": "Date"}, inplace=True)


for date in df['Date']:
    # df = df.replace(date,date[:10])
    df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f GMT%z').strftime('%Y-%m-%d %H:%M'))

df = df.sort_values('Date').reset_index(drop=True)
# df = df[::-1] #reverse roes

# print(df)

save_address = "data/new/"+file_name+".csv"
df.to_csv(save_address, index=False)