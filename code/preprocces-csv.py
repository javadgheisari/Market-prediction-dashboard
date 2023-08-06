import pandas as pd 
from datetime import datetime
import os

# file_lists = os.listdir('forex-data/')
# for address in file_lists:
    # file_name = address[:3]+'-'+address[3:6]
    # addr = 'forex-data/'+address
df = pd.read_csv('forex-data/EURUSD_6h.csv')
df.rename(columns={"Local time": "Date"}, inplace=True)


for date in df['Date']:
    # df = df.replace(date,date[:10])
    df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f GMT%z').strftime('%Y-%m-%d %H:%M'))

df = df.sort_values('Date').reset_index(drop=True)

# print(df)

df.to_csv('forex-data/new/EURUSD_6h.csv', index=False)
