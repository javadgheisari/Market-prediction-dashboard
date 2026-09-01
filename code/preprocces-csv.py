import pandas as pd 
from datetime import datetime
import os
from pathlib import Path
from easygui import fileopenbox

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




path_file = fileopenbox(title="choose csv file", filetypes=["*.csv", ])

address = path_file.split("Market-prediction-dashboard")[1]
# address = '/code/data/forex/EUR-USD_Minute_2025-10-14_UTC.csv'

file_name = address.split("/")[4].split("_")[0]
# new_address = "data/"+address
new_address = str(Path.cwd())+address
df = pd.read_csv(new_address)
# df.rename(columns={"Local time": "Date"}, inplace=True)
df.rename(columns={"Gmt time": "Asia/Tehran"}, inplace=True)
# print(df.iloc[:,0])

for date in df['Asia/Tehran']:
    new_time = " ".join(date.split(" ")[0:2])
    # df = df.replace(date,date[:10])
    # df = df.replace(date, datetime.strptime(date, '%d.%m.%Y %H:%M:%S.%f GMT%z').strftime('%Y-%m-%d %H:%M'))
    df = df.replace(date, datetime.strptime(new_time, '%d.%m.%Y %H:%M:%S.%f').strftime('%Y-%m-%d %H:%M'))

df = df.sort_values('Asia/Tehran').reset_index(drop=True)
# df = df[::-1] #reverse roes

print(df)
# save_address = "live-data/crypto/"+file_name+".csv"
save_address = str(Path.cwd())+"/code/data/forex/"+file_name+".csv"

df.to_csv(save_address, index=False)