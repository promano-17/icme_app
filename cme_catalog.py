#when you run this script, include the start time and end time 
# after the filename in the correct format


import sys
import requests
import json
# import datetime ## waiting to use this until we need to specify day
from bs4 import BeautifulSoup
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe
import wget

# Start month used to look through JSONs (YYYY-MM-DD).
start = sys.argv[1]
# End month used to look through JSONs (YYYY-MM-DD).
end = sys.argv[2]


# Variables for looping through start/end period.
start_year = int(start[0:4])
start_month = int(start[5:7])
start_day = int(start[8:10])
end_year = int(end[0:4])
end_month = int(end[5:7])
end_day = int(end[8:10])

# https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/SEP?startDate=yyyy-MM-dd&endDate=yyyy-MM-dd

url = 'https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CME?startDate=' + str(start) + '&endDate=' + str(end)

print(url)

url_sep = requests.get(url)
# text = url.text
data = url_sep.json()

#print(data)

filename = f"cme_data{start}_{end}.txt"

with open(filename, "w") as f:
    json.dump(data, f, indent=4)
    print("CME Data has been downloaded and exported to a .txt file.")

# # Store JSON from each URL in list.
# json_list = []
# json_list_error = []
 
# sep_json = requests.get(url)
# print(sep_json)
# print(sep_json._content())

# json_list.append(sep_json.json)
 
# print(json_list)
 

