import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
#获取雅虎股票接口
import datetime
#国内股票包
import tushare as ts

ts.set_token('bc0bfe57b6acaa3e8941931566e97186de1143d43061a8f1bdb7d186')

# 把全部数据都读取下来

# pro = ts.pro_api()
#
# data = ts.pro_bar(ts_code='002352.SZ', adj='qfq', start_date='20150101', end_date='20200824')
#
# data.to_csv("employee.csv")
#
# print(data)
#
# data1 = pro.stock_basic(industry="银行",exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,list_date')
#
# data1[data1['industry'] == '证券'].to_csv("employee1.csv",encoding='utf_8_sig')


# 把需要的数据整理成csv

data = pd.read_csv("employee1.csv",header=0)
print(data)
code_list = data['ts_code'].values.tolist()
print(code_list)

for code in code_list:
    csv_name = "code_" + code + ".csv"
    code_data = ts.pro_bar(ts_code=code, start_date='20190401', end_date='20200825')
    date = code_data.shape[0]
    for count in range(0, code_data.shape[0]):
        print(code_data.iloc[count, 1])
        code_data.iloc[count, 1] = date
        date -= 1
    print(code_data)
    code_data.to_csv("stock_spilt_by_code//sort_date//"+csv_name)

# 建立三维数组保存为numpy


