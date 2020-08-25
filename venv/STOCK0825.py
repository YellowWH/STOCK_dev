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

# data1 = pd.read_csv("employee1.csv",header=0)
# print(data1)
# s=data1['ts_code'].values.tolist()
# print(s)
# d=""
# for si in s:
#     d = d + si + ","
#
# print(d)
# data = ts.pro_bar(ts_code=d, start_date='20190824', end_date='20200824')
#
# data.to_csv("stock1.csv")

