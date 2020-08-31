import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 获取雅虎股票接口
import datetime
# 国内股票包
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

data = pd.read_csv("employee1.csv", header=0)
print(data)
code_list = data['ts_code'].values.tolist()
print(code_list)

# 保存为有序或者按日期的csv

# for code in code_list:
#     csv_name = "code_" + code + ".csv"
#     code_data = ts.pro_bar(ts_code=code, start_date='20190401', end_date='20200825')
#     date = code_data.shape[0]
#     for count in range(0, code_data.shape[0]):
#         print(code_data.iloc[count, 1])
#         code_data.iloc[count, 1] = date
#         date -= 1
#     print(code_data)
#     code_data.to_csv("stock_spilt_by_code//sort_date//"+csv_name)


# 保存为训练用初步数据集
X = []
Y = []

for code in code_list:
    csv_name = "code_" + code + ".csv"
    code_data = ts.pro_bar(ts_code=code, start_date='20190401', end_date='20200825')
    date = code_data.shape[0]
    datein = code_data.shape[0]
    for count in range(0, code_data.shape[0]):
        print(code_data.iloc[count, 1])
        code_data.iloc[count, 1] = date
        date -= 1
    if datein > 30:
        # i表示一个表里面可以生成多少 对 数据 比如 50天的数据就可以生成20对数据
        i = datein - 30
        code_data = code_data.drop(["ts_code"], axis=1)
        code_data1 = code_data
        code_data = code_data.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        for counti in range(i):
            foo = []
            for count30 in range(29, -1, -1):
                foo.append(np.array(code_data.iloc[counti + count30 + 1, 1:10]))
            # [counti, 2]表示最高值 所以没有负数 这次先设成收市价即[counti, 4]
            Y.append(np.array(round(((code_data1.iloc[counti, 4] - code_data1.iloc[counti, 1]) / code_data1.iloc[counti, 1]), 2)+0.1))
            X.append(foo)

print(len(X), len(Y))
np.save("X.npy", X)
np.save("Y.npy", Y)

# 建立三维数组保存为numpy
