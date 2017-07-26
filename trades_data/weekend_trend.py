import datetime
import pandas as pd
import numpy as np
from math import log
from collections import OrderedDict

df = pd.read_csv('yunbi_sccny_minute.csv')

dic = OrderedDict()

for row in df.itertuples():
    date = datetime.datetime.fromtimestamp(int(row.time_stamp)).date()
    if date not in dic:
        dic[date] = {"price": [], "volume": []}
    dic[date]["price"].append(row.vwap)
    dic[date]["volume"].append(row.volume)

res = OrderedDict()

for date in dic:
    weekday = date.weekday()
    if weekday not in res:
        res[weekday] = {"std": [], "change": [], "volume": [], "sig": []}
    res[weekday]["std"].append(np.std(dic[date]["price"]) / np.mean(dic[date]["price"]))
    res[weekday]["change"].append(dic[date]["price"][-1] / dic[date]["price"][0] - 1.0)
    res[weekday]["sig"].append(1 if dic[date]["price"][-1] - dic[date]["price"][0] > 0 else -1)
    res[weekday]["volume"].append(log(np.sum(dic[date]["volume"])))

raw_df = {'weekday': [], 'std': [], 'change': [], 'sig': [], 'volume': []}
for weekday in res:
    print "weekday: " + str(weekday)
    print np.mean(res[weekday]['std']), np.mean(res[weekday]['change']), np.mean(res[weekday]['sig']), np.mean(res[weekday]['volume'])
    print res[weekday]['std'], res[weekday]['change'], res[weekday]['sig'], res[weekday]['volume']

    raw_df['weekday'].append(weekday + 1)
    for key in res[weekday]:
        raw_df[key].append(np.mean(res[weekday][key]))

df = pd.DataFrame(raw_df).sort_values('weekday').set_index('weekday')
print df


# print datetime.datetime.now().weekday()
