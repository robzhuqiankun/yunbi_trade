import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from bokeh.plotting import figure, show

BEGIN_TIMESTAMP = 1495663410
END_TIMESTAMP = 1497267864

file_list = ['yunbi_ethcny_minute.csv', 'yunbi_etccny_minute.csv', 'yunbi_zeccny_minute.csv']

class EMA(object):
    def __init__(self, n):
        self.n = n
        self.ema = 0.0

    def put(self, price):
        if self.ema == 0.0:
            self.ema = price
        else:
            self.ema = (price * 2 + self.ema * (self.n - 1)) / (self.n + 1)

    def get(self):
        return self.ema


def read_minute_line(fpath):
    df = pd.read_csv(fpath).set_index('time_stamp')
    vwap = df['vwap'].copy(deep=True)
    assert isinstance(vwap, pd.Series)
    x = vwap[BEGIN_TIMESTAMP < vwap.index].copy(deep=True)
    return x[END_TIMESTAMP > x.index].copy(deep=True)

pca_x = []
last_n = 0
for fp in file_list:
    s = read_minute_line(fp)
    pca_x.append(s.tolist())
    if last_n > 0:
        while len(pca_x[-1]) > last_n:
            pca_x[-1].pop()
    last_n = len(pca_x[-1])

pca = PCA(n_components=1)
x = np.array(pca_x).transpose()
pca.fit(x)
y = pca.transform(x)
y += 2000
print pca.explained_variance_ratio_

trend = []
ema = EMA(60)
for p in y.transpose()[0]:
    ema.put(p)
    trend.append(ema.get())

plot = figure(width=1300, y_axis_type='log')
plot.scatter(x=range(last_n), y=x.transpose()[0], color='red', legend='eth')
plot.scatter(x=range(last_n), y=x.transpose()[1]*10, color='yellow', legend='etc')
plot.scatter(x=range(last_n), y=x.transpose()[2], color='blue', legend='zec')
plot.scatter(x=range(last_n), y=y.transpose()[0], color='black', legend='pca')
plot.line(x=range(last_n), y=trend, color='green', legend='ema')
show(plot)
