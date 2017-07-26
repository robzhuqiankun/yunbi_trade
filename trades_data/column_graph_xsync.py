from bokeh.plotting import figure, output_file, show
from bokeh.layouts import column
import pandas as pd
import numpy as np
import random
import datetime
import bppl
from sklearn.linear_model import LinearRegression


def randcolor():
    de = ("%02x" % random.randint(50, 255))
    re = ("%02x" % random.randint(50, 255))
    we = ("%02x" % random.randint(50, 255))
    return '#' + de + re + we


def Scatter(data, x, y, types, title="", height=500, width=1000):
    type_set = set(data[types])
    colors = {t: randcolor() for t in type_set}
    p = figure(height=height, width=width, title=title)
    for t in type_set:
        p.circle(x=data[data[types] == t][x], y=data[data[types] == t][y], size=10, color=colors[t], legend=t)
    return p


def Line(data, x, y, title="", **kwargs):
    if type(y) == str:
        y = [y]
    p = figure(height=500, width=1000, title=title, **kwargs)
    for label in y:
        p.line(x=data[x], y=data[label], color=randcolor(), legend=label, line_width=2)

    return p


def Bar(data, x, y, title="", **kwargs):
    if type(y) == str:
        y = [y]
    p = figure(height=500, width=1000, title=title, **kwargs)
    top = data[y[0]][:]
    p.vbar(x=data[x], width=0.5, bottom=0, top=top, color=randcolor(), legend=y[0])

    for i in range(1, len(y)):
        p.vbar(x=data[x], width=0.5, bottom=top, top=top + data[y[i]], color=randcolor(), legend=y[i])
        top = top + data[y[i]]

    return p


class ColumnLayout(object):
    def __init__(self, x_axis_type='datetime', x_sync=False, fix_width=1000):
        self.figure_list = []
        self._x_axis_type = x_axis_type
        self._x_sync = x_sync
        self._fix_width = fix_width

    def add(self, fig):
        self.figure_list.append(fig)
        if self._x_sync and len(self.figure_list) > 1:
            self.figure_list[-1].x_range = self.figure_list[0].x_range

    def show(self):
        show(column(self.figure_list))


class EMA(object):
    def __init__(self, n):
        self.n = n
        self.ema = 0.0

    def generate_from_series(self, s):
        ls = s.tolist()
        res = []
        for x in ls:
            self.put(x)
            res.append(self.get())
        return res

    def generate_from_ratio(self, s, ratio):
        ls = s.tolist()
        ratio = ratio.tolist()
        res = []
        for i in range(len(ls)):
            if np.isnan(ratio[i]):
                self.put(ls[i])
            else:
                self.put(ls[i], self.n * ratio[i])
            res.append(self.get())
        return res

    def put(self, price, n=0):
        if n == 0:
            n = self.n
        if self.ema == 0.0:
            self.ema = price
        else:
            self.ema = (price * 2 + self.ema * (n - 1)) / (n + 1)

    def get(self):
        return self.ema


def sum_square_diff(x):
    return np.sum(np.square(np.diff(x)))


def generate_epi2(x):
    return sum_square_diff(x) / (2 * len(x))


def estimate_k(n, sigma2, epi2):
    k = int(np.power(12.0 * epi2 * epi2 * n * n / (sigma2 ** 2), 1.0 / 3))
    print k, epi2, sigma2
    if k < 2:
        return 2
    if k >= n:
        return n - 1
    return k


def estimate_sigma2(x, k):
    n = len(x)
    avg_k = 0.0
    for i in range(k):
        avg_k += sum_square_diff(x[i::k])
    avg_k /= k
    sigma = avg_k - (n - k + 1.0) / (k * n) * sum_square_diff(x)
    return sigma


def generate_sigma(x, detail=False):
    epi2 = generate_epi2(x)
    k = 12  # int(np.power(len(x), 2.0/3))
    sigma2 = estimate_sigma2(x, k)
    for i in range(0):
        k_next = estimate_k(len(x), sigma2=sigma2, epi2=epi2)
        if k_next == k:
            break
        k = k_next
        sigma2 = estimate_sigma2(x, k)
    return np.sqrt(sigma2)


def get_data(fpath, window=150):
    df = pd.read_csv(fpath)
    df['time_stamp'] = df['time_stamp'].apply(lambda x: datetime.datetime.fromtimestamp(float(x)))
    df['log'] = np.log(df['close'])
    df['vol'] = np.square(df['log'].diff())
    df['sigma'] = df['log'].rolling(window=window).apply(generate_sigma)
    df['sigma'] = df['sigma'].fillna(method='ffill')
    df['sigma'][df['sigma'] < 0.005] = 0.005
    df['sigma_avg'] = np.sqrt(np.square(df['sigma']).rolling(window=window * 4).mean())
    df['sigma_3d'] = np.sqrt(np.square(df['sigma']).rolling(window=60 * 3).mean())
    df['sigma_ratio'] = df['sigma'] / df['sigma_3d']
    df['epi'] = df['vol'].rolling(window=window, center=False).sum()
    df['epi'] = df['epi'].apply(lambda x: np.sqrt(x / (2 * window)))
    df['logma'] = df['log'].rolling(window=750, center=False).mean()
    df['ssma'] = EMA(15).generate_from_series(df['log'])
    df['sma'] = EMA(300).generate_from_series(df['log'])
    df['vma'] = EMA(150).generate_from_series(df['volume'])
    df['volume_ratio'] = df['vma'] / df['volume']
    df['volume_ratio'][df['volume_ratio'] > 5] = 5
    df['wvma'] = EMA(300).generate_from_ratio(df['log'], df['volume_ratio'])
    df['dma'] = EMA(300).generate_from_ratio(df['log'], df['sigma_ratio'])
    df['lma'] = EMA(900).generate_from_series(df['log'])
    df['band'] = 1.2 * df['sigma_avg'] + 0.01
    return df


def generate_trade_break(df):
    position = 0
    trades = []
    minmax = []
    assert isinstance(df, pd.DataFrame)
    for row in df.itertuples():
        if not np.isnan(row.sigma):
            if row.ssma > row.logma + row.band:
                while position <= 0:
                    trades.append((row.log + 0.002, 1, row.time_stamp))
                    minmax.append([row.log, row.log])
                    position += 1
                continue
            if row.ssma < row.logma - row.band:
                while position >= 0:
                    trades.append((row.log - 0.002, -1, row.time_stamp))
                    minmax.append([row.log, row.log])
                    position -= 1
                continue
            if position != 0:
                minmax[-1][0] = min(minmax[-1][0], row.log)
                minmax[-1][1] = max(minmax[-1][1], row.log)
                if position > 0:
                    if row.ssma < trades[-1][0] - 0.006 - row.sigma_avg * 0.9:
                        trades.append((row.log - 0.002, -1, row.time_stamp))
                        minmax.append([row.log, row.log])
                        position -= 1
                else:
                    if row.ssma > trades[-1][0] + 0.006 + row.sigma_avg * 0.9:
                        trades.append((row.log + 0.002, 1, row.time_stamp))
                        minmax.append([row.log, row.log])
                        position += 1
    if position != 0:
        trades.pop()
    return pd.DataFrame(trades, columns=['price', 'side', 'time'])


def generate_trade_trend(df):
    position = 0
    trades = []
    minmax = []
    last_dma = df['dma'][0]
    assert isinstance(df, pd.DataFrame)
    for row in df.itertuples():
        if not np.isnan(row.sigma):
            if row.dma > last_dma:
                while position <= 0:
                    trades.append((row.log + 0.002, 1, row.time_stamp))
                    minmax.append([row.log, row.log])
                    position += 1
            if row.dma < last_dma:
                while position >= 0:
                    trades.append((row.log - 0.002, -1, row.time_stamp))
                    minmax.append([row.log, row.log])
                    position -= 1
        last_dma = row.dma
    if position != 0:
        trades.pop()
    return pd.DataFrame(trades, columns=['price', 'side', 'time'])


def lr_mse(x, y, sigma=None):
    x = np.matrix(x).transpose()
    y = np.array(y).transpose()
    lr = LinearRegression()
    if sigma is None:
        lr.fit(x, y)
    else:
        weight = 1 / np.array(sigma).transpose()
        lr.fit(x, y, weight)
    return lr


def linear_reg(price, sigma, old_y_t=None, alpha=800):
    x = range(len(price))
    x = x[::10]
    price = price[::10]
    sigma = np.power(sigma[::10], 0)  # 0 for no weight
    avg_weight = 0.0
    for sig in sigma:
        avg_weight += 1 / (sig ** 2)
    avg_weight /= len(sigma)
    min_mse = None
    best_i = None
    left_lr = None
    right_lr = None
    for i in range(int(len(price) * 0.4), int(len(price) * 0.8), 3):
        llr = lr_mse(x[0:i], price[0:i], sigma[0:i])
        rlr = lr_mse(x[i:], price[i:], sigma[i:])
        new_y_t = rlr.predict(x[-1])
        if old_y_t is None:
            old_y_t = new_y_t
        sum_mse = llr._residues + rlr._residues + avg_weight * alpha * np.square(new_y_t - old_y_t) + abs(
            llr._residues / i - rlr._residues / (len(price) - i)) * len(price) * 0.5 * 0.01
        if min_mse is None or (min_mse and min_mse > sum_mse):
            min_mse = sum_mse
            best_i = i
            left_lr = llr
            right_lr = rlr
    """
    if best_i == 1000:
        print x[best_i:]
        print price[best_i:]
        print sigma[best_i:]
    """
    long_trend = lr_mse(x, price, sigma)

    # print len(x[:best_i]), len(x[best_i:])
    # print best_i, left_lr.predict(0), left_lr.predict(best_i), right_lr.predict(best_i), right_lr.predict(len(price))
    # return right_lr
    avg_residues = right_lr._residues / len(x[best_i:])
    return right_lr.predict(x[best_i]), right_lr.predict(x[-1]), best_i * 10, long_trend.predict(
        x[-1]), np.sqrt(avg_residues)
    # print best_i, min_mse
    # print left_lr.predict(0), left_lr.predict(best_i), right_lr.predict(best_i), right_lr.predict(len(price))


# eth_df = get_data('yunbi_ethcny_minute.csv')[42000:72000]
eth_df = get_data('yunbi_btccny_minute.csv')[5000:].copy(deep=True)
eth_df.index = range(len(eth_df))


def test_residue():
    test_y = eth_df['log'].tolist()
    sigma = np.power(eth_df['sigma'], 0).tolist()
    test_x = range(len(test_y))
    residues = []

    test_lr = lr_mse(test_x, test_y, sigma)
    line_y = []
    for x in test_x:
        residues.append((test_lr.predict(x) - test_y[x]) / sigma[x])
        line_y.append(test_lr.predict(x))
    df = pd.DataFrame({'x': test_x, 'y': line_y, 'residues': residues, 'time_stamp': eth_df['time_stamp'].tolist()})
    df['residues'] = np.square(df['residues'])
    return df


resd_df = test_residue()

time_x = []
break_time_x = []
head_y = []
pred_y = []
delta = []

from enum import Enum


class Signal(object):
    break_buy = 9
    trend_buy = 2
    none = 0
    trend_sell = -2
    break_sell = -9


def gen_signal(ema, short_trend, long_trend, band, delta, offset, sigma_avg):
    max_pos = 0.015 / sigma_avg
    max_pos = 1 if max_pos > 1 else max_pos
    std_delta = delta / band
    bar = 0.5
    pos_band = 1 * band
    neg_band = 1 * band
    pos_bar = 0.5
    neg_bar = 0.5

    # if ema > min(long_trend + short_trend, short_trend) + band
    if std_delta > -neg_bar:
        if ema > short_trend + pos_band:
            return Signal.break_buy, max_pos
    else:
        if ema > short_trend + 0.85 * pos_band:
            return Signal.break_buy, 0

    # if ema < max(long_trend - short_trend, short_trend) - band
    if std_delta < pos_bar:
        if ema < short_trend - neg_band:
            return Signal.break_sell, -max_pos
    else:
        if ema < short_trend - 1.02 * neg_band:
            return Signal.break_sell, 0
    """
    if std_delta > pos_bar and offset > 0.3:
        return Signal.break_buy, 1
    if std_delta < -neg_bar and offset < -0.3:
        return Signal.break_sell, -1

    if offset > 0.3:
        return Signal.break_buy, 0
    if offset < -0.3:
        return Signal.break_sell, 0
    """
    if std_delta > pos_bar and offset > 0.3:
        return Signal.break_buy, max_pos
    if std_delta < -neg_bar and offset < -0.3:
        return Signal.break_sell, -max_pos
    if std_delta > pos_bar and offset > 0:
        return Signal.trend_buy, max_pos
    if std_delta < -neg_bar and offset < 0:
        return Signal.trend_sell, -max_pos
    """
    if 0.4 * std_delta + 0.6 * offset > 0.5 * pos_bar:
        return Signal.trend_buy, 0
    if 0.4 * std_delta + 0.6 * offset < -0.5 * neg_bar:
        return Signal.trend_sell, 0
    """

    return Signal.none, 0


signal_df = {"time_x": time_x, "predict_y": pred_y, "upper": [], "lower": [], "trend": [], "ema": [], "price": [],
             "long_trend_y": [], "long_trend_upper": [], "long_trend_lower": [], "band": [], 'signal': [],
             "position": [], "residue_band": [], "trend_offset": [], "prev_change": [], "turnover": [], "date": [],
             "pnl": [], "profit_money": []}

cur_position = 0
avg_price = 0
profit = 0.0
profit_money = 0
residues_ema = EMA(10)
sigma_ema = EMA(200)
window_len = 60 * 36
for i in range(window_len, len(eth_df), 10):
    row = eth_df.loc[i]
    time_x.append(row.time_stamp)
    signal_df["date"].append(row.time_stamp.date())
    prev_y_t = None if len(pred_y) == 0 else pred_y[-1]
    (y_0, y_t, best_i, long_trend_y, residue_sigma) = linear_reg(eth_df['log'][i - window_len:i].tolist(),
                                                                 eth_df['sigma'][i - window_len:i].tolist(), prev_y_t)
    break_time_x.append(eth_df['time_stamp'][i - window_len + best_i - 1:i - window_len + best_i].tolist()[0])
    head_y.append(y_0)
    pred_y.append(float(y_t))
    delta.append(y_t - y_0)
    band = 1.2 * float(row.sigma_avg) + 0.01
    sigma_ema.put(row.sigma_avg)
    signal_df["long_trend_y"].append(float(long_trend_y))
    signal_df["upper"].append(float(y_t + band))
    signal_df["lower"].append(float(y_t - band))
    signal_df["band"].append(float(band))
    signal_df["long_trend_upper"].append(float(long_trend_y + band))
    signal_df["long_trend_lower"].append(float(long_trend_y - band))
    signal_df["ema"].append(row.ssma)
    signal_df["price"].append(row.close)
    signal_df["trend"].append(float((y_t - y_0) / band))
    signal_df["residue_band"].append(1.2 * float(residue_sigma) + 0.03)

    signal_df["prev_change"].append(0)

    residues_ema.put(signal_df["ema"][-1] - pred_y[-1])

    signal_df["trend_offset"].append(residues_ema.get() / residue_sigma)

    (buy_sell_signal, target) = gen_signal(signal_df["ema"][-1], pred_y[-1], long_trend_y, signal_df["band"][-1],
                                           float(y_t - y_0), signal_df["trend_offset"][-1],
                                           0.5 * sigma_ema.get() + 0.5 * row.sigma_avg)
    signal_df["signal"].append(target)

    delta_position = 0

    if cur_position < target and int(buy_sell_signal) > 0:
        delta_position = min(0.05 * int(buy_sell_signal), target - cur_position)
    elif cur_position > target and int(buy_sell_signal) < 0:
        delta_position = max(0.05 * int(buy_sell_signal), target - cur_position)

    if abs(delta_position) > abs(cur_position) and delta_position * cur_position < 0:
        delta_position = -cur_position

    if abs(delta_position) > 0:
        if delta_position * cur_position >= 0:
            avg_price = (avg_price * cur_position + signal_df["price"][-1] * delta_position) / (
                delta_position + cur_position)
        else:
            profit += (avg_price - signal_df["price"][-1]) / avg_price * delta_position
            profit_money += (avg_price - signal_df["price"][-1]) * delta_position
    cur_position += delta_position
    float_profit = (signal_df["price"][-1] - avg_price) / avg_price * cur_position if avg_price > 0 else 0
    signal_df["position"].append(cur_position)
    signal_df["pnl"].append(profit + float_profit)
    signal_df["profit_money"].append(profit_money + float_profit * avg_price)
    signal_df["turnover"].append(abs(delta_position))

signal_df = pd.DataFrame(signal_df)

# linear_reg(eth_df['log'].tolist(), eth_df['sigma'].tolist())
eth_df['summit'] = (eth_df['ssma'] - eth_df['sma']) / np.power(eth_df['sigma_avg'], 0)
eth_df['summit_d'] = eth_df['summit'].diff()


def old_show():
    trade_df = generate_trade_break(eth_df)
    layout = ColumnLayout(x_sync=True, fix_width=1300)
    layout.add(Line(eth_df, 'time_stamp', ['close'], x_axis_type='datetime'))
    layout.add(Line(eth_df, 'time_stamp', ['log', 'dma', 'sma', 'wvma', 'ssma'], x_axis_type='datetime'))
    for i in range(0, len(head_y), 3):
        layout.figure_list[1].line([break_time_x[i], time_x[i]], [head_y[i], pred_y[i]])
    layout.figure_list[1].line(time_x, pred_y, color='red')
    # layout.figure_list[1].line(resd_df['time_stamp'], resd_df['y'], color='green')
    layout.figure_list[1].line(eth_df['time_stamp'], eth_df['logma'] + eth_df['band'])
    layout.figure_list[1].line(eth_df['time_stamp'], eth_df['logma'] - eth_df['band'])
    layout.figure_list[1].circle_x(trade_df[trade_df['side'] > 0]['time'], trade_df[trade_df['side'] > 0]['price'],
                                   color="red", fill_color=None, size=20, line_width=5)
    layout.figure_list[1].square_cross(trade_df[trade_df['side'] < 0]['time'], trade_df[trade_df['side'] < 0]['price'],
                                       color="blue", fill_color=None, size=20, line_width=5)
    lr_df = pd.DataFrame({'time_stamp': time_x, 'delta': delta})

    # layout.add(Bar(lr_df, 'time_stamp', ['delta'], x_axis_type='datetime'))
    layout.add(Bar(resd_df, 'time_stamp', ['residues'], x_axis_type='datetime'))

    layout.add(Bar(eth_df, 'time_stamp', ['sigma'], x_axis_type='datetime'))
    # layout.add(Bar(eth_df, 'time_stamp', ['sigma_3d'], x_axis_type='datetime'))
    # layou t.add(Bar(eth_df, 'time_stamp', ['sigma'], x_axis_type='datetime'))
    layout.show()


def new_show():
    layout = ColumnLayout(x_sync=True, fix_width=1300)

    layout.add(Line(eth_df, 'time_stamp', ['close'], x_axis_type='datetime'))
    layout.add(Line(signal_df, 'time_x', ['predict_y', 'long_trend_y', 'ema', ], x_axis_type='datetime'))
    layout.figure_list[-1].line(signal_df['time_x'], signal_df['upper'], color='black')
    layout.figure_list[-1].line(signal_df['time_x'], signal_df['lower'], color='black')
    # layout.figure_list[-1].line(signal_df['time_x'], signal_df['long_trend_upper'], color='red')
    # layout.figure_list[-1].line(signal_df['time_x'], signal_df['long_trend_lower'], color='red')
    layout.figure_list[-1].line(signal_df['time_x'], signal_df['predict_y'] + signal_df['residue_band'], color='red',
                                legend="upper_resi")
    layout.figure_list[-1].line(signal_df['time_x'], signal_df['predict_y'] + signal_df['residue_band'], color='red',
                                legend="lower_resi")
    layout.add(Bar(signal_df, 'time_x', ['position'], x_axis_type='datetime'))
    layout.add(Line(signal_df, 'time_x', ['pnl'], x_axis_type='datetime'))
    layout.add(Line(signal_df, 'time_x', ['profit_money'], x_axis_type='datetime'))
    layout.add(Bar(signal_df, 'time_x', ['signal'], x_axis_type='datetime'))
    layout.add(Bar(signal_df, 'time_x', ['trend'], x_axis_type='datetime'))
    layout.add(Bar(signal_df, 'time_x', ['trend_offset'], x_axis_type='datetime'))
    layout.add(Bar(signal_df, 'time_x', ['residue_band'], x_axis_type='datetime'))
    layout.add(Bar(signal_df, 'time_x', ['band'], x_axis_type='datetime'))
    layout.show()


new_show()

print signal_df['date']


def performance_output():
    res_df = pd.DataFrame()
    res_df['return'] = signal_df['pnl'].groupby(signal_df['date']).mean()
    res_df['turnover'] = signal_df['turnover'].groupby(signal_df['date']).sum()
    res_df['return'] = res_df['return'].diff()
    print res_df
    res_df.to_csv('btc_trend.csv')


performance_output()
