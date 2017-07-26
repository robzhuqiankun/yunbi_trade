from ..trade.trade_lib import TradeClient, Order, Snapshot, MinuteLine
import datetime
from collections import deque
import logging
import pandas
import numpy
import json
from ..trade.utility import init_logging
from math import sqrt, pow


class Volatility(object):
    MAXL = 10000

    def __init__(self, n):
        self.q = deque()
        self.n = n
        self.vols = deque()
        self.mean = 0.0
        self.std = 0.0

    def is_local_min(self):
        if len(self.vols) <= 30:
            return False
        min_i = -30
        min_vol = self.vols[-30]
        for i in range(-30, -1):
            if self.vols[i] < min_vol:
                min_vol = self.vols[i]
                min_i = i
        if -25 <= min_i <= -5:
            return True
        return False

    def vol_mean(self, n):
        if len(self.vols) <= n:
            return 0
        else:
            arr = []
            for i in range(-n, -1):
                arr.append(self.vols[i])
            return numpy.mean(arr)

    def get_mean(self, n=-1):
        if n == -1:
            return self.mean

    def get_std(self, n=-1):
        if n == -1:
            return self.std

    def get_normal_std(self):
        return self.std / self.mean

    def put(self, minute_line):
        assert isinstance(minute_line, MinuteLine)
        self.q.append(minute_line)
        while len(self.q) > self.n:
            self.q.popleft()
        # if len(self.q) >= self.n:
        if len(self.q) >= 1:
            self.sumavg = 0.0
            self.sumsq = 0.0
            self.sumvol = 0.000001
            """
            # weighted std
            for minute_ in self.q:
                self.sumavg += minute_.vwap * minute_.volume
                self.sumvol += minute_.volume
            self.mean = self.sumavg / self.sumvol
            for minute_ in self.q:
                self.sumsq += ((minute_.vwap - self.mean) ** 2) * minute_.volume
            """
            # weighted std
            for minute_ in self.q:
                self.sumavg += minute_.vwap
                self.sumvol += 1
            self.mean = self.sumavg / self.sumvol
            for minute_ in self.q:
                self.sumsq += ((minute_.vwap - self.mean) ** 2)

            self.std = sqrt(self.sumsq / (self.sumvol - 1))
            self.vols.append(self.std)

        while len(self.vols) > self.MAXL:
            self.vols.popleft()


class MA(object):
    def __init__(self, n):
        self.n = n
        self.sum = 0.0
        self.ema = 0.0
        self.q = deque()
        self.tot_vol = 0.0
        self.last_value = 0.0

    def put(self, minute_line):
        assert isinstance(minute_line, MinuteLine)
        self.q.append(minute_line)
        self.sum += minute_line.vwap * pow(minute_line.volume, 0.25)
        self.tot_vol += pow(minute_line.volume, 0.25)
        while len(self.q) > self.n:
            rm = self.q.popleft()
            self.sum -= rm.vwap * pow(rm.volume, 0.25)
            self.tot_vol -= pow(rm.volume, 0.25)

    def get_std(self, n):
        if len(self.q) <= n:
            return 0
        mas = []
        for i in range(-n, 0):
            mas.append(self.q[i].vwap)
        return numpy.std(mas)

    def get_mean(self, n):
        if len(self.q) <= n:
            return self.q[-1].vwap
        mas = []
        for i in range(-n, 0):
            mas.append(self.q[i].vwap)
        return numpy.mean(mas)

    def get(self):
        if self.tot_vol > 0.000001:
            self.last_value = self.sum / self.tot_vol
        return self.last_value


class EMA(object):
    def __init__(self, n):
        self.n = n
        self.ema = 0.0

    def put(self, minute_line):
        price = minute_line
        if isinstance(minute_line, MinuteLine):
            price = minute_line.vwap
        if self.ema == 0.0:
            self.ema = price
        else:
            self.ema = (price * 2 + self.ema * (self.n - 1)) / (self.n + 1)

    def get(self):
        return self.ema


class Strategy(TradeClient):
    def __init__(self, symbols, account, config_path, live=False, **params):
        print "init"
        init_logging('macd_%s_trade_log' % symbols)
        super(Strategy, self).__init__(symbols=symbols,
                                       account=account,
                                       live=live,
                                       **params)
        self.last_signal = 0
        self.signal_flip_time = 0
        self.order_qty = 0
        self.target_position = 0
        self.short_ma_len = 0
        self.long_ma_len = 0
        self.trade_symbol = None
        self.config_path = config_path
        self.read_config()
        self.vols = Volatility(90)
        self.short_ma = EMA(self.short_ma_len)
        self.long_ma = EMA(self.long_ma_len)
        self.tick_data = {"ma1": [], "ma5": [], "ma15": [], "bid": [], "ask": [], "time": [], "upper": [], "lower": [],
                          "mean": [], "std": [], "avg_std": []}
        # self.enter_price = 0.0
        self.last_timestamp = 0
        self.last_macd = 0
        self.last_macd_time = 0

        self.tick_counter = 0
        self.trade_trigger = False
        self.register_minute_callback(self.on_minute_callback)
        self.bolling = self.BollingSignal(self.tick_data, 30)
        self.bolling.vols = self.vols
        self.init_account()

    class BollingSignal(object):
        MAXL = 10000

        def __init__(self, tick_data, state_duration=15):
            self.track_state = 0
            self.minute_q = deque(maxlen=10000)
            self.shift_q = deque(maxlen=1000)
            self.state_q = deque(maxlen=10000)
            self.state_q.append(0)
            self.minute_states = deque(maxlen=10000)
            self.vols = None
            self.state_duration = state_duration
            self.tick_data = tick_data
            self.signal = 0
            self.short_eva = EMA(60)
            self.long_eva = EMA(1500)
            self.const_std = 0.04
            self.vol_range = 0.0
            self.jump_signal = 0

        def put(self, minute_line):
            self.minute_q.append(minute_line)
            sum_wvap = 0.0
            sum_vol = 0.0
            for i in range(1, min(5, len(self.minute_q)) + 1):
                sum_wvap += self.minute_q[-i].vwap * self.minute_q[-i].volume
                sum_vol += self.minute_q[-i].volume
            if sum_vol > 0:
                self.shift_q.append((sum_wvap / sum_vol - self.vols.get_mean()) / self.vols.get_mean())
            else:
                self.shift_q.append((self.minute_q[-1].vwap - self.vols.get_mean()) / self.vols.get_mean())
            self.short_eva.put(self.vols.get_normal_std() ** 2)
            self.long_eva.put(self.vols.get_normal_std() ** 2)
            self.vol_range = 25 * (self.short_eva.get() * 0.25 + self.long_eva.get() * 0.5 + (
                self.const_std ** 2) * 0.25)

            self.jump_signal = 0
            for i in range(2, min(100, len(self.shift_q)) + 1):
                if self.shift_q[-1] ** 2 - self.shift_q[-i] ** 2 < -self.vol_range:
                    self.jump_signal = -1
                    break
                if self.shift_q[-1] ** 2 - self.shift_q[-i] ** 2 > self.vol_range:
                    self.jump_signal = 1
                    break

            self.minute_states.append(self.get_state(minute_line.vwap))
            jump_duration = 50
            if len(self.minute_states) >= jump_duration:
                state = self.minute_states[-jump_duration]
                for i in range(-jump_duration, 0):
                    if state != self.minute_states[i]:
                        state = 0
                        break
                if state == 2 or state == -2:
                    if state != self.state_q[-1]:
                        # if state * self.state_q[-1] <= 0:
                        #     self.signal = state / 2
                        self.state_q.append(state)
                        print state, self.signal, minute_line
                        return

            if len(self.minute_q) >= self.state_duration:
                counter = {-2: 0, -1: 0, 1: 0, 2: 0}
                for i in range(-self.state_duration, 0):
                    counter[self.minute_states[i]] += 1

                for key in counter:
                    if counter[key] >= self.state_duration * 0.95:
                        if key > self.state_q[-1]:
                            if key * self.state_q[-1] <= 0:
                                self.signal = 1
                            self.state_q.append(key)
                            print key, self.signal, minute_line
                        elif key < self.state_q[-1]:
                            if key * self.state_q[-1] <= 0:
                                self.signal = -1
                            print key, self.signal, minute_line
                            self.state_q.append(key)

                """
                state = self.get_state(self.minute_q[-self.state_minutes].vwap)
                for i in range(-self.state_minutes, 0):
                    if self.state_q[-1] == self.get_state(self.minute_q[i].vwap):
                        state = 0
                        break


                if state != 0:
                    if state > self.state_q[-1]:
                        self.signal = 1
                    if state < self.state_q[-1]:
                        self.signal = -1
                    self.state_q.append(state)
                """

        def minute_trend(self, mins_list, length):
            if len(mins_list) < length:
                return 0
            cntp = 0
            cntn = 0
            for i in range(-length, -1):
                if mins_list[i + 1] - mins_list[i] > 0:
                    cntp += 1
                else:
                    cntn += 1
            if cntp > 0.8 * length and mins_list[-1] - mins_list[-length] > 0 and mins_list[-1] - mins_list[-2] > 0:
                return 1
            if cntn > 0.8 * length and mins_list[-1] - mins_list[-length] < 0 and mins_list[-1] - mins_list[-2] < 0:
                return -1
            return 0

        def trend(self):
            ma5trd = self.minute_trend(self.tick_data["ma5"], 30)
            ma15trd = self.minute_trend(self.tick_data["ma15"], 30)
            if self.tick_data["ma5"][-1] > self.tick_data["ma15"][-1] and ma5trd > 0 and ma15trd > 0:
                return 1
            if self.tick_data["ma5"][-1] < self.tick_data["ma15"][-1] and (ma5trd < 0 and ma15trd < 0):
                return -1
            return 0

        def break_trend(self):
            if len(self.state_q) < 3:
                return 0
            ma5trd = self.minute_trend(self.tick_data["ma5"], 10)
            if self.state_q[-1] > 0 and self.state_q[-2] == -1 and self.state_q[-3] == -2 and ma5trd > 0:
                return 1
            if self.state_q[-1] < 0 and self.state_q[-2] == 1 and self.state_q[-3] == 2:
                return -1
            return 0

        def escape_top(self):
            vwap_list = []
            if len(self.minute_q) > 300:
                for i in range(-300, 0):
                    vwap_list.append(self.minute_q[i].vwap)
                vwap_list.sort(reverse=True)
                max_vwap = numpy.mean(vwap_list[0:5])
                if self.minute_q[-1].vwap < max_vwap * 0.95 and self.vols.is_local_min():
                    return True
            return False

        def is_low_vol(self):
            return self.vols.get_std() < 0.9 * self.vols.vol_mean(240)

        def low_vol_escape(self):
            if self.is_low_vol() and self.vols.is_local_min() and self.minute_trend(self.tick_data["ma5"],
                                                                                    60) < 0 and self.signal < 0:
                return True

        def get_state(self, price):
            upper = self.tick_data["mean"][-1] * (1 + sqrt(self.vol_range))
            mean = self.tick_data["mean"][-1]
            lower = self.tick_data["mean"][-1] * (1 - sqrt(self.vol_range))

            if price < lower:
                return -2
            elif price < mean:
                return -1
            elif price < upper:
                return 1
            else:
                return 2

        def get_signal(self):
            trd = self.trend()
            # trend or large jump breaker
            if self.break_trend() > 5 or (self.signal > 0 and trd > 0):
                return 1
            if self.break_trend() < -5 or (self.signal < 0 and trd < 0 and self.is_low_vol()) or (
                        self.low_vol_escape() and False):
                return -1
            return 0

    def read_config(self):
        with open(self.config_path) as f:
            config = json.load(f)
            self.order_qty = float(config['order_qty'])
            self.target_position = float(config['target_position'])
            self.trade_symbol = str(config['symbol'])
            self.short_ma_len = int(config['short_ma_len'])
            self.long_ma_len = int(config['long_ma_len'])

    def init_account(self):
        while True:
            try:
                (self.balance, self.position) = self.query_account(self.trade_symbol[0:-3])
                logging.info("balance: %s, position: %s" % (self.balance, self.position))
                break
            except Exception, e:
                logging.error(str(e))

    def on_minute_callback(self, minute_line):
        self.read_config()
        assert isinstance(minute_line, MinuteLine)
        self.vols.put(minute_line)
        self.short_ma.put(minute_line)
        self.long_ma.put(minute_line)

        self.tick_data["ma5"].append(self.short_ma.get())
        self.tick_data["ma15"].append(self.long_ma.get())
        self.tick_data["mean"].append(self.vols.get_mean())
        self.tick_data["time"].append(datetime.datetime.fromtimestamp(minute_line.time_stamp + 8 * 3600))
        self.tick_data["std"].append(self.vols.get_std())
        self.bolling.put(minute_line)
        if self.is_live is True:
            logging.info("EMA(%s,%s) M|%s" % (self.short_ma.get(), self.long_ma.get(), minute_line))
        else:
            self.tick_data["ma1"].append(minute_line.vwap)
            self.tick_data["bid"].append(minute_line.vwap)
            self.tick_data["ask"].append(minute_line.vwap)
            self.tick_data["upper"].append(self.vols.get_mean() + sqrt(self.bolling.vol_range) * self.vols.get_mean())
            self.tick_data["lower"].append(self.vols.get_mean() - sqrt(self.bolling.vol_range) * self.vols.get_mean())
            self.tick_data["avg_std"].append(self.vols.vol_mean(240))

        cur_signal = self.bolling.get_signal()
        if cur_signal != self.last_signal:
            self.last_signal = cur_signal
            self.signal_flip_time = minute_line.time_stamp

        if len(self.tick_data["ma15"]) > 450:
            self.trade_trigger = True

        if self.trade_trigger is False:
            return
        if self.position < 0.9 * self.target_position:
            if self.last_timestamp:
                if minute_line.time_stamp - self.last_timestamp < 36:
                    return
            # if self.breaker_signal(minute_line) > 0:
            if cur_signal > 0 and minute_line.time_stamp - self.signal_flip_time < 7200:
                print "signal: %s" % self.bolling.signal
                logging.info("try to buy")
                try:
                    order = Order(symbol=self.trade_symbol, side=1, price=minute_line.close * 1.003,
                                  quantity=self.order_qty, at=minute_line.time_stamp)
                    (self.balance, self.position) = self.hit_place(order, self.target_position)
                    logging.info("Order: %s" % order)
                    logging.info(str("balance: %s, position: %s" % (self.balance, self.position)))
                except Exception, e:
                    logging.error(str(e))
                # self.enter_price = minute_line.close

        if self.position > 0.01 * self.target_position:
            # if self.breaker_signal(minute_line) < 0:
            if cur_signal < 0:
                logging.info("try to sell")
                if minute_line.close < self.tick_data["ma15"][-1] * 0.7:
                    logging.warning("large jump price: %s" % minute_line.close)
                    return
                try:
                    order = Order(symbol=self.trade_symbol, side=-1, price=minute_line.close * 0.997,
                                  quantity=self.order_qty, at=minute_line.time_stamp)
                    (self.balance, self.position) = self.hit_place(order, self.target_position)
                    if self.position < self.order_qty:
                        self.last_timestamp = minute_line.time_stamp
                    logging.info("Order: %s" % order)
                    logging.info(str("balance: %s, position: %s" % (self.balance, self.position)))
                except Exception, e:
                    logging.error(str(e))

    def macd_up(self):
        if self.short_ma.get() > self.long_ma.get():
            if len(self.tick_data["time"]) > 100:
                if (self.tick_data["ma5"][-1] - self.tick_data["ma5"][-30] > 0) and (
                                self.tick_data["ma15"][-1] - self.tick_data["ma15"][-30] > 0):
                    if self.short_ma.get() - self.long_ma.get() < 0.0005:
                        return False
                    sum_inc = 0.0
                    for i in range(-1, -len(self.tick_data["time"]), -1):
                        if (self.tick_data["time"][-1] - self.tick_data["time"][i - 1]).total_seconds() > 900:
                            break
                        sum_inc += (self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1]) / self.tick_data["ma5"][-1]
                    if sum_inc > 0.0005:
                        return True
        return False

    def macd_down(self):
        if len(self.tick_data["time"]) > 100:
            if (self.tick_data["ma5"][-1] - self.tick_data["ma5"][-30] < 0) and (
                        True or self.tick_data["ma15"][-1] - self.tick_data["ma15"][-30] < 0):
                if self.short_ma.get() < self.long_ma.get():
                    return True
                """
                sum_inc = 0.0
                for i in range(-1, -len(self.tick_data["time"]), -1):
                    if (self.tick_data["time"][-1] - self.tick_data["time"][i - 1]).total_seconds() > 900:
                        break
                    if self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1] < 0:
                        sum_inc += (self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1]) / self.tick_data["ma5"][-1]
                if sum_inc < -0.001:
                    return True
                """
        return False

    def stop_loss(self, price):
        if price / self.enter_price < 0.92:
            return True
        else:
            return False

    def breaker_signal(self, minute_line):
        if len(self.tick_data["time"]) < 300:
            return 0
        if min(self.tick_data["ma1"][-3:]) > self.vols.get_mean() + 1.8 * self.vols.get_std() and (
                    self.tick_data["ma15"][-1] - self.tick_data["ma15"][-120]) > 0:
            return 1
        if (True or (self.tick_data["ma1"][
                         -1] < self.vols.get_mean() and self.vols.is_local_min() and self.vols.get_std()
            < 0.9 * self.vols.vol_mean(240))) and self.tick_data["ma1"][
            -1] < self.vols.get_mean() - self.vols.get_std():
            return -1
        return 0

    def minute_trend(self):
        if len(self.tick_data["time"]) > 100:
            if (self.tick_data["ma15"][-1] - self.tick_data["ma15"][-2]) > 0 and (
                        self.tick_data["ma5"][-1] - self.tick_data["ma5"][-60]) > 0:
                return 1
        return 0

    def back_test_performance(self):
        if len(self.sim_account.trades) > 0 and self.sim_account.trades[-1].side == 1:
            self.sim_account.trades.append(
                Order(price=self.tick_data["ma1"][-1], side=-1, symbol=self.trade_symbol, quantity=self.position,
                      at=self.bolling.minute_q[-1].time_stamp))
            self.sim_account.nets.append(
                self.sim_account.balance + self.sim_account.position * self.tick_data["ma1"][-1])

        super(Strategy, self).back_test_performance()

        def random_color():
            import random
            x = random.random() * 256 * 256 * 256
            prefix = '#'
            for i in range(6 - len(str(hex(int(x))[2:]))):
                prefix = prefix + '0'
            return prefix + str(hex(int(x))[2:]).upper()

        from bokeh.plotting import figure, show, Column

        buy_orders = {"time": [], "price": []}
        sell_orders = {"time": [], "price": []}
        for trade in self.sim_account.trades:
            assert isinstance(trade, Order)
            t_time = trade.at + datetime.timedelta(hours=8)
            if trade.side > 0:
                buy_orders["time"].append(t_time)
                buy_orders["price"].append(trade.price)
            else:
                sell_orders["time"].append(t_time)
                sell_orders["price"].append(trade.price)

        plot = figure(x_axis_type="datetime", width=1300)
        bar_plot = figure(x_axis_type="datetime", width=1300)
        # plot.line(x=self.tick_data["time"], y=self.tick_data["ma1"], legend="ma10", color=random_color())
        plot.circle_x(x=buy_orders["time"], y=buy_orders["price"], color="red", fill_color=None, size=20, line_width=5)
        plot.square_cross(x=sell_orders["time"], y=sell_orders["price"], color="blue", fill_color=None, size=20,
                          line_width=5)
        plot.line(x=self.tick_data["time"], y=self.tick_data["ma5"], legend="ma30", color=random_color())
        plot.line(x=self.tick_data["time"], y=self.tick_data["ma15"], legend="ma90", color=random_color())
        plot.line(x=self.tick_data["time"], y=self.tick_data["mean"], legend="mean", color='black')
        plot.line(x=self.tick_data["time"], y=self.tick_data["upper"], legend="upper", color='black')
        plot.line(x=self.tick_data["time"], y=self.tick_data["lower"], legend="lower", color='black')
        plot.scatter(x=self.tick_data["time"], y=self.tick_data["bid"], legend="bid", color=random_color())
        plot.scatter(x=self.tick_data["time"], y=self.tick_data["ask"], legend="ask", color=random_color())
        bar_plot.vbar(x=self.tick_data["time"], top=self.tick_data["std"], bottom=0, width=1)
        bar_plot.line(x=self.tick_data["time"], y=self.tick_data["avg_std"])
        show(Column(plot, bar_plot))
