from backtest_lib import BackTest, Order, Snapshot
import datetime
from collections import deque
from bokeh.plotting import figure, show


class MA(object):
    def __init__(self, n):
        self.n = n
        self.sum = 0.0
        self.ema = 0.0
        self.q = deque()
        self.cnt = 0

    def put(self, snap):
        assert isinstance(snap, Snapshot)
        self.q.append(snap)
        self.sum += snap.last_price
        self.cnt += 1
        while len(self.q) > 0 and snap.time_stamp - self.q[0].time_stamp > self.n:
            rm = self.q.popleft()
            self.sum -= rm.last_price
            self.cnt -= 1

        if self.ema == 0.0:
            self.ema = snap.last_price
        else:
            self.ema = (snap.last_price * 2 + self.ema * (self.n - 1)) / (self.n + 1)

    def get(self):
        return self.ema
        # return self.sum / self.cnt


class Strategy(BackTest):
    def __init__(self, symbol):
        print "init"
        super(Strategy, self).__init__(symbol, ["2017-05-29.log", "2017-05-30.log", "2017-05-31.log"])
        self.position = 0
        self.ma2 = MA(120)
        self.ma5 = MA(1800)
        self.ma15 = MA(5400)
        self.tick_data = {"ma2": [], "ma5": [], "ma15": [], "bid": [], "ask": [], "time": []}
        self.enter_price = 0.0
        self.last_time = 0
        self.last_macd = 0
        self.last_macd_time = 0

    def on_snap(self, snapshot):
        assert isinstance(snapshot, Snapshot)
        self.ma2.put(snapshot)
        self.ma5.put(snapshot)
        self.ma15.put(snapshot)
        self.tick_data["ma2"].append(self.ma2.get())
        self.tick_data["ma5"].append(self.ma5.get())
        self.tick_data["ma15"].append(self.ma15.get())
        self.tick_data["bid"].append(snapshot.bid)
        self.tick_data["ask"].append(snapshot.ask)
        self.tick_data["time"].append(datetime.datetime.fromtimestamp(snapshot.time_stamp + 8 * 3600))
        if self.ma2.get() > self.ma5.get() and self.last_macd <= 0:
            self.last_macd = 1
            self.last_macd_time = snapshot.time_stamp
        if self.ma2.get() < self.ma5.get() and self.last_macd >= 0:
            self.last_macd = -1
            self.last_macd_time = snapshot.time_stamp
        if self.position == 0:
            if self.macd_up(snapshot):
                self.place(Order("sccny", 1, snapshot.ask, 100, snapshot.time_stamp))
                self.enter_price = snapshot.ask
                self.position = 100
                self.last_time = snapshot.time_stamp
        else:
            if (self.macd_down(snapshot)) or self.stop_loss(snapshot.bid):
                self.place(Order("sccny", -1, snapshot.bid, 100, snapshot.time_stamp))
                self.position = 0
                self.last_time = snapshot.time_stamp

    def macd_up(self, snapshot):
        if self.last_time:
            if snapshot.time_stamp - self.last_time < 3600:
                return False
        if self.ma5.get() > self.ma15.get():
            if len(self.tick_data["time"]) > 1000:
                if (self.tick_data["ma5"][-1] - self.tick_data["ma5"][-3] > 0) and (
                        self.tick_data["ma15"][-1] - self.tick_data["ma15"][-3] > 0):
                    sum_inc = 0.0
                    for i in range(-1, -len(self.tick_data["time"]), -1):
                        if (self.tick_data["time"][-1] - self.tick_data["time"][i - 1]).total_seconds() > 900:
                            break
                        sum_inc += (self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1]) / self.tick_data["ma5"][-1]
                    if sum_inc > 0.0005:
                        return True
        return False

    def macd_down(self, snapshot):
        if len(self.tick_data["time"]) > 1000:
            if (self.tick_data["ma5"][-1] - self.tick_data["ma5"][-10] < 0) and (
                    self.tick_data["ma15"][-1] - self.tick_data["ma15"][-3] < 0):
                if self.ma5.get() < self.ma15.get():
                    return True
                sum_inc = 0.0
                for i in range(-1, -len(self.tick_data["time"]), -1):
                    if (self.tick_data["time"][-1] - self.tick_data["time"][i - 1]).total_seconds() > 900:
                        break
                    if self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1] < 0:
                        sum_inc += (self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1]) / self.tick_data["ma5"][-1]
                if sum_inc < -0.001:
                    return True
        return False

    def trend_up(self, snapshot):
        if self.last_time:
            if snapshot.time_stamp - self.last_time < 3600:
                return False
        # if snapshot.time_stamp - self.last_macd_time > 1800:
        #    return False
        if self.tick_data["ask"][-1] > self.ma2.get() * 1.01:
            return False
        if self.ma2.get() > self.ma5.get() > self.ma15.get() and len(self.tick_data["time"]) > 10:
            sum_inc = 0.0
            for i in range(-1, -len(self.tick_data["time"]), -1):
                if (self.tick_data["time"][-1] - self.tick_data["time"][i - 1]).total_seconds() > 300:
                    break
                """
                if self.tick_data["ma2"][i] - self.tick_data["ma2"][i - 1] < 0:
                    return False
                if self.tick_data["ma5"][i] - self.tick_data["ma5"][i - 1] < 0:
                    return False
                if self.tick_data["ma15"][i] - self.tick_data["ma15"][i - 1] < 0:
                    return False
                """
                sum_inc += (self.tick_data["ma2"][i] - self.tick_data["ma2"][i - 1]) / self.tick_data["ma2"][-1]
            if sum_inc / 300 > 0.001 / 600:
                return True
        return False

    def trend_down(self, snapshot):
        if self.ma5.get() < self.ma15.get():
            return True
        sum_delta = 0.0
        for i in range(-1, -len(self.tick_data["time"]), -1):
            if (self.tick_data["time"][-1] - self.tick_data["time"][i]).total_seconds() > 500:
                break
            sum_delta = min(self.tick_data["ma2"][-1] - self.tick_data["ma2"][i - 1], sum_delta)
        if sum_delta / self.tick_data["ma2"][-1] < -0.01:
            return True
        return False

    def stop_profit(self, price):
        if price / self.enter_price > 1.04:
            return True
        else:
            return False

    def stop_loss(self, price):
        if price / self.enter_price < 0.92:
            return True
        else:
            return False

    def show(self):
        super(Strategy, self).show()

        def random_color():
            import random
            x = random.random() * 256 * 256 * 256
            prefix = '#'
            for i in range(6 - len(str(hex(int(x))[2:]))):
                prefix = prefix + '0'
            return prefix + str(hex(int(x))[2:]).upper()

        plot = figure(x_axis_type="datetime")
        plot.line(x=self.tick_data["time"], y=self.tick_data["ma2"], legend="ma2", color=random_color())
        plot.line(x=self.tick_data["time"], y=self.tick_data["ma5"], legend="ma5", color=random_color())
        plot.line(x=self.tick_data["time"], y=self.tick_data["ma15"], legend="ma15", color=random_color())
        plot.scatter(x=self.tick_data["time"], y=self.tick_data["bid"], legend="bid", color=random_color())
        plot.scatter(x=self.tick_data["time"], y=self.tick_data["ask"], legend="ask", color=random_color())
        show(plot)


