from yunbi_datafeed_lib import DataFeed, Snapshot
from pandas import Series
import datetime
from bokeh.plotting import figure, show


class Order(object):
    def __init__(self, symbol, side, price, quantity, at):
        self.symbol = symbol
        self.side = int(side)
        self.price = float(price)
        self.quantity = int(quantity)
        self.at = datetime.datetime.fromtimestamp(at)

    def __str__(self):
        return '%s@%s side:%s price:%s quantity:%s' % (self.symbol, self.at, self.side, self.price, self.quantity)


class BackTest(object):
    def __init__(self, symbols, data_files, init_balance=0):
        self.balance = init_balance
        self.trades = []
        self.nets = []
        self.df_client = DataFeed()
        self.df_client.setup_replay(data_files, self.on_snap)
        self.df_client.subscribe(symbols)

    def run(self):
        self.df_client.replay()
        print "replay done!"
        self.show()

        # Series(self.nets).plot()

    def show(self):
        for trade in self.trades:
            print trade
        print self.nets

    def on_snap(self, snapshot):
        pass

    def place(self, order):
        self._place(order)

    def _place(self, order):
        assert isinstance(order, Order)
        self.balance -= order.price * order.quantity * order.side + 0.001 * order.price * order.quantity
        self.trades.append(order)
        if order.side < 0:
            self.nets.append(self.balance)
