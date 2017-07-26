from ..yunbi_data.yunbi_datafeed_lib import DataFeed, Snapshot, MinuteLine, DataFeedType
from async_client import AsyncClient
from client import get_api_path
import datetime
import time
import threading
from utility import lock


class Order(object):
    def __init__(self, symbol, side, price, quantity, at):
        self.symbol = symbol
        self.side = int(side)
        self.price = float(price)
        self.quantity = float(quantity)
        self.at = datetime.datetime.fromtimestamp(at)

    def __str__(self):
        return '%s@%s side:%s price:%s quantity:%s' % (self.symbol, self.at, self.side, self.price, self.quantity)


class AccountSim(object):
    def __init__(self, balance=100000000, position=0):
        self.balance = balance
        self.position = position
        self.trades = []
        self.nets = []

    def __str__(self):
        return 'Sim account|balance: %s position: %s' % (self.balance, self.position)


class TradeClient(object):
    trade_lock = threading.Lock()

    def __init__(self, symbols, account, live=False, **params):
        self.is_live = live
        self.client = AsyncClient(access_key=account['access_key'], secret_key=account['secret_key'])
        self.balance = 0
        self.position = 0

        self.snapshot_callback = None
        self.minute_callback = None
        self.df_client = DataFeed()
        self.df_client.subscribe(DataFeedType.MINUTE_LINE, symbols, callback=self._on_minute)
        if self.is_live:
            pass
        else:
            self.df_client.setup_replay_files(
                {DataFeedType.MINUTE_LINE: params['data_files']})
            self.sim_account = AccountSim(params.get('balance', 100000000), params.get('position', 0))

    def run(self):
        if self.is_live:
            self.df_client.live()
        else:
            self.df_client.replay_minute_line()
            self.back_test_performance()

    def back_test_performance(self):
        from collections import OrderedDict
        import pandas
        daily_ret = OrderedDict()

        for trade in self.sim_account.trades:
            print trade

        print self.sim_account.nets
        num = len(self.sim_account.nets)
        print '#trades: ' + str(len(self.sim_account.nets))
        wins = 0
        for i in range(0, len(self.sim_account.trades) - 1):
            if not (self.sim_account.trades[i].side == 1 and self.sim_account.trades[i + 1].side == -1):
                continue
            delta = (self.sim_account.trades[i].price * self.sim_account.trades[i].side + self.sim_account.trades[
                        i + 1].price * self.sim_account.trades[i + 1].side) / self.sim_account.trades[i].price
            t_date = self.sim_account.trades[i + 1].at.date()
            if t_date in daily_ret:
                daily_ret[t_date] += -delta
            else:
                daily_ret[t_date] = -delta
            if self.sim_account.trades[i].price * self.sim_account.trades[i].side + self.sim_account.trades[
                        i + 1].price * self.sim_account.trades[i + 1].side < 0:
                wins += 1
        if len(daily_ret) > 0:
            begin_date = self.sim_account.trades[0].at.date()
            new_ret = OrderedDict()
            for day in daily_ret:
                while day > begin_date:
                    new_ret[begin_date] = 0
                    begin_date = (begin_date + datetime.timedelta(days=1))
                new_ret[begin_date] = daily_ret[day]
                begin_date = (begin_date + datetime.timedelta(days=1))
            ret = pandas.Series(new_ret)
            print 'Total return: %s' % ret.sum()
            ret.to_csv('pnl.csv')

        print "win: %s, loss: %s, ratio: %s, profit: %s" % (wins, num - wins, 1.0 * wins/num, self.sim_account.nets[-1])

    def register_snapshot_callback(self, callback):
        self.snapshot_callback = callback

    def register_minute_callback(self, callback):
        self.minute_callback = callback

    @lock(trade_lock)
    def _on_snap(self, snapshot):
        self.snapshot_callback(snapshot)

    @lock(trade_lock)
    def _on_minute(self, minute_line):
        self.minute_callback(minute_line)

    def query_account(self, code):
        if self.is_live is False:
            return self.sim_account.balance, self.sim_account.position
        bl = 0
        pos = 0
        account_info = self.client.get(get_api_path('members'))
        for currency in account_info['accounts']:
            if currency['currency'] == 'cny':
                bl = currency['balance']
            if currency['currency'] == code:
                pos = currency['balance']
        return float(bl), float(pos)

    def hit_place(self, order, target_position):
        assert isinstance(order, Order)
        code = order.symbol[0:-3]

        if self.is_live is False:
            self._back_test_place(order)
            return self.query_account(code)

        (balance, position) = self.query_account(code)
        self.balance, self.position = balance, position
        if order.side > 0:
            if position > 0.9 * target_position:
                print 'has bought'
                return balance, position
            if balance > order.price * order.quantity * 1.2:
                params = {'market': order.symbol, 'side': 'buy', 'volume': order.quantity, 'price': order.price}
                print params
                place_res = self.client.post(get_api_path('orders'), params)  # exchange server will rounding
                time.sleep(3)
                cancel_res = self.client.post(get_api_path('clear'))
                return self.query_account(code)

        if order.side < 0:
            if position < 0.01 * target_position:
                print 'has sold'
                return balance, position
            if position > order.quantity * 0.1:
                params = {'market': order.symbol, 'side': 'sell', 'volume': min(order.quantity, position), 'price': order.price}
                place_res = self.client.post(get_api_path('orders'), params)
                time.sleep(3)
                cancel_res = self.client.post(get_api_path('clear'))
                return self.query_account(code)

    def place(self, order, callback=None):
        pass

    def _back_test_place(self, order):
        assert isinstance(order, Order)
        self.sim_account.balance -= order.price * order.quantity * order.side + 0.001 * order.price * order.quantity
        self.sim_account.position += order.quantity * order.side
        self.sim_account.trades.append(order)
        if self.sim_account.position == 0:
            self.sim_account.nets.append(self.sim_account.balance)
        print order
