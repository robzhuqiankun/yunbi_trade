import json
import time
import logging
import threading
import datetime
import Queue
import pandas
from yunbi.trade.utility import IntervalTimer, lock
from yunbi.trade.client import Client, get_api_path


class Snapshot(object):
    def __init__(self, symbol, bid, ask, last_price, time_stamp, **kwargs):
        self.symbol = symbol
        self.bid = float(bid)
        self.ask = float(ask)
        self.last_price = float(last_price)
        self.time_stamp = int(time_stamp)

    def __str__(self):
        return 'S|%s@%s bid:%s ask:%s last:%s' % (self.symbol, self.time_stamp, self.bid, self.ask, self.last_price)


class Trade(object):
    def __init__(self, symbol, price, volume, side, broker_id, time_stamp, **kwargs):
        self.symbol = symbol
        self.price = float(price)
        self.volume = float(volume)
        self.side = 1 if (str(side) == "up") else -1
        self.broker_id = int(broker_id)
        self.time_stamp = int(time_stamp)

    def __str__(self):
        return 'T|%s#%s @%s price:%s volume:%s side:%s' % (self.symbol, self.broker_id, self.time_stamp, self.price,
                                                           self.volume, self.side)


class MinuteLine(object):
    def __init__(self, symbol, open_price, close_price, high, low, vwap, volume, t_time, **kwargs):
        self.symbol = symbol
        self.open = float(open_price)
        self.close = float(close_price)
        self.high = float(high)
        self.low = float(low)
        self.vwap = float(vwap)
        self.volume = float(volume)
        self.t_time = t_time
        assert isinstance(t_time, datetime.datetime)
        self.time_stamp = int(time.mktime(t_time.timetuple()))

    def __str__(self):
        return 'M|%s@%s time:%s open:%s close:%s vol:%s vwap:%s' % (self.symbol, self.time_stamp, self.t_time,
                                                                    self.open, self.close, self.volume, self.vwap)


class DataFeedType(object):
    ORDER_BOOK = 1
    SNAPSHOT = 2
    TRADE = 3
    MINUTE_LINE = 4


class DataFeed(object):
    def __init__(self, query_interval=1):
        self.replays = None
        self.sub_list = []
        self.callbacks = {}
        self.query_interval = query_interval
        self.living = False

        self.msg_queue = Queue.Queue()
        self.client = Client(access_key='access', secret_key='xx')

    """
    def subscribe(self, symbols):
        if isinstance(symbols, str):
            self.sub_list.append(symbols)
        else:
            for symbol in symbols:
                self.sub_list.append(symbol)
    """

    def subscribe(self, data_type, symbol, callback):
        if (data_type, symbol) in self.callbacks:
            self.callbacks[(data_type, symbol)].append(callback)
        else:
            self.callbacks[(data_type, symbol)] = [callback, ]
            if self.living:
                if data_type is DataFeedType.TRADE:
                    threading.Thread(target=self._trades_producer, args=(symbol,)).start()
                if data_type is DataFeedType.MINUTE_LINE:
                    threading.Thread(target=self._minute_line_producer, args=(symbol,)).start()

    @staticmethod
    def to_snap(snap, symbol):
        ticker = snap['ticker']
        return Snapshot(symbol, ticker['buy'], ticker['sell'], ticker['last'], snap['at'])

    def setup_live(self):
        pass

    def setup_replay_files(self, replays_dict):
        """
        :param replays_dict: {DataFeedType.ORDER_BOOK: [file1_path, ...], DataFeedType.TRADES: [], ...}
        """
        self.replays = replays_dict

    def _data_dispatcher(self, empty_once=False):
        while True:
            if not self.msg_queue.empty():
                (data_type, symbol, data) = self.msg_queue.get()
                if (data_type, symbol) in self.callbacks:
                    callback_list = self.callbacks[(data_type, symbol)]
                    for callback in callback_list:
                        callback(data)
            if empty_once and self.msg_queue.empty():
                break

    def _snapshot_producer(self):
        latest_snapshot = dict()

        def _is_updated_snapshot(symbol_, snapshot_raw):
            if symbol_ not in latest_snapshot:
                latest_snapshot[symbol_] = snapshot_raw
                return True
            else:
                if str(latest_snapshot[symbol_][u'ticker']) != str(snapshot_raw[u'ticker']):
                    latest_snapshot[symbol_] = snapshot_raw
                    return True
            return False

        while True:
            try:
                snapshots = self.client.get_public(get_api_path('all_tickers'), timeout=5)
                for symbol in snapshots:
                    if (DataFeedType.SNAPSHOT, symbol) not in self.callbacks:
                        continue
                    if _is_updated_snapshot(symbol, snapshots[symbol]):
                        self.msg_queue.put((DataFeedType.SNAPSHOT, symbol, self.to_snap(snapshots[symbol], symbol)))
                time.sleep(self.query_interval)
            except Exception, e:
                logging.error("Datafeed updating snapshot: " + str(e))
                time.sleep(3)

    def _trades_producer(self, symbol):
        last_trade_id = None
        while True:
            try:
                raw_trades = self.client.get_public(get_api_path("trades"), {"market": symbol, "limit": 1}, 10)
                if raw_trades is None or len(raw_trades) == 0:
                    continue
                last_trade_id = int(raw_trades[0]["id"])
                break
            except Exception, e:
                logging.error("Datafeed updating trades: " + str(e))
        logging.info("Datafeed updating trades from broker_id: " + str(last_trade_id))
        while True:
            try:
                raw_trades = self.client.get_public(get_api_path("trades"), {"market": symbol, "from": last_trade_id,
                                                    "limit": 1000}, timeout=10)
                if raw_trades is None or len(raw_trades) == 0:
                    continue
                trade_list = []
                for trade in raw_trades:
                    trade_list.append(Trade(symbol=trade["market"], price=trade["price"], volume=trade["volume"],
                                            side=trade["side"], broker_id=trade["id"], time_stamp=trade["at"]))
                last_trade_id = int(raw_trades[0]["id"])
                trade_list.reverse()
                for trade in trade_list:
                    self.msg_queue.put((DataFeedType.TRADE, symbol, trade))
                time.sleep(5)
            except Exception, e:
                logging.error("Datafeed updating trades: " + str(e))
                time.sleep(5)

    def _minute_line_producer(self, symbol, interval=60):
        class LineGenerator(object):
            trades_cache = []
            mutex = threading.Lock()
            last_minute_line = None

            @classmethod
            def trades_callback(cls, trade):
                with cls.mutex:
                    cls.trades_cache.append(trade)

            @classmethod
            def get_minute_line(cls):
                with cls.mutex:
                    if len(cls.trades_cache) == 0:
                        if cls.last_minute_line is not None:
                            close = cls.last_minute_line.close
                            cls.last_minute_line = MinuteLine(symbol=symbol, open_price=close, close_price=close,
                                                              high=close, low=close, vwap=close, volume=0,
                                                              t_time=datetime.datetime.now())
                    else:
                        total_qty = 0.0
                        total_mkt = 0.0
                        high = None
                        low = None
                        for trade in cls.trades_cache:
                            total_qty += trade.volume
                            total_mkt += trade.price * trade.volume
                            high = trade.price if high is None else max(high, trade.price)
                            low = trade.price if low is None else min(low, trade.price)
                        cls.last_minute_line = MinuteLine(symbol=symbol, high=high, low=low, vwap=total_mkt / total_qty,
                                                          open_price=cls.trades_cache[0].price,
                                                          close_price=cls.trades_cache[-1].price,
                                                          volume=total_qty, t_time=datetime.datetime.now())
                    cls.trades_cache = []
                if cls.last_minute_line is not None and cls.last_minute_line.volume > 0:  # No minute line of zero vol
                    self.msg_queue.put((DataFeedType.MINUTE_LINE, symbol, cls.last_minute_line))

        scheduler = IntervalTimer(interval, LineGenerator.get_minute_line)
        scheduler.start()

        self.subscribe(DataFeedType.TRADE, symbol, LineGenerator.trades_callback)

    def live(self):
        print 'start datafeed live!'
        self.living = True
        sub_snapshot = False
        for (data_type, symbol) in self.callbacks:
            if data_type is DataFeedType.SNAPSHOT and sub_snapshot is False:
                sub_snapshot = True
                threading.Thread(target=self._snapshot_producer).start()
            if data_type is DataFeedType.TRADE:
                threading.Thread(target=self._trades_producer, args=(symbol,)).start()
            if data_type is DataFeedType.MINUTE_LINE:
                threading.Thread(target=self._minute_line_producer, args=(symbol,)).start()

        consumer = threading.Thread(target=self._data_dispatcher)
        consumer.start()
        consumer.join()

    def replay(self, from_stamp=0, to_stamp=-1):
        print "start replay"
        for df in self.replays:
            with open(df) as f:
                print "start read " + df
                raw_lines = f.readlines()
                print "end read " + df
                for line in raw_lines:
                    if line.count('@$'):
                        (head, js,) = line.split('||')
                        symbol = head.split('$')[-1]
                        if symbol in self.sub_list:
                            self.callback(self.to_snap(json.loads(js), symbol))

    def replay_minute_line(self, from_stamp=0, to_stamp=-1):

        def in_time_range(time_stamp):
            if time_stamp > from_stamp:
                if to_stamp == -1 or time_stamp < to_stamp:
                    return True
            return False

        for data_file in self.replays[DataFeedType.MINUTE_LINE]:
            df = pandas.read_csv(data_file)
            for row in df.itertuples():
                if in_time_range(row.time_stamp):
                    dic = dict(row.__dict__)
                    dic['open_price'] = dic['open']
                    dic['close_price'] = dic['close']
                    dic['t_time'] = datetime.datetime.fromtimestamp(dic['time_stamp'])
                    self.msg_queue.put((DataFeedType.MINUTE_LINE, row.symbol, MinuteLine(**dic)))

        consumer = threading.Thread(target=self._data_dispatcher, args=(True, ))
        consumer.start()
        consumer.join()


def test():
    def callback(s):
        print s

    client = DataFeed()
    client.setup_replay_files({DataFeedType.MINUTE_LINE: ['D:/bopu/yunbi_trade/trades_data/yunbi_sccny_minute.csv', ]})
    client.subscribe(DataFeedType.MINUTE_LINE, 'sccny', callback=callback)
    client.replay_minute_line()


def test_live():
    from yunbi.trade.utility import lock, init_logging
    from threading import Timer
    import threading
    mutex = threading.Lock()

    @lock(mutex)
    def callback(s):
        print s
        time.sleep(3)
        print 'sleep end!'

    init_logging("test_datafeed_log")

    client = DataFeed()
    client.subscribe(DataFeedType.SNAPSHOT, 'sccny', callback)
    client.subscribe(DataFeedType.MINUTE_LINE, 'sccny', callback)
    client.subscribe(DataFeedType.MINUTE_LINE, 'dgdcny', callback)

    t = threading.Thread(target=client.live())
    t.start()
    t.join()

if __name__ == '__main__':
    test()
    # test_live()
    # 29787307
