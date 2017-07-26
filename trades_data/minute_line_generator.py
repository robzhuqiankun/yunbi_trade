import pandas
import datetime
from yunbi.yunbi_data.yunbi_datafeed_lib import MinuteLine

MINUTE = 1
BEGIN_TIMESTAMP = 1494433206
out_file_path = "yunbi_btccny_minute.csv"
# data_files = ['yunbi_zeccny_28083872_30098613.csv', 'yunbi_zeccny_30098921_32303965.csv']
# data_files = ['yunbi_anscny_28048769_30874950.csv']
# data_files = ['yunbi_ethcny_27864118_29496077.csv', 'yunbi_ethcny_29496085_30098752.csv', 'yunbi_ethcny_30098777_30874954.csv']
# data_files = ['yunbi_btscny_4140067_29501435.csv', 'yunbi_btscny_29501464_30098751.csv', 'yunbi_btscny_30098756_30764671.csv']
# data_files = ['yunbi_etccny_24191032_29579858.csv', 'yunbi_etccny_29579922_32297029.csv']
# data_files = ['yunbi_gntcny_27407319_32296885.csv', ]
# data_files = ['yunbi_ethcny_25497371_33295710.csv', 'yunbi_ethcny_33295760_34720940.csv', 'yunbi_ethcny_34721063_38444267.csv']
data_files = ['yunbi_btccny_25497352_33295496.csv', 'yunbi_btccny_33295813_38444283.csv']
df = None
for data_file in data_files:
    if df is None:
        df = pandas.read_csv(data_file)
    else:
        df = pandas.concat([df, pandas.read_csv(data_file)], ignore_index=True)
df = df.sort_values('id')
df = df.sort_values('at')
df.rename(columns={"market": "symbol"}, inplace=True)
minute_list = []


class LineGenerator(object):
    trades_cache = []
    last_minute_line = None
    last_timestamp = None

    @classmethod
    def trades_callback(cls, trade):
        if cls.last_timestamp is None:
            cls.last_timestamp = trade.at

        if cls.last_timestamp + MINUTE * 60 > trade.at:
            cls.trades_cache.append(trade)
        else:
            cls.last_timestamp += MINUTE * 60
            cls.get_minute_line(trade.symbol)
            cls.trades_callback(trade)

    @classmethod
    def get_minute_line(cls, symbol):
        if len(cls.trades_cache) == 0:
            if cls.last_minute_line is not None:
                close = cls.last_minute_line.close
                cls.last_minute_line = MinuteLine(symbol=symbol, open_price=close,
                                                  close_price=close, high=close, low=close, vwap=close, volume=0,
                                                  t_time=datetime.datetime.fromtimestamp(cls.last_timestamp))
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
                                              volume=total_qty,
                                              t_time=datetime.datetime.fromtimestamp(cls.last_timestamp))
        cls.trades_cache = []
        if cls.last_minute_line is not None:
            dic = {}
            for key in cls.last_minute_line.__dict__:
                assert isinstance(key, str)
                if not key.startswith("__"):
                    dic[key] = cls.last_minute_line.__dict__[key]
            minute_list.append(dic)


for row in df.itertuples():
    if row.at < BEGIN_TIMESTAMP:
        continue
    LineGenerator.trades_callback(row)

df = pandas.DataFrame(minute_list).set_index('time_stamp')

df.to_csv(out_file_path)
