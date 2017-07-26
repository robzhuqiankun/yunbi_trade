from yunbi.strategy.macd import Strategy, Order
from yunbi.trade.client import get_api_path
from config import ACCOUNT_INFO
import os


def test_sccny_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_sc.json")
    demo = Strategy(symbols='sccny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_sccny_minute.csv', ])
    demo.run()


def test_anscny_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_ans.json")
    demo = Strategy(symbols='anscny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_anscny_minute.csv', ])
    demo.run()


def test_eth_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_eth.json")
    demo = Strategy(symbols='ethcny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_ethcny_minute.csv', ])
    demo.run()


def test_etc_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_etc.json")
    demo = Strategy(symbols='etccny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_etccny_minute.csv', ])
    demo.run()


def test_bts_minutes():
    demo = Strategy(symbols='btscny', account=ACCOUNT_INFO, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_btscny_minute.csv', ])
    demo.set_config({'order_qty': 1, 'enter_price': 0.3, 'symbol': 'btscny'})
    demo.run()


def test_qtum_minutes():
    demo = Strategy(symbols='qtumcny', account=ACCOUNT_INFO, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_qtumcny_minute.csv', ])
    demo.set_config({'order_qty': 1, 'enter_price': 10, 'symbol': 'qtumcny'})
    demo.run()


def test_zec_minutes():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_zec.json")
    demo = Strategy(symbols='zeccny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_zeccny_minute.csv', ])
    demo.run()


def test_gnt_minutes():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_gnt.json")
    demo = Strategy(symbols='gntcny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_gntcny_minute.csv', ])
    demo.run()

if __name__ == '__main__':
    # back test
    # test_sccny_minute()
    # test_anscny_minute()
    # test_eth_minute()
    # test_etc_minute()
    # test_bts_minutes()
    # test_qtum_minutes()
    # test_zec_minutes()
    test_gnt_minutes()
    pass
