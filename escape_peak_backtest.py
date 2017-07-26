from yunbi.strategy.escape_peak import Strategy, Order
from yunbi.trade.client import get_api_path
from config import ACCOUNT_INFO
import os


def test_sccny_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_sc.json")
    demo = Strategy(symbols='sccny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_sccny_minute.csv', ])
    demo.run()


def test_ethny_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_eth.json")
    demo = Strategy(symbols='ethcny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_ethcny_minute.csv', ])
    demo.run()

def test_anscny_minute():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_ans.json")
    demo = Strategy(symbols='anscny', account=ACCOUNT_INFO, config_path=path, live=False,
                    data_files=['D:/bopu/yunbi_trade/trades_data/yunbi_anscny_minute.csv', ])
    demo.run()

if __name__ == '__main__':
    # back test
    # demo = Strategy(symbols='ethcny', account=ACCOUNT_INFO, live=False,
    #                data_files=["2017-05-29.log", "2017-05-30.log", "2017-05-31.log"])
    # demo.set_config({'order_qty': 100, 'enter_price': 0.5, 'symbol': 'ethcny'})
    # demo.run()
    # test_sccny_minute()
    # test_ethny_minute()
    test_anscny_minute()
