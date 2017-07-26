from yunbi.strategy.macd import Strategy, Order
from yunbi.trade.client import get_api_path
from config import ACCOUNT_INFO
import os

if __name__ == '__main__':
    # back test
    # demo = Strategy(symbols='ethcny', account=ACCOUNT_INFO, live=False,
    #                data_files=["2017-05-29.log", "2017-05-30.log", "2017-05-31.log"])
    # demo.set_config({'order_qty': 100, 'enter_price': 0.5, 'symbol': 'ethcny'})
    # demo.run()
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "trade_sc.json")
    demo = Strategy(symbols='sccny', account=ACCOUNT_INFO, config_path=path, live=True)
    demo.run()
