import pandas
import sys
import time
from yunbi.trade.client import Client, get_api_path
from yunbi.trade.utility import init_logging
import logging

init_logging("trades_crawler_log")

min_id = 0
last_id = 0
total_len = 0
symbol = ""
if len(sys.argv) == 4:
    symbol = str(sys.argv[1])
    last_id = int(sys.argv[2])
    total_len = int(sys.argv[3])
elif len(sys.argv) == 5:
    symbol = str(sys.argv[1])
    min_id = int(sys.argv[2])
    last_id = int(sys.argv[3])
    total_len = int(sys.argv[4])
else:
    exit()

client = Client(access_key='xx', secret_key='yy')
id_set = set()
trade_list = []
while True:
    try:
        trades = client.get_public(get_api_path("trades"), {"market": symbol, "to": last_id, "limit": 1000}, 15)

    except Exception, e:
        print str(e)
        time.sleep(10)
        continue

    if len(trades) == 0:
        break

    temp_id = last_id
    for trade in trades:
        trade_id = trade["id"]
        if temp_id > trade_id:
            temp_id = trade_id
        if trade_id not in id_set and trade_id > min_id:
            id_set.add(trade_id)
            trade_list.append(trade)
    logging.info("Fetched trades [%s, %s), total: %s" % (temp_id, last_id, len(id_set)))
    if temp_id <= min_id:
        break
    last_id = temp_id
    if len(id_set) > total_len:
        break
    time.sleep(5)

if len(trade_list) > 0:
    df = pandas.DataFrame(trade_list)
    df = df.set_index("id")
    df.to_csv("trades_data/yunbi_%s_%s_%s.csv" % (symbol, min(id_set), max(id_set)))
