from __future__ import unicode_literals
from wxpy import get_wechat_logger


from wxpy import *
from wechat_sender import listen, Sender
logger = get_wechat_logger()

while True:
    import time
    import sys
    logger.warning(sys.argv[1])
    time.sleep(10)

"""
@bot.register(Friend)
def reply_test(msg):
    # msg.reply('test')
    pass
"""

#sender = Sender()
#sender.send_to("Hello From Wechat Sender", 'ww')

"""
my_friend = bot.friends().search('zhuqiankun')[0]

my_friend.send('Hello WeChat!')
Sender().send('test')
"""

# listen(bot)




