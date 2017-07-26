import threading
import os
import logging
import datetime


def lock(mutex):
    def decorator(func):
        def wrapper(*args, **kw):
            with mutex:
                ret = func(*args, **kw)
            return ret

        return wrapper

    return decorator


class IntervalTimer(object):
    def __init__(self, interval, callback, *args, **kwargs):
        self.callback = callback
        self.interval = interval
        self.args = args
        self.kwargs = kwargs
        self.timer = None

    def start(self):
        self.timer = threading.Timer(self.interval, self.start)
        self.timer.start()
        self.callback(*self.args, **self.kwargs)


def init_logging(log_path_name):
    """
    Gets the logger that prints to both local file and the screen.

    If you want to log in another thread, you must call get_logger in that thread.
    and use the logger retrieved to write logs.

    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # use current timestamp as the log file name
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), log_path_name)):
        os.makedirs(os.path.join(os.path.dirname(os.path.realpath(__file__)), log_path_name))
    log_file = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__)), log_path_name),
                            '{}.log'.format(datetime.datetime.now().strftime('%Y-%m-%d')))

    # write log to file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)-8s %(message)s',
                                                datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # write log to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    logger.addHandler(console_handler)

