from multiprocessing.pool import ThreadPool
from client import Client


class AsyncClient(Client):
    def __init__(self, access_key=None, secret_key=None):
        super(AsyncClient, self).__init__(access_key, secret_key)
        self.pool = ThreadPool(5)

    def async_get(self, callback, path, params=None):
        self.pool.apply_async(self.get, (path, params, ), callback=callback)

    def async_post(self, callback, path, params=None):
        self.pool.apply_async(self.post, (path, params,), callback=callback)


class test(object):
    def __init__(self):
        self.pool = ThreadPool(5)

    def do(self, b):
        print b

    def asy(self):
        pass