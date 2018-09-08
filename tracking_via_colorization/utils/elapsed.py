#-*- cording: utf-8 -*-
r""" utils/elapsed
"""
import time


class Elapsed():
    r"""Elapsed check utility class"""
    def __init__(self):
        self.clear()

    def clear(self):
        self.timestamps = [('total', time.time())]
        self.elapsed = {}

    def tic(self, name):
        self.timestamps.append((name, time.time()))

    def calc(self):
        self.elapsed = {'total': self.timestamps[-1][1] - self.timestamps[0][1]}
        self.elapsed.update({t[0]: t[1] - self.timestamps[i][1] for i, t in enumerate(self.timestamps[1:])})
        return self.elapsed

    def __repr__(self):
        self.calc()
        return ' '.join(['{}:{:.3f}'.format(key, self.elapsed[key]) for key, _ in self.timestamps])
