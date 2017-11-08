#!/usr/bin/env python
from __future__ import print_function

class timer():
    '''A class to provide simpling timing statistics on
    specific parts of an executed code. Example:
    >>> import simple_timer
    >>> timer = simple_timer.timer()
    >>> timer.start("per1+2") # "per1+2" is a period name
    >>> func1() # executes a series of codes
    >>> timer.start("per2")
    >>> func2() # another series of codes
    >>> timer.end("per2")
    >>> timer.end("per1+2")
    >>> timer.close() # will print the times for periods
    >>>               # "per1+2", "per2", and the total time
    '''

    def __init__(self):
        import time
        self.start_time = time.time()
        self.period = {}
        self.pstart = {}
        self.key_order = []

    def start(self, key):
        '''Starts a timer for period <key>'''
        import time
        # make key case insensitive and limit to 20 chars
        key = key.lower()[:20].rjust(20)
        # start the timer for this period if it's set to None
        if key in self.pstart.keys():
            if self.pstart[key] is None:
                self.pstart[key] = time.time()
            else:
                pass # key already started, do nothing
        else:
            self.pstart.update({key: time.time()})
        # initialize total period time for this key
        if not key in self.period.keys():
            self.period.update({key: 0.})

    def end(self, key):
        '''Ends the timer for period <key>'''
        import time
        # make key case insensitive and limit to 20 chars
        key = key.lower()[:20].rjust(20)
        # quick error checks: if they key don't yet exist return
        if not (key in self.period.keys() and key in self.pstart.keys()): return
        if self.pstart[key] is None: return
        # period times are cummulative (add to old period time)
        self.period[key] += time.time() - self.pstart[key]
        self.pstart[key] = None
        # add key to key_order
        if key not in self.key_order: self.key_order.append(key)

    def close(self):
        '''End and print all period times'''
        import time
        print ("="*80+"\nTIMING STATISTICS:")
        # get total time
        ttime = time.time() - self.start_time
        # print warning for all unended periods
        [print ("WARNING: '{0}' period not ended!".format(key.lstrip()))
            for key in self.pstart.keys() if self.pstart[key] is not None]
        # print all separate period times
        [print ("  "+key+" {0:9.2f}s   ({1:5.2f}%)".format(self.period[key],
            self.period[key]*100/ttime)) for key in self.key_order
            if (key in self.period.keys() and self.pstart[key] is None)]
        # print total time
        print ("TOTAL".rjust(22)+" {0:9.2f}s".format(ttime)+"\n"+"="*80)

