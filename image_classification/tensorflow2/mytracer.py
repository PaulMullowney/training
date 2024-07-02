import time
import HIP.roctx as roctx

#import tensorflow as tf


#@tf.py_function(Tout=[])
def trace_time(func):
#    @tf.py_function(Tout=[])
    def timed(*args, **kwargs):
        
        ts = time.time()
        roctx.push(func.__name__)
        result = func(*args, **kwargs)
        roctx.pop()
        te = time.time()
        #tf.print('Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        #print('Function', func.__name__, 'time:', round((te -ts)*1000,1), 'ms')
        #print(flush=True)
        
        return result
    return timed
