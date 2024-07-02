import sys, os
import HIP.hip_tools as hip

roctx_profile = True

roctx_push = None
roctx_pop = None
roctx_start = None
roctx_stop = None


def init():
    from ctypes import cdll, c_int, c_char_p
    global roctx_push, roctx_pop
    global roctx_start, roctx_stop 

    hip_library_name = f'./HIP/libHIPcode.so'
    
    mylib = cdll.LoadLibrary( hip_library_name )
    
    mylib.test()
    #hip_library_name2 = f'/opt/rocm-6.1.1/lib/libroctx64.so'
    #mylib = cdll.LoadLibrary( hip_library_name2 )
    #print(f'Loading HIP2 library: {hip_library_name2}')
    
    roctx_push = mylib.push
    roctx_push.argtypes = [ c_char_p ]
    roctx_push.resypes = c_int 
    roctx_pop = mylib.pop
    
    roctx_start = mylib.start

    roctx_stop = mylib.stop
    

encode = lambda s : s.encode('utf-8')

def start():
    roctx_start()

def stop():
    roctx_stop()

def push( name ):
  if not roctx_profile: return
  id = roctx_push( encode(name) )
  #id = mylib.roctxRangePushA("t") #encode(name+"t")); 
  return id
  
def pop( id=None, sync_device=False ):
  if not roctx_profile: return
  if sync_device: hip.sync_device()
  roctx_pop(  )
  #mylib.roctxRangeStop()
