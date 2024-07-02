import numpy as np
import ctypes
import hip_tools as hip


class GPUArray:
  def __init__(self, log, np_arr=None, alloc_data=None, copy_to_device=True):
    self.dtype   = None
    self.shape   = None
    self.size    = 0
    self.nbytes  = 0
    self.dev_ptr = None 
    if alloc_data is not None:
      self.dtype   = alloc_data['dtype']
      self.size    = alloc_data['size']
      self.nbytes  = alloc_data['nbytes']
      self.shape   = alloc_data['shape']
      self.dev_ptr = alloc_data['dev_ptr'] 
      return
    if np_arr is not None:
      np_dtype    = np_arr.dtype 
      np_size     = np_arr.size
      np_shape    = np_arr.shape
      np_itemsize = np_arr.itemsize
      np_ptr      = np_arr.ctypes.data_as( ctypes.c_void_p )
      self.dtype   = np_dtype
      self.size    = np_size
      self.nbytes  = np_itemsize * np_size
      self.shape   = np_shape
      self.dev_ptr = ctypes.c_void_p( hip.allocate_device( self.nbytes ) )
      hip.gpu_allocated_memory += self.nbytes
      if copy_to_device: hip.copy_host_to_device( self.dev_ptr, np_ptr, self.nbytes )
      return 
  
  def free(self):
    if self.dev_ptr is not None: hip.free_device( self.dev_ptr )
    
  def get(self):
    if self.dev_ptr is None: return None
    host_arr = np.empty( self.shape, dtype=self.dtype )
    host_ptr = host_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
    hip.copy_device_to_host( host_ptr, self.dev_ptr, self.nbytes )
    hip.synchronize_device()
    return host_arr

  def set(self, log, np_arr):
    np_dtype    = np_arr.dtype 
    np_size     = np_arr.size
    np_shape    = np_arr.shape
    np_itemsize = np_arr.itemsize
    np_ptr      = np_arr.ctypes.data_as( ctypes.c_void_p )
    if self.dev_ptr is None:
      print('ERROR: GPUArray set. device array not allocated. ')
      return
    if self.size != np_size:
      print('ERROR: GPUArray set. np_size != size ')
      return 
    if self.nbytes != np_itemsize*np_size:
      print('ERROR: GPUArray set. np_nbytes != nbytes ')
      return
    hip.copy_host_to_device( self.dev_ptr, np_ptr, self.nbytes )
    
  def copy_to_host( self, log, np_arr ):
    np_dtype    = np_arr.dtype 
    np_size     = np_arr.size
    np_shape    = np_arr.shape
    np_itemsize = np_arr.itemsize
    np_ptr      = np_arr.ctypes.data_as( ctypes.c_void_p )
    if self.dev_ptr is None:
      print('ERROR: GPUArray copy_to_host. device array not allocated. ')
      return
    if self.size != np_size:
      print('ERROR: GPUArray copy_to_host. np_size != size ')
      return 
    if self.nbytes != np_itemsize*np_size:
      print('ERROR: GPUArray copy_to_host. np_nbytes != nbytes ')
      return
    hip.copy_device_to_host( np_ptr, self.dev_ptr, self.nbytes )
    hip.synchronize_device()
    

