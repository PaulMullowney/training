import sys, os
from ctypes import cdll, c_int, c_size_t, c_void_p, c_double
import ctypes 
c_double_p = ctypes.POINTER(c_double)

# HIP functions
set_device = None
get_device_properties = None
sync_device = None

allocate_device = None
free_device = None

copy_device_to_host = None
copy_host_to_device = None
copy_device_to_device = None

daxpy = None

def init():
  global set_device, get_device_properties
  global sync_device,  allocate_device, free_device
  global copy_host_to_device, copy_device_to_host, copy_device_to_device
  global daxpy
  
  global libhip2

  hip_library_name = f'./HIP/libHIPcode.so'
  print(f'Loading HIP library: {hip_library_name}')
  
  hip_library_name2 = f'/opt/rocm-6.0.2/lib/libroctx64.so'
  print(f'Loading HIP2 library: {hip_library_name2}')

  libhip = cdll.LoadLibrary( hip_library_name )
  libhip2 = cdll.LoadLibrary( hip_library_name2 )

  set_device = libhip.hip_set_device
  set_device.argtypes = [ c_int ]
  set_device.restypes = c_int 

  allocate_device = libhip.hip_allocate_device
  allocate_device.argtypes = [ c_size_t ]
  allocate_device.restype = c_void_p   

  free_device = libhip.hip_free_device
  free_device.argtypes = [ c_void_p ]

  free_device = libhip.hip_free_device
  free_device.argtypes = [ c_void_p ]    
    
  copy_host_to_device = libhip.hip_copy_host_to_device
  copy_host_to_device.argtypes = [ c_void_p, c_void_p, c_size_t ]

  copy_device_to_host = libhip.hip_copy_device_to_host
  copy_device_to_host.argtypes = [ c_void_p, c_void_p, c_size_t ]

  copy_device_to_device = libhip.hip_copy_device_to_device
  copy_device_to_device.argtypes = [ c_void_p, c_int, c_void_p, c_int, c_size_t ]
  
  daxpy = libhip.gpu_daxpy
  daxpy.argtypes = [ c_int, c_double, c_void_p, c_void_p ]

