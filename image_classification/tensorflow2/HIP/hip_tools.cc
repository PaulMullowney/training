
#include "hip_global.h"

extern "C" {

void* hip_allocate_device( size_t nbytes ){	
	void* dev_array;
	CHECK(hipMalloc( &dev_array, nbytes ));
	return dev_array;
}


void hip_free_device( void* dev_array ){
	CHECK(hipFree( dev_array));
}

int hip_set_device(int proc_id ){ 
	
  int n_devices; 
	CHECK( hipGetDeviceCount(&n_devices) );

	if (n_devices == 0){
		std::cout << " NO HIP-CAPABLE DEVICES FOUND!" << std::endl;
		return 0;
	}

	int device_id = proc_id % n_devices;
	CHECK( hipSetDevice(device_id) ); 

	hipDeviceProp_t prop;
	CHECK( hipGetDeviceProperties( &prop, device_id ) );
	std::cout <<  "Using HIP device: " << device_id << "/" << n_devices << "  " << prop.name  << std::endl;
	return device_id;
} 


void hip_get_device_properties( int device_id ){

	hipDeviceProp_t prop;
	CHECK( hipGetDeviceProperties( &prop, device_id ) );

  std::cout << "Device: " << device_id << " name: " << prop.name  << std::endl;
	std::cout << "Total global memory (bytes): " << prop.totalGlobalMem  << std::endl;
	std::cout << "Shared memory per block (bytes): " << prop.sharedMemPerBlock  << std::endl;
	std::cout << "Registers per block: " << prop.regsPerBlock  << std::endl;	
	std::cout << "Warp size: " << prop.warpSize  << std::endl;	
	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock  << std::endl;	

}


void hip_sync_device(){ CHECK( hipDeviceSynchronize() ); }


void hip_reset_device(){ CHECK( hipDeviceReset() ); }


void hip_copy_host_to_device( void *dev_array, void *host_array, size_t nbytes ){
	CHECK(hipMemcpy( dev_array, host_array, nbytes, hipMemcpyHostToDevice));
}

void hip_copy_device_to_host( void *host_array, void *dev_array, size_t nbytes ){
	CHECK(hipMemcpy( host_array, dev_array, nbytes, hipMemcpyDeviceToHost));
}

void hip_copy_device_to_device( void *dst_array, int dst_offset, 
																void *src_array, int src_offset, 
																size_t nbytes ){
	double *src_arr = (double *)src_array + src_offset;
	double *dst_arr = (double *)dst_array + dst_offset;
	CHECK(hipMemcpy( dst_arr, src_arr, nbytes, hipMemcpyDeviceToDevice));
}


}