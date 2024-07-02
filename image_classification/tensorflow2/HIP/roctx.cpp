
#include <roctx.h>
#include <roctracer_ext.h>
#include <hip/hip_runtime.h>

__global__
void kernel()
{
}

extern "C" {

	void test()
	{
		kernel<<<1,1>>>();
	}


void start(){
  roctracer_start();
}

void stop(){

// hipDeviceSynchronize();
  roctracer_stop();
}



void push( char *c){
  roctxRangePush(c);
}

void pop(){
  roctxRangePop();
}


}
