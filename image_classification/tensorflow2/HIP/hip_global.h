#ifndef HIP_GLOBAL_H
#define HIP_GLOBAL_H

#include <stdio.h>
#include <iostream>
#include "hip/hip_runtime.h"


#define HIP_ERROR_CHECK


#define CHECK(command) {   \
  hipError_t status = command; \
  if (status!=hipSuccess) {    \
    std::cout << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
    std::abort(); }}


inline void __hipCheckError( const char *file, const int line )
{
#ifdef HIP_ERROR_CHECK
    hipError_t err = hipGetLastError();
    if ( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }

    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    err = hipDeviceSynchronize();
    if( hipSuccess != err )
    {
        fprintf( stderr, "hipCheckError() with sync failed at %s:%i : %s\n",
                 file, line, hipGetErrorString( err ) );
        exit( -1 );
    }
#endif
    return;
}





#endif // HIP_GLOBAL_H

