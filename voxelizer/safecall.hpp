#ifndef __SAFECALL_HPP__
#define __SAFECALL_HPP__

#include <cuda_runtime_api.h>
#include <string>
#include <exception>

static inline void mysafecall(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        printf("ERROR at FILE %s (LINE: %d): %s\n", file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#define cudaSafeCall(expr) mysafecall(expr, __FILE__, __LINE__)

#endif 