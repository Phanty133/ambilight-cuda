#ifndef __CUDA_UTILS_H
#define __CUDA_UTILS_H

#include <stdio.h>
#include "NvFBC.h"
#include "cuda.h"

/**
 * Initializes CUDA and creates a CUDA context.
 *
 * \param [in] cuCtx
 *   A pointer to the created CUDA context.
 *
 * \return
 *   NVFBC_TRUE in case of success, NVFBC_FALSE otherwise.
 */
NVFBC_BOOL cudaInit(CUcontext *cuCtx);

#endif