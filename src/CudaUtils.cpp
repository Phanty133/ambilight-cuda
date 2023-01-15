#include "CudaUtils.h"

NVFBC_BOOL cudaInit(CUcontext *cuCtx)
{
	CUresult cuRes;
	CUdevice cuDev;

	cuRes = cuInit(0);
	if (cuRes != CUDA_SUCCESS) {
		fprintf(stderr, "Unable to initialize CUDA (result: %d)\n", cuRes);
		return NVFBC_FALSE;
	}

	cuRes = cuDeviceGet(&cuDev, 0);
	if (cuRes != CUDA_SUCCESS) {
		fprintf(stderr, "Unable to get CUDA device (result: %d)\n", cuRes);
		return NVFBC_FALSE;
	}

	cuRes = cuCtxCreate_v2(cuCtx, CU_CTX_SCHED_AUTO, cuDev);
	if (cuRes != CUDA_SUCCESS) {
		fprintf(stderr, "Unable to create CUDA context (result: %d)\n", cuRes);
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}