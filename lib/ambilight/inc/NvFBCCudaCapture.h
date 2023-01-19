#ifndef __NVFBC_CUDA_CAPTURE_H
#define __NVFBC_CUDA_CAPTURE_H

#include "cuda.h"
#include "NvFBC.h"
#include "NvFBCUtils.h"

class NvFBCCudaCapture {
private:
	void* libNvFBC = NULL;
	NVFBC_API_FUNCTION_LIST instance;
	NVFBC_SESSION_HANDLE fbcHandle;
	CUcontext cuCtx;
public:
	NvFBCCudaCapture(CUcontext cuCtx);

	bool waitUntilReady = false;

	NVFBC_BOOL load();
	NVFBC_BOOL createInstance();
	NVFBC_BOOL createSessionHandle();
	NVFBC_BOOL createCaptureSession();
	NVFBC_BOOL setupCaptureSession();

	NVFBC_BOOL grabFrame(CUdeviceptr &cuDevicePtr);

	NVFBC_BOOL destroyCaptureSession();
	NVFBC_BOOL destroySessionHandle();
};

#endif