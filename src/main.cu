#include <stdio.h>
#include <cuda.h>
#include <dlfcn.h>
#include "CudaUtils.h"
#include "NvFBCUtils.h"
#include "NvFBC.h"

__global__ void frameTestKernel(unsigned char* frame) {
	for (int i = 0; i < 12; i++) {
		printf("%i ", *(frame + i));
	}

	printf("\n");
}

int main() {
	void *libNVFBC = NULL;
	PNVFBCCREATEINSTANCE NvFBCCreateInstance_ptr = NULL;
	NVFBC_API_FUNCTION_LIST pFn;
	NVFBC_SESSION_HANDLE fbcHandle;
	CUcontext cuCtx;

	/*
	 * Dynamically load the NvFBC library.
	 */
	loadNvFBC(libNVFBC);

	NVFBC_BOOL fbcBool = cudaInit(&cuCtx);
	if (fbcBool != NVFBC_TRUE) {
		return EXIT_FAILURE;
	}

	nvfbcResolveCreateInstance(libNVFBC, NvFBCCreateInstance_ptr);
	nvfbcCreateInstance(NvFBCCreateInstance_ptr, pFn);
	nvfbcCreateSessionHandle(pFn, fbcHandle);
	nvfbcCreateCaptureSession(pFn, fbcHandle);	
	nvfbcSetupCaptureSession(pFn, fbcHandle);

	CUdeviceptr cuDevicePtr;

	nvfbcGrabFrame(pFn, fbcHandle, cuDevicePtr);

	frameTestKernel<<<1,1>>>((unsigned char*)cuDevicePtr);

	nvfbcDestroyCaptureSession(pFn, fbcHandle);
	nvfbcDestroySessionHandle(pFn, fbcHandle);

	return 0;
}