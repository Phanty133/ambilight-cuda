#include <stdio.h>
#include <cuda.h>
#include <dlfcn.h>
#include "CudaUtils.h"
#include "NvFBCCudaCapture.h"
#include "NvFBC.h"

__global__ void frameTestKernel(unsigned char* frame) {
	for (int i = 0; i < 12; i++) {
		printf("%i ", *(frame + i));
	}

	printf("\n");
}

int main() {
	CUcontext cuCtx;

	if (cudaInit(&cuCtx) != NVFBC_TRUE) {
		return EXIT_FAILURE;
	}

	NvFBCCudaCapture fbcCapture(cuCtx);

	fbcCapture.load();
	fbcCapture.createInstance();
	fbcCapture.createSessionHandle();
	fbcCapture.createCaptureSession();
	fbcCapture.setupCaptureSession();

	CUdeviceptr cuDevicePtr;

	fbcCapture.grabFrame(cuDevicePtr);

	frameTestKernel<<<1,1>>>((unsigned char*)cuDevicePtr);

	fbcCapture.destroyCaptureSession();
	fbcCapture.destroySessionHandle();

	return 0;
}