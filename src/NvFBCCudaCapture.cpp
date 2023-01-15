#include "NvFBCCudaCapture.h"

NvFBCCudaCapture::NvFBCCudaCapture(CUcontext cuCtx) {
	this->cuCtx = cuCtx;
}

NVFBC_BOOL NvFBCCudaCapture::load() {
	return loadNvFBC(this->libNvFBC);
}

NVFBC_BOOL NvFBCCudaCapture::createInstance() {
	PNVFBCCREATEINSTANCE createInstancePtr;

	auto res = nvfbcResolveCreateInstance(this->libNvFBC, createInstancePtr);

	if (res == NVFBC_FALSE) {
		return NVFBC_FALSE;
	}

	return nvfbcCreateInstance(createInstancePtr, this->instance);
}

NVFBC_BOOL NvFBCCudaCapture::createSessionHandle() {
	return nvfbcCreateSessionHandle(this->instance, this->fbcHandle);
}

NVFBC_BOOL NvFBCCudaCapture::createCaptureSession() {
	return nvfbcCreateCaptureSession(this->instance, this->fbcHandle);
}

NVFBC_BOOL NvFBCCudaCapture::setupCaptureSession() {
	return nvfbcSetupCaptureSession(this->instance, this->fbcHandle);
}

NVFBC_BOOL NvFBCCudaCapture::grabFrame(CUdeviceptr &cuDevicePtr) {
	return nvfbcGrabFrame(this->instance, this->fbcHandle, cuDevicePtr);
}

NVFBC_BOOL NvFBCCudaCapture::destroyCaptureSession() {
	return nvfbcDestroyCaptureSession(this->instance, this->fbcHandle);
}

NVFBC_BOOL NvFBCCudaCapture::destroySessionHandle() {
	return nvfbcDestroySessionHandle(this->instance, this->fbcHandle);
}