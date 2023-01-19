#include "NvFBCUtils.h"

NVFBC_BOOL loadNvFBC(void* &libNvfbcPtr) {
	void* libNVFBC = dlopen(LIB_NVFBC_NAME, RTLD_NOW);

	if (libNVFBC == NULL) {
		fprintf(stderr, "Unable to open '%s'\n", LIB_NVFBC_NAME);
		return NVFBC_FALSE;
	}

	libNvfbcPtr = libNVFBC;

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcResolveCreateInstance(
	void* libNvfbcPtr,
	PNVFBCCREATEINSTANCE &createInstancePtr
) {
	PNVFBCCREATEINSTANCE NvFBCCreateInstance_ptr =
		(PNVFBCCREATEINSTANCE) dlsym(libNvfbcPtr, "NvFBCCreateInstance");

	if (NvFBCCreateInstance_ptr == NULL) {
		fprintf(stderr, "Unable to resolve symbol 'NvFBCCreateInstance'\n");
		return NVFBC_FALSE;
	}

	createInstancePtr = NvFBCCreateInstance_ptr;

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcCreateInstance(
	PNVFBCCREATEINSTANCE createInstancePtr,
	NVFBC_API_FUNCTION_LIST &instance
) {
	memset(&instance, 0, sizeof(NVFBC_API_FUNCTION_LIST));

	instance.dwVersion = NVFBC_VERSION;

	NVFBCSTATUS fbcStatus = createInstancePtr(&instance);

	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "Unable to create NvFBC instance (status: %d)\n",
				fbcStatus);
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcCreateSessionHandle(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
) {
	NVFBC_CREATE_HANDLE_PARAMS createHandleParams;

	memset(&createHandleParams, 0, sizeof(createHandleParams));

	createHandleParams.dwVersion = NVFBC_CREATE_HANDLE_PARAMS_VER;

	NVFBCSTATUS fbcStatus = instance.nvFBCCreateHandle(&fbcHandle, &createHandleParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcCreateCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
) {
	NVFBC_CREATE_CAPTURE_SESSION_PARAMS createCaptureParams;
	NVFBC_SIZE frameSize = { 0, 0 };

	memset(&createCaptureParams, 0, sizeof(createCaptureParams));

	createCaptureParams.dwVersion     = NVFBC_CREATE_CAPTURE_SESSION_PARAMS_VER;
	createCaptureParams.eCaptureType  = NVFBC_CAPTURE_SHARED_CUDA;
	createCaptureParams.bWithCursor   = NVFBC_FALSE;
	createCaptureParams.frameSize     = frameSize;
	createCaptureParams.eTrackingType = NVFBC_TRACKING_DEFAULT;

	NVFBCSTATUS fbcStatus = instance.nvFBCCreateCaptureSession(fbcHandle, &createCaptureParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcSetupCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
) {
	NVFBC_TOCUDA_SETUP_PARAMS setupParams;

	memset(&setupParams, 0, sizeof(setupParams));

	setupParams.dwVersion     = NVFBC_TOCUDA_SETUP_PARAMS_VER;
	setupParams.eBufferFormat = NVFBC_BUFFER_FORMAT_RGB;

	NVFBCSTATUS fbcStatus = instance.nvFBCToCudaSetUp(fbcHandle, &setupParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcDestroyCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
) {
	NVFBC_DESTROY_CAPTURE_SESSION_PARAMS destroyCaptureParams;

	memset(&destroyCaptureParams, 0, sizeof(destroyCaptureParams));

	destroyCaptureParams.dwVersion = NVFBC_DESTROY_CAPTURE_SESSION_PARAMS_VER;

	NVFBCSTATUS fbcStatus = instance.nvFBCDestroyCaptureSession(fbcHandle, &destroyCaptureParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcDestroySessionHandle(NVFBC_API_FUNCTION_LIST &instance, NVFBC_SESSION_HANDLE &fbcHandle) {
	NVFBC_DESTROY_HANDLE_PARAMS destroyHandleParams;

	memset(&destroyHandleParams, 0, sizeof(destroyHandleParams));

	destroyHandleParams.dwVersion = NVFBC_DESTROY_HANDLE_PARAMS_VER;

	NVFBCSTATUS fbcStatus = instance.nvFBCDestroyHandle(fbcHandle, &destroyHandleParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}

NVFBC_BOOL nvfbcGrabFrame(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle,
	CUdeviceptr &cuDevicePtr,
	bool waitUntilReady
) {
	NVFBC_TOCUDA_GRAB_FRAME_PARAMS grabParams;
	NVFBC_FRAME_GRAB_INFO frameInfo;

	memset(&grabParams, 0, sizeof(grabParams));
	memset(&frameInfo, 0, sizeof(frameInfo));

	grabParams.dwVersion = NVFBC_TOCUDA_GRAB_FRAME_PARAMS_VER;

	grabParams.dwFlags = 
		waitUntilReady
		? NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT_IF_NEW_FRAME_READY
		: NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT;

	// grabParams.dwFlags = NVFBC_TOCUDA_GRAB_FLAGS_NOWAIT_IF_NEW_FRAME_READY;

	grabParams.pFrameGrabInfo = &frameInfo;
	grabParams.pCUDADeviceBuffer = &cuDevicePtr;

	NVFBCSTATUS fbcStatus = instance.nvFBCToCudaGrabFrame(fbcHandle, &grabParams);
	if (fbcStatus != NVFBC_SUCCESS) {
		fprintf(stderr, "%s\n", instance.nvFBCGetLastErrorStr(fbcHandle));
		return NVFBC_FALSE;
	}

	return NVFBC_TRUE;
}
