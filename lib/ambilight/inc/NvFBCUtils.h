#ifndef __NVFBC_UTILS_H
#define __NVFBC_UTILS_H

#define LIB_NVFBC_NAME "libnvidia-fbc.so.1"

#include <dlfcn.h>
#include <stdio.h>
#include <memory.h>
#include "cuda.h"
#include "NvFBC.h"

// Dynamically loads the NvFBC library
NVFBC_BOOL loadNvFBC(void* &libNvfbcPtr);

/*
 * Resolve the 'NvFBCCreateInstance' symbol that will allow us to get
 * the API function pointers.
 */
NVFBC_BOOL nvfbcResolveCreateInstance(
	void* libNvfbcPtr,
	PNVFBCCREATEINSTANCE &createInstancePtr
);

/*
 * Create an NvFBC instance.
 *
 * API function pointers are accessible through the instance.
 */
NVFBC_BOOL nvfbcCreateInstance(
	PNVFBCCREATEINSTANCE createInstancePtr,
	NVFBC_API_FUNCTION_LIST &instance
);

/*
 * Create a session handle that is used to identify the client.
 */
NVFBC_BOOL nvfbcCreateSessionHandle(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
);

/*
 * Create a capture session.
 */
NVFBC_BOOL nvfbcCreateCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
);

/*
 * Set up the capture session.
 */
NVFBC_BOOL nvfbcSetupCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
);

/*
 * Grab frame to CUDA memory buffer
*/
NVFBC_BOOL nvfbcGrabFrame(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle,
	CUdeviceptr &cuDevicePtr,
	bool waitUntilReady = false
);

/*
 * Destroy capture session, tear down resources.
 */
NVFBC_BOOL nvfbcDestroyCaptureSession(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
);

/*
 * Destroy session handle, tear down more resources.
 */
NVFBC_BOOL nvfbcDestroySessionHandle(
	NVFBC_API_FUNCTION_LIST &instance,
	NVFBC_SESSION_HANDLE &fbcHandle
);

#endif