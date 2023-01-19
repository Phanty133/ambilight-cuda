#ifndef __AMBILIGHT_PROCESSOR_H
#define __AMBILIGHT_PROCESSOR_H

#include <stdint.h>
#include <math.h>
#include <cuda.h>
#include <chrono>
#include <mutex>
#include <vector>
#include "FrameProcessing.cuh"
#include "NvFBCCudaCapture.h"
#include "CudaUtils.h"
#include "LargestRectSolve.h"

struct FrameData {
	unsigned int sectorCount;
	HSVPixel* sectors;
};

class AmbilightProcessor{
private:
	// Config
	KernelParams params;
	Sector* sectorMap;
	Sector largestEmptySector;

	size_t outputMemSize;
	size_t sectorMemSize;

	// CUDA
	CUcontext cuCtx = nullptr;
	NvFBCCudaCapture* fbcCapture = nullptr;
	unsigned int numBlocks;

	KernelParams* cuParams = nullptr;
	Sector* cuSectorMap = nullptr;
	SectorData* cuFrameOutput = nullptr;
	Sector* cuLargestEmptySector = nullptr;

	// Host memory
	SectorData* frameOutput = nullptr;

	static void decodeHSV(uint64_t &encodedHsv, WideHSVPixel &hsv);
public:
	AmbilightProcessor(KernelParams kernelParams, Sector* sectorMap);
	~AmbilightProcessor();

	bool initCUDA();
	void allocMemory();
	void deallocMemory();
	bool initCapture();

	void grabFrame(std::mutex* outputMutex = nullptr);
	size_t getFrameSize();
	void getFrame(AveragedHSVPixel* outData);
};

#endif