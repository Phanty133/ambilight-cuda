#ifndef __AMBILIGHT_PROCESSOR_H
#define __AMBILIGHT_PROCESSOR_H

#include <stdint.h>
#include <cuda.h>
#include "FrameProcessing.cuh"
#include "NvFBCCudaCapture.h"

struct FrameData {
	unsigned int sectorCount;
	HSVPixel* sectors;
};

class AmbilightProcessor{
private:
	// Config
	KernelParams params;
	Sector* sectorMap;

	size_t outputMemSize;
	size_t sectorMemSize;

	// CUDA
	CUcontext cuCtx;
	NvFBCCudaCapture* fbcCapture;
	unsigned int numBlocks;

	KernelParams* cuParams;
	Sector* cuSectorMap;
	SectorData* cuFrameOutput;

	// Host memory
	SectorData* frameOutput;

	static void decodeHSV(uint64_t &encodedHsv, WideHSVPixel &hsv);
public:
	AmbilightProcessor(CUcontext cuCtx, KernelParams kernelParams, Sector* sectorMap);
	~AmbilightProcessor();

	void allocMemory();
	void deallocMemory();
	void initCapture();

	void grabFrame();
	size_t getFrameSize();
	void getFrame(AveragedHSVPixel* outData);
};

#endif