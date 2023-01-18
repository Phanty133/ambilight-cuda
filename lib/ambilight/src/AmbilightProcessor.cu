#include "AmbilightProcessor.cuh"

AmbilightProcessor::AmbilightProcessor(
	KernelParams kernelParams,
	Sector* sectorMap
) {
	this->params = kernelParams;
	this->sectorMap = sectorMap;

	this->numBlocks = (params.frameSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

AmbilightProcessor::~AmbilightProcessor() {
	if (this->fbcCapture != nullptr) delete this->fbcCapture;
}

bool AmbilightProcessor::initCUDA() {
	if (cudaInit(&this->cuCtx) != NVFBC_TRUE) {
		return false;
	}

	this->fbcCapture = new NvFBCCudaCapture(this->cuCtx);

	return true;
}

void AmbilightProcessor::allocMemory() {
	this->outputMemSize = sizeof(SectorData) * params.sectorCount;
	this->sectorMemSize = sizeof(Sector) * params.sectorCount;

	// Kernel params
	size_t paramsSize = sizeof(KernelParams);
	cudaMalloc(&this->cuParams, paramsSize);
	cudaMemcpy(this->cuParams, &params, paramsSize, cudaMemcpyHostToDevice);

	// Sector map
	cudaMalloc(&this->cuSectorMap, this->sectorMemSize);
	cudaMemcpy(this->cuSectorMap, this->sectorMap, this->sectorMemSize, cudaMemcpyHostToDevice);

	// Output data
	cudaMalloc(&this->cuFrameOutput, this->outputMemSize);
	this->frameOutput = (SectorData*)malloc(this->outputMemSize);
}

bool AmbilightProcessor::initCapture() {
	if (this->cuCtx == nullptr) {
		printf("Error initializing capture: CUDA not initialized\n");
		return false;
	}

	if (!this->fbcCapture->load()) {
		printf("Error initializing capture: Error loading NvFBC\n");
		return false;
	}

	if (!this->fbcCapture->createInstance()) {
		printf("Error initializing capture: Error creating NvFBC instance\n");
		return false;
	}

	if (!this->fbcCapture->createSessionHandle()) {
		printf("Error initializing capture: Error creating NvFBC session handle\n");
		return false;
	}

	if (!this->fbcCapture->createCaptureSession()) {
		printf("Error initializing capture: Error creating capture session\n");
		return false;
	}
	
	if (!this->fbcCapture->setupCaptureSession()) {
		printf("Error initializing capture: Error setting up capture session\n");
		return false;
	}

	return true;
}

// TODO: Check if the frame has changed enough by first getting a difference map
// TODO: Possibly check which parts of the screen should be updated based on the difference map?
void AmbilightProcessor::grabFrame(std::mutex* outputMutex) {
	// Init memory
	cudaMemset(cuFrameOutput, 0, this->outputMemSize);

	// Grab frame with NvFBC
	CUdeviceptr cuFrame;
	this->fbcCapture->grabFrame(cuFrame);

	// Parse the frame
	parseFrameKernel<<<numBlocks, BLOCK_SIZE>>>(
		(RGBPixel*)cuFrame,
		this->cuFrameOutput,
		this->cuParams,
		this->cuSectorMap
	);

	cudaDeviceSynchronize();

	std::unique_lock<std::mutex> outputLock;

	if (outputMutex != nullptr) {
		outputLock = std::unique_lock<std::mutex>(*outputMutex);
	}

	// Copy data from GPU to host
	cudaMemcpy(
		this->frameOutput,
		this->cuFrameOutput,
		this->outputMemSize,
		cudaMemcpyDeviceToHost
	);

	if (outputMutex != nullptr) {
		outputLock.unlock();
	}
}

void AmbilightProcessor::decodeHSV(uint64_t &encodedHsv, WideHSVPixel &hsv) {
	uint64_t dataBitmask = 0x1FFFFF; // Bitmask to retrieve only the lowest 21 bits

	hsv.h = encodedHsv & dataBitmask;
	hsv.s = (encodedHsv >> 21) & dataBitmask;
	hsv.v = encodedHsv >> 42;
}

size_t AmbilightProcessor::getFrameSize() {
	return sizeof(AveragedHSVPixel) * this->params.sectorCount;
}

void AmbilightProcessor::getFrame(AveragedHSVPixel* outData) {
	memset(outData, 0, this->getFrameSize());

	for (int i = 0; i < params.sectorCount; i++) {
		auto sector = this->sectorMap[i];
		int sectorSize = (sector.maxX - sector.minX) * (sector.maxY - sector.minY);
		uint64_t encodedHsv = this->frameOutput[i].hsvData;
		
		WideHSVPixel hsv;
		AmbilightProcessor::decodeHSV(encodedHsv, hsv);

		outData[i].h = (float)hsv.h / sectorSize;
		outData[i].s = (float)hsv.s / sectorSize;
		outData[i].v = (float)hsv.v / sectorSize;
	}
}

void AmbilightProcessor::deallocMemory() {
	if (this->cuFrameOutput != nullptr) cudaFree(this->cuFrameOutput);
	if (this->cuSectorMap != nullptr) cudaFree(this->cuSectorMap);
	if (this->cuParams != nullptr) cudaFree(this->cuParams);

	if (this->frameOutput != nullptr) free(this->frameOutput);

	if (this->fbcCapture != nullptr) {
		this->fbcCapture->destroyCaptureSession();
		this->fbcCapture->destroySessionHandle();
	}
}
