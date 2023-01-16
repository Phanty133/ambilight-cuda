#include "AmbilightProcessor.cuh"

AmbilightProcessor::AmbilightProcessor(
	CUcontext cuCtx,
	KernelParams kernelParams,
	Sector* sectorMap
) {
	this->cuCtx = cuCtx;
	this->fbcCapture = new NvFBCCudaCapture(cuCtx);

	this->params = kernelParams;
	this->sectorMap = sectorMap;

	this->numBlocks = (params.frameSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

AmbilightProcessor::~AmbilightProcessor() {
	delete this->fbcCapture;
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

void AmbilightProcessor::initCapture() {
	this->fbcCapture->load();
	this->fbcCapture->createInstance();
	this->fbcCapture->createSessionHandle();
	this->fbcCapture->createCaptureSession();
	this->fbcCapture->setupCaptureSession();
}

void AmbilightProcessor::grabFrame() {
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

	// Copy data from GPU to host
	cudaMemcpy(
		this->frameOutput,
		this->cuFrameOutput,
		this->outputMemSize,
		cudaMemcpyDeviceToHost
	);
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
	cudaFree(this->cuFrameOutput);
	cudaFree(this->cuSectorMap);
	cudaFree(this->cuParams);

	free(this->frameOutput);

	this->fbcCapture->destroyCaptureSession();
	this->fbcCapture->destroySessionHandle();
}
