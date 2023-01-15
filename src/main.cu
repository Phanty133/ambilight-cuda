#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include <stdint.h>
#include <bitset>
#include "CudaUtils.h"
#include "NvFBCCudaCapture.h"

#define BLOCK_SIZE 256

struct RGBPixel {
	uint8_t r;
	uint8_t g;
	uint8_t b;
};

struct HSVPixel {
	uint8_t h;
	uint8_t s;
	uint8_t v;
};

struct KernelParams {
	unsigned int frameSize;
	unsigned int frameWidth;
	unsigned int frameHeight;
	unsigned int sectorCount;
};

struct Sector {
	unsigned int index;
	unsigned int minX;
	unsigned int minY;
	unsigned int maxX;
	unsigned int maxY;
};

struct SectorData {
	// 16 bit HSV color sums encoded in a single number
	// to reduce the number of atomic adds required
	// H - bits 1-21
	// S - bits 22-42
	// V - bits 43-63
	unsigned long long hsvData;
};

__device__ void rgbToHSV(RGBPixel &rgb, HSVPixel &hsv) {
	float r = rgb.r * 0.003921569f;
	float g = rgb.g * 0.003921569f;
	float b = rgb.b * 0.003921569f;

	float h;
	float s;
	float v = fmaxf(r, fmaxf(g, b));
	
	float delta = v - fminf(r, fminf(g, b));

	if (v == 0.0f) {
		s = 0.0f;
	} else {
		s = delta / v;
	}

	if (delta == 0.0f) {
		h = 0.0f;
	} else if (v == r) {
		h = (g - b) / delta;
	} else if (v == g) {
		h = (b - r) / delta + 2;
	} else {
		h = (r - g) / delta + 4;
	}

	h *= 0.166667f;
	if (h < 0.0f) h += 1.0f;

	// HSV in 16 bit color
	hsv.h = (uint8_t)(h * 32.0f);
	hsv.s = (uint8_t)(s * 32.0f);
	hsv.v = (uint8_t)(v * 32.0f);
}

__device__ int getSectorIndex(
	int pixelIndex,
	KernelParams* params,
	Sector* sectors
) {
	unsigned int x = pixelIndex % params->frameWidth;
	unsigned int y = (pixelIndex - x) / params->frameWidth;

	for (int i = params->sectorCount - 1; i >= 0; i--) {
		Sector sector = sectors[i];

		if (
			x >= sector.minX
			&& x < sector.maxX
			&& y >= sector.minY
			&& y < sector.maxY
		) {
			return i;
		}
	}

	return -1;
}

__device__ void parsePixel(RGBPixel &pixel, int &sectorIndex, SectorData* frameOut) {
	HSVPixel color;
	rgbToHSV(pixel, color);

	unsigned long long hsvData = color.h + ((uint64_t)color.s << 21) + ((uint64_t)color.v << 42);

	SectorData* sector = frameOut + sectorIndex;
	atomicAdd(&(sector->hsvData), hsvData);
}

__global__ void parseFrameKernel(
	RGBPixel* frame,
	SectorData* frameOut,
	KernelParams* params,
	Sector* sectors
) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < params->frameSize; i += stride) {
		int sectorIndex = getSectorIndex(i, params, sectors);

		if (sectorIndex == -1) continue;

		parsePixel(frame[i], sectorIndex, frameOut);
	}
}

int main() {
	CUcontext cuCtx;

	if (cudaInit(&cuCtx) != NVFBC_TRUE) {
		return EXIT_FAILURE;
	}

	KernelParams params;
	params.frameWidth = 1920;
	params.frameHeight = 1080;
	params.frameSize = params.frameWidth * params.frameHeight;
	params.sectorCount = 40;

	auto sectors = new Sector[params.sectorCount];

	// Generate basic test sectors

	int secWidth = 2560 / params.sectorCount;

	for (int i = 0; i < params.sectorCount; i++) {
		sectors[i].index = i;
		sectors[i].minX = i * secWidth;
		sectors[i].minY = 0;
		sectors[i].maxX = (i + 1) * secWidth;
		sectors[i].maxY = 100;
	}

	KernelParams* cuParams;
	size_t paramsSize = sizeof(KernelParams);
	cudaMalloc(&cuParams, paramsSize);
	cudaMemcpy(cuParams, &params, paramsSize, cudaMemcpyHostToDevice);

	Sector* cuSectorMap;
	size_t sectorMapSize = sizeof(Sector) * params.sectorCount; 
	cudaMalloc(&cuSectorMap, sectorMapSize);
	cudaMemcpy(cuSectorMap, sectors, sectorMapSize, cudaMemcpyHostToDevice);

	NvFBCCudaCapture fbcCapture(cuCtx);

	fbcCapture.load();
	fbcCapture.createInstance();
	fbcCapture.createSessionHandle();
	fbcCapture.createCaptureSession();
	fbcCapture.setupCaptureSession();

	auto t1 = std::chrono::system_clock::now();

	CUdeviceptr cuFrame;
	fbcCapture.grabFrame(cuFrame);

	SectorData* cuFrameOutput;
	size_t outputSize = sizeof(SectorData) * params.sectorCount;
	cudaMalloc(&cuFrameOutput, outputSize);
	cudaMemset(cuFrameOutput, 0, outputSize);

	int numBlocks = (params.frameSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
	parseFrameKernel<<<numBlocks, BLOCK_SIZE>>>(
		(RGBPixel*)cuFrame,
		cuFrameOutput,
		cuParams,
		cuSectorMap
	);

	cudaDeviceSynchronize();

	auto frameOut = (SectorData*)malloc(outputSize);

	cudaMemcpy(frameOut, cuFrameOutput, outputSize, cudaMemcpyDeviceToHost);

	for (int i = 0; i < params.sectorCount; i++) {
		auto sector = sectors[i];
		int sectorSize = (sector.maxX - sector.minX) * (sector.maxY - sector.minY);
		auto colorData = (frameOut + i)->hsvData;

		uint64_t dataBitmask = 0x1FFFFF; // Bitmask to retrieve only the lowest 21 bits

		uint32_t h = colorData & dataBitmask;
		uint32_t s = (colorData >> 21) & dataBitmask;
		uint32_t v = colorData >> 42;

		float hAvg = (float)h / sectorSize;
		float sAvg = (float)s / sectorSize;
		float vAvg = (float)v / sectorSize;

		printf("%i: (%f, %f, %f)\n", i, hAvg, sAvg, vAvg);
	}

	auto t2 = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);

	std::cout << elapsed.count() << "us\n";

	cudaFree(cuFrameOutput);
	cudaFree(cuSectorMap);
	cudaFree(cuParams);

	free(frameOut);

	fbcCapture.destroyCaptureSession();
	fbcCapture.destroySessionHandle();

	return 0;
}