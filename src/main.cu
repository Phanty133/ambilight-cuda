#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <iostream>
#include "AmbilightProcessor.cuh"
#include "CudaUtils.h"

int main() {
	CUcontext cuCtx;

	if (cudaInit(&cuCtx) != NVFBC_TRUE) {
		return EXIT_FAILURE;
	}

	// Generate basic test data

	// Set kernel parameters
	KernelParams params;
	params.frameWidth = 2560;
	params.frameHeight = 1600;
	params.frameSize = params.frameWidth * params.frameHeight;
	params.sectorCount = 10;

	// Define display sectors
	auto sectors = new Sector[params.sectorCount];
	int secWidth = params.frameWidth / params.sectorCount;

	for (int i = 0; i < params.sectorCount; i++) {
		sectors[i].index = i;
		sectors[i].minX = i * secWidth;
		sectors[i].minY = 0;
		sectors[i].maxX = (i + 1) * secWidth;
		sectors[i].maxY = 100;
	}

	// Init processor
	AmbilightProcessor processor(cuCtx, params, sectors);

	processor.allocMemory();
	processor.initCapture();

	auto output = new AveragedHSVPixel[processor.getFrameSize()];

	auto t1 = std::chrono::system_clock::now();

	processor.grabFrame();
	processor.getFrame(output);

	printf("Sector: (H, S, V)");

	for (int i = 0; i < params.sectorCount; i++) {
		auto sector = output[i];
		int h = (sector.h / 32) * 360;
		int s = (sector.s / 32) * 100;
		int v = (sector.v / 32) * 100;

		printf("%i: (%ideg, %i%%, %i%%)\n", i, h, s, v);
	}

	auto t2 = std::chrono::system_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

	std::cout << "Time taken: " << elapsed.count() << "ms\n";

	// Clean-up

	processor.deallocMemory();
	delete[] output;

	return 0;
}