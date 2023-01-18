#include <stdio.h>
#include <cuda.h>
#include <thread>
#include "AmbilightProcessor.cuh"
#include "TAmbilightProcessor.cuh"
#include "KernelParams.h"

extern "C" {
	/*
		Synchronous processor functions
	*/

	AmbilightProcessor* createProcessor(KernelParams params, Sector* sectorMap) {
		return new AmbilightProcessor(params, sectorMap);
	}

	void destroyProcessor(AmbilightProcessor* p) {
		if (p != NULL) {
			delete p;
			p = NULL;
		}
	}

	bool pInitCUDA(AmbilightProcessor* p) {
		return p->initCUDA();
	}

	void pAllocMemory(AmbilightProcessor* p) {
		p->allocMemory();
	}

	void pDeallocMemory(AmbilightProcessor* p) {
		p->deallocMemory();
	}

	void pInitCapture(AmbilightProcessor* p) {
		p->initCapture();
	}

	void pGrabFrame(AmbilightProcessor* p) {
		p->grabFrame();
	}

	size_t pGetFrameSize(AmbilightProcessor* p) {
		return p->getFrameSize();
	}

	void pGetFrame(AmbilightProcessor* p, AveragedHSVPixel* outData) {
		p->getFrame(outData);
	}

	/*
		Threaded processor functions
	*/

	TAmbilightProcessor* createTProcessor(KernelParams params, Sector* sectorMap) {
		return new TAmbilightProcessor(params, sectorMap);
	}

	std::thread* tpCreateThread(TAmbilightProcessor* p) {
		return p->createThread();
	}

	std::thread* tpGetThread(TAmbilightProcessor* p) {
		return p->getThread();
	}

	bool tpDetachThread(TAmbilightProcessor* p) {
		return p->detachThread();
	}

	void tpStart(TAmbilightProcessor* p, int targetFPS) {
		p->start(targetFPS);
	}

	void tpStop(TAmbilightProcessor* p) {
		p->stop();
	}

	void tpKill(TAmbilightProcessor* p) {
		p->kill();
	}

	void tpGetFrame(TAmbilightProcessor* p, AveragedHSVPixel* outData) {
		p->getFrame(outData);
	}

	bool tpIsActive(TAmbilightProcessor* p) {
		return p->isActive();
	}

	int tpGetTargetFPS(TAmbilightProcessor* p) {
		return p->getTargetFPS();
	}

	float tpGetActualFPS(TAmbilightProcessor* p) {
		return p->getActualFPS();
	}
}

#include <chrono>
#include <iostream>

int main() {
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

	TAmbilightProcessor tProcessor(params, sectors);

	std::thread otherT([&tProcessor, &params](){
		std::this_thread::sleep_for(std::chrono::milliseconds(3000));

		tProcessor.start(60);
		auto output = new AveragedHSVPixel[params.sectorCount];

		int n = 0;

		while (n++ < 30) {
			printf("%f FPS\n", tProcessor.getActualFPS());
			// tProcessor.getFrame(output);

			// printf("Sector: (H, S, V)\n");

			// for (int i = 0; i < params.sectorCount; i++) {
			// 	auto sector = output[i];
			// 	int h = (sector.h / 32) * 360;
			// 	int s = (sector.s / 32) * 100;
			// 	int v = (sector.v / 32) * 100;

			// 	printf("%i: (%ideg, %i%%, %i%%)\n", i, h, s, v);
			// }

			std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		}

		delete[] output;

		std::this_thread::sleep_for(std::chrono::milliseconds(3000));

		tProcessor.stop();

		std::this_thread::sleep_for(std::chrono::milliseconds(1000));

		tProcessor.kill();
	});

	tProcessor.createThread()->join();
	otherT.join();

	// processor.getFrame(output);

	// printf("Sector: (H, S, V)");

	// for (int i = 0; i < params.sectorCount; i++) {
	// 	auto sector = output[i];
	// 	int h = (sector.h / 32) * 360;
	// 	int s = (sector.s / 32) * 100;
	// 	int v = (sector.v / 32) * 100;

	// 	printf("%i: (%ideg, %i%%, %i%%)\n", i, h, s, v);
	// }

	// auto t2 = std::chrono::system_clock::now();
	// auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);

	// std::cout << "Time taken: " << elapsed.count() << "ms\n";

	// Clean-up

	// processor.deallocMemory();
	// delete[] output;

	return 0;
}