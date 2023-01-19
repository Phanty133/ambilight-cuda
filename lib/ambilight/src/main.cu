#include <stdio.h>
#include <cuda.h>
#include <thread>
#include <functional>
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
	
	void pSetCaptureReadyMode(AmbilightProcessor* p, bool waitUntilReady) {
		p->setCaptureReadyMode(waitUntilReady);
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

	void tpStart(TAmbilightProcessor* p, int targetFPS, bool waitUntilReady) {
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

	void tpOnFrameReady(TAmbilightProcessor* p, std::function<void()> cb) {
		return p->onFrameReady(cb);
	}

	bool tpIsFrameReady(TAmbilightProcessor* p) {
		return p->isFrameReady();
	}

	void tpClearFrameReadyFlag(TAmbilightProcessor* p) {
		p->clearFrameReadyFlag();
	}
}

int main() {
	// Generate basic test data

	// Set kernel parameters
	// KernelParams params;
	// params.frameWidth = 2560;
	// params.frameHeight = 1600;
	// params.frameSize = params.frameWidth * params.frameHeight;
	// params.sectorCount = 40;

	// // Define display sectors
	// int xSectors = 10;
	// int ySectors = 10;

	// auto sectors = new Sector[params.sectorCount];
	// int secWidth = params.frameWidth / xSectors;
	// int secHeight = (params.frameHeight - 100 * 2) / ySectors;

	// // Top
	// for (int i = 0; i < xSectors; i++) {
	// 	int sI = i;
	// 	sectors[sI].index = sI;
	// 	sectors[sI].minX = i * secWidth;
	// 	sectors[sI].minY = 0;
	// 	sectors[sI].maxX = (i + 1) * secWidth;
	// 	sectors[sI].maxY = 100;
	// }

	// // Bottom
	// for (int i = 0; i < xSectors; i++) {
	// 	int sI = i + 20;
	// 	sectors[sI].index = sI;
	// 	sectors[sI].minX = i * secWidth;
	// 	sectors[sI].minY = params.frameHeight - 100;
	// 	sectors[sI].maxX = (i + 1) * secWidth;
	// 	sectors[sI].maxY = params.frameHeight;
	// }

	// // Left
	// for (int i = 0; i < ySectors; i++) {
	// 	int sI = i + 30;
	// 	sectors[sI].index = sI;
	// 	sectors[sI].minX = 0;
	// 	sectors[sI].minY = 100 + i * secHeight;
	// 	sectors[sI].maxX = 100;
	// 	sectors[sI].maxY = 100 + (i + 1) * secHeight;
	// }

	// // Right
	// for (int i = 0; i < ySectors; i++) {
	// 	int sI = i + 10;
	// 	sectors[sI].index = sI;
	// 	sectors[sI].minX = 2500;
	// 	sectors[sI].minY = 100 + (i + 1) * secHeight;
	// 	sectors[sI].maxX = 2600;
	// 	sectors[sI].maxY = i * secHeight;
	// }

	// // // Init processor

	// TAmbilightProcessor tProcessor(params, sectors);

	// std::thread otherT([&tProcessor, &params](){
	// 	std::this_thread::sleep_for(std::chrono::milliseconds(3000));

	// 	tProcessor.start(200, true);
	// 	auto output = new AveragedHSVPixel[params.sectorCount];

	// 	int frames = 0;

	// 	tProcessor.onFrameReady([&tProcessor, &output, &frames](){
	// 		tProcessor.getFrame(output);
	// 		frames++;

	// 		// printf("Sector: (H, S, V)\n");

	// 		// for (int i = 0; i < params.sectorCount; i++) {
	// 		// 	auto sector = output[i];
	// 		// 	int h = (sector.h / 32) * 360;
	// 		// 	int s = (sector.s / 32) * 100;
	// 		// 	int v = (sector.v / 32) * 100;

	// 		// 	printf("%i: (%ideg, %i%%, %i%%)\n", i, h, s, v);
	// 		// }
	// 	});

	// 	int n = 0;

	// 	while (n++ < 30) {
	// 		printf("%f FPS, counted: %i\n", tProcessor.getActualFPS(), frames);
	// 		frames = 0;
	// 		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	// 	}

	// 	delete[] output;

	// 	std::this_thread::sleep_for(std::chrono::milliseconds(3000));

	// 	tProcessor.stop();

	// 	std::this_thread::sleep_for(std::chrono::milliseconds(1000));

	// 	tProcessor.kill();
	// });

	// tProcessor.createThread()->join();
	// otherT.join();

	return 0;
}