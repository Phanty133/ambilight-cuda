#ifndef __FRAMETIMER_H
#define __FRAMETIMER_H

#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <functional>
#include <condition_variable>
#include "AmbilightProcessor.cuh"
#include "KernelParams.h"
#include "RollingAverage.h"
#include <stdio.h>

class TAmbilightProcessor {
private:
	// Threading
	std::thread* t = nullptr;
	std::atomic<bool> grabActive{false};
	std::atomic<bool> mustKillThread{false};

	std::mutex tActiveMutex;
	std::condition_variable tActiveCondVar;
	std::mutex frameMutex;

	// Timing
	std::atomic<int> targetFrameTime{0}; // In microseconds
	int frameAveragingTime; // seconds
	RollingAverage frameTimeAvg;

	// Processor
	KernelParams kernelParams;
	Sector* sectorMap;
	AmbilightProcessor* tProcessor;

	void waitUntilActiveOrKilled();
	int timedGrabFrame(AmbilightProcessor* tProcessor);
public:
	TAmbilightProcessor(KernelParams kernelParams, Sector* sectorMap, int frameAveragingTime = 1);
	~TAmbilightProcessor();

	std::thread* createThread();
	std::thread* getThread();
	bool detachThread();

	void start(int targetFPS);
	void stop();
	void kill();

	void getFrame(AveragedHSVPixel* outData);

	bool isActive();
	int getTargetFPS();
	float getActualFPS();
};

#endif
