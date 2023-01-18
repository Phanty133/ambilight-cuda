#include "TAmbilightProcessor.cuh"

// int max(int a, int b) {
// 	return a > b ? a : b;
// }

TAmbilightProcessor::TAmbilightProcessor(
	KernelParams kernelParams,
	Sector* sectorMap,
	int frameAveragingTime // Seconds
) {
	this->kernelParams = kernelParams;
	this->sectorMap = sectorMap;
	this->frameAveragingTime = frameAveragingTime;
}

std::thread* TAmbilightProcessor::createThread() {
	// Create a new thread for the ambilight processor
	return new std::thread([this](){
		printf("Ambilight processor thread created!\n");

		// Initialize processor
		tProcessor = new AmbilightProcessor(this->kernelParams, this->sectorMap);

		tProcessor->initCUDA();
		tProcessor->allocMemory();
		tProcessor->initCapture();

		// Action loop
		while (!this->mustKillThread) {
			// Wait until the processor should be active
			this->waitUntilActiveOrKilled();

			// Grab frames while active
			while (this->grabActive) {
				int execTimeUs = this->timedGrabFrame(tProcessor);
				int delayTimeUs = max(0, this->targetFrameTime - execTimeUs);

				printf(""); // For some inexplicable reason required to print stuff from the main thread
				// printf("Exec: %fms; Delay: %fms\n", execTimeUs / 1000.0f, delayTimeUs / 1000.0f);

				this->frameTimeAvg.addItem(max(execTimeUs, this->targetFrameTime));

				std::this_thread::sleep_for(std::chrono::microseconds(delayTimeUs));
			}
		}

		// Deallocate the processor when the thread is removed
		tProcessor->deallocMemory();

		printf("Ambilight processor thread killed!\n");
	});
}

TAmbilightProcessor::~TAmbilightProcessor() {
	if (this->t != nullptr) delete this->t;
}

std::thread* TAmbilightProcessor::getThread() {
	return this->t;
}

bool TAmbilightProcessor::detachThread() {
	if (!this->t->joinable()) return false;

	this->t->detach();

	return false;
}

void TAmbilightProcessor::waitUntilActiveOrKilled() {
	std::unique_lock<std::mutex> lock(this->tActiveMutex);
	this->tActiveCondVar.wait(lock, [this](){
		return this->grabActive.load() || this->mustKillThread.load();
	});
}

int TAmbilightProcessor::timedGrabFrame(AmbilightProcessor* tProcessor) {
	auto t0 = std::chrono::system_clock::now();
	tProcessor->grabFrame(&this->frameMutex);
	auto t1 = std::chrono::system_clock::now();
	
	return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

void TAmbilightProcessor::start(int targetFPS) {
	this->targetFrameTime = 1000000.0f / targetFPS;
	this->frameTimeAvg = RollingAverage(this->frameAveragingTime * targetFPS);

	this->grabActive = true;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::stop() {
	this->grabActive = false;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::kill() {
	this->mustKillThread = true;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::getFrame(AveragedHSVPixel* output) {
	auto outputLock = std::unique_lock<std::mutex>(this->frameMutex);

	this->tProcessor->getFrame(output);

	outputLock.unlock();
}

int TAmbilightProcessor::getTargetFPS() {
	return 1000000.0f / this->targetFrameTime;
}

bool TAmbilightProcessor::isActive() {
	return this->grabActive && !this->mustKillThread;
}

float TAmbilightProcessor::getActualFPS() {
	return 1000000.0f / this->frameTimeAvg.getAverage();
}