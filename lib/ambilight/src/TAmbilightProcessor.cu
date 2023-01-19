#include "TAmbilightProcessor.cuh"

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
	this->t = new std::thread([this](){
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
			tProcessor->setCaptureReadyMode(this->waitUntilReady);

			// Grab frames while active
			while (this->grabActive) {
				int execTimeUs = this->timedGrabFrame(tProcessor);
				int delayTimeUs = max(0, this->targetFrameTime - execTimeUs);

				printf(""); // For some inexplicable reason required to print stuff from the main thread
				// printf("Exec: %fms; Delay: %fms\n", execTimeUs / 1000.0f, delayTimeUs / 1000.0f);

				this->frameTimeAvg.addItem(max(execTimeUs, this->targetFrameTime));

				if (!this->frameReady) {
					this->frameReady = true;
					this->execFrameCallbacks();
				}

				std::this_thread::sleep_for(std::chrono::microseconds(delayTimeUs));
			}
		}

		// Deallocate the processor when the thread is removed
		tProcessor->deallocMemory();

		printf("Ambilight processor thread killed!\n");
	});

	return this->t;
}

TAmbilightProcessor::~TAmbilightProcessor() {
	if (this->t != nullptr) {
		this->kill();
		delete this->t;
	}
}

std::thread* TAmbilightProcessor::getThread() {
	return this->t;
}

bool TAmbilightProcessor::detachThread() {
	if (this->t == nullptr) return false;
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

void TAmbilightProcessor::start(int targetFPS, bool waitUntilReady) {
	this->targetFrameTime = targetFPS == 0 ? 0 : (1000000.0f / targetFPS);
	this->waitUntilReady = waitUntilReady;

	// TODO: Improve rolling average frame counting when waitUntilReady=true
	int rollingAverageFrameCount = 15;

	if (!waitUntilReady) {
		rollingAverageFrameCount = this->frameAveragingTime * targetFPS;
	}

	this->frameTimeAvg = RollingAverage(rollingAverageFrameCount);

	this->grabActive = true;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::stop() {
	this->grabActive = false;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::kill() {
	if (this->t == nullptr) return;

	this->mustKillThread = true;
	this->tActiveCondVar.notify_all();
}

void TAmbilightProcessor::getFrame(AveragedHSVPixel* output) {
	auto outputLock = std::unique_lock<std::mutex>(this->frameMutex);

	this->tProcessor->getFrame(output);
	this->frameReady = false;

	outputLock.unlock();
}

int TAmbilightProcessor::getTargetFPS() {
	return 1000000.0f / this->targetFrameTime;
}

bool TAmbilightProcessor::isActive() {
	return this->grabActive && !this->mustKillThread;
}

float TAmbilightProcessor::getActualFPS() {
	float avg =this->frameTimeAvg.getAverage();

	if (avg == 0) return 0;

	return 1000000.0f / avg;
}

void TAmbilightProcessor::onFrameReady(std::function<void()> cb) {
	this->frameCallbacks.push_back(cb);
}

void TAmbilightProcessor::execFrameCallbacks() {
	for (auto cb : this->frameCallbacks) {
		cb();
	}
}

bool TAmbilightProcessor::isFrameReady() {
	return this->frameReady;
}

void TAmbilightProcessor::clearFrameReadyFlag() {
	this->frameReady = false;
}
