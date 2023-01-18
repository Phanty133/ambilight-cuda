#include "FrameProcessing.cuh"

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

// TODO: Precalculate middle of the screen minX, maxX, minY, minY values
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