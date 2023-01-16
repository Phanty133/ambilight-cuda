#ifndef __FRAME_PROCESSING_H
#define __FRAME_PROCESSING_H

#include <cuda.h>
#include "ColorStructs.h"

#define BLOCK_SIZE 256

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

/// @brief Converts 24-bit RGB to 16-bit HSV
/// @param rgb 24-bit RGB, 0-255
/// @param hsv Output, 16-bit HSV, 0-32
/// @return 
__device__ void rgbToHSV(RGBPixel &rgb, HSVPixel &hsv);

/// @brief Gets the sector corresponding to a pixel index
/// @param pixelIndex Pixel index counting from top-left
/// @param params Kernel parameters
/// @param sectors Display sector map
/// @return Sector index if pixel is within a sector, -1 otherwise
__device__ int getSectorIndex(
	unsigned int pixelIndex,
	KernelParams* params,
	Sector* sectors
);

/// @brief Adds the pixel to the sector's color total
/// @param pixel Pixel RGB color
/// @param sectorIndex The sector the pixel corresponds to
/// @param frameOut Output data pointer
/// @return 
__device__ void parsePixel(
	RGBPixel &pixel,
	int &sectorIndex,
	SectorData* frameOut
);

/// @brief Calculates the average HSV colors in each sector in the given frame
/// @param frame CUDA buffer of a screenshot grabbed by NvFBC
/// @param frameOut Output data pointer on the GPU 
/// @param params Kernel parameter pointer on the GPU
/// @param sectors Dislay sector map pointer on the GPU
/// @return 
__global__ void parseFrameKernel(
	RGBPixel* frame,
	SectorData* frameOut,
	KernelParams* params,
	Sector* sectors
);

#endif