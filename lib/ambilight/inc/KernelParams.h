#ifndef __KERNEL_PARAMS_H
#define __KERNEL_PARAMS_H

#include "LargestRectSolve.h"

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

	Rect toRect() {
		return Rect(minX, minY, maxX - minX, maxY - minY);
	}

	static Sector fromRect(Rect r, int index = 0) {
		Sector s;
		s.index = index;
		s.minX = r.x;
		s.minY = r.y;
		s.maxX = r.x + r.w;
		s.maxY = r.y + r.h;

		return s;
	}
};

struct SectorData {
	// 16 bit HSV color sums encoded in a single number
	// to reduce the number of atomic adds required
	// H - bits 1-21
	// S - bits 22-42
	// V - bits 43-63
	unsigned long long hsvData;
};

#endif