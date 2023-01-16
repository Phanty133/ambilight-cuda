#ifndef __COLOR_STRUCTS_H
#define __COLOR_STRUCTS_H

#include <stdint.h>

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

struct WideHSVPixel {
	uint32_t h;
	uint32_t s;
	uint32_t v;
};

struct AveragedHSVPixel {
	float h;
	float s;
	float v;
};

#endif