#ifndef __LARGEST_RECT_SOLVE_H
#define __LARGEST_RECT_SOLVE_H

// Exposes function for finding the largest rectangle that can fit between other rectangles
// Used to precalculate the largest part of the screen without any sectors, 
// which is then used to decrease the average number of comparisons in the frame processing kernel
// when determining the sector index of a pixel

// Based on https://gist.github.com/zed/776423

#include <stack>
#include <vector>
#include <stdio.h>

struct Rect {
	int x, y, w, h;

	Rect() {
		this->x = 0;
		this->y = 0;
		this->w = 0;
		this->h = 0;
	}

	Rect(int x, int y, int w, int h) {
		this->x = x;
		this->y = y;
		this->w = w;
		this->h = h;
	}

	int area() {
		return w * h;
	}

	static Rect max(Rect &r1, Rect &r2) {
		return r1.area() >= r2.area() ? r1 : r2;
	}
};

Rect largestRectBetweenRects(int width, int height, std::vector<Rect> &rects);
Rect largestRectUnderHistogram(std::vector<int> &hist);

#endif