#include "LargestRectSolve.h"

struct StackEntry {
	int start;
	int height;
};

// Returns a matrix where every empty pixel is false and every pixel with a rect is true
std::vector<std::vector<int>> rectsToHistMatrix(int width, int height, std::vector<Rect> &rects) {
	std::vector<std::vector<int>> matrix(height, std::vector<int>(width, 1));

	for (Rect rect : rects) {
		for (int x = rect.x; x < rect.x + rect.w; x++) {
			for (int y = rect.y; y < rect.y + rect.h; y++) {
				matrix[y][x] = 0;
			}
		}
	}

	// printf("Initial matrix: \n");

	// for (int r = 0; r < height; r++) {
	// 	for (int c = 0; c < width; c++) {
	// 		printf("%i ", (int)matrix[r][c]);
	// 	}
	// 	printf("\n");
	// }

	for (int x = 0; x < width; x++) {
		for (int y = 1; y < height; y++) {
			if (matrix[y][x] == 0) continue;

			matrix[y][x] = matrix[y - 1][x] + 1;
		}
	}

	// printf("Hist matrix: \n");

	// for (int r = 0; r < height; r++) {
	// 	for (int c = 0; c < width; c++) {
	// 		printf("%i ", (int)matrix[r][c]);
	// 	}
	// 	printf("\n");
	// }

	return matrix;
}

Rect largestRectBetweenRects(int width, int height, std::vector<Rect> &rects) {
	auto histMat = rectsToHistMatrix(width, height, rects);
	Rect maxRect;

	for (int row = 0; row < height; row++) {
		Rect rowRect = largestRectUnderHistogram(histMat[row]);
		rowRect.y = row - rowRect.h + 1;

		maxRect = Rect::max(maxRect, rowRect);
	}

	return maxRect;
}

Rect largestRectUnderHistogram(std::vector<int> &hist) {
	std::stack<StackEntry> rectStack;
	// int maxSize[2] = {0, 0}; // Height, Width
	int start, height, pos;
	Rect maxRect;

	for (pos = 0; pos < hist.size(); pos++) {
		height = hist[pos];
		start = pos; // pos where rect starts

		while (true) {
			if (rectStack.empty() || height > rectStack.top().height) {
				StackEntry entry;
				entry.height = height;
				entry.start = start;

				rectStack.push(entry);
			} else if (!rectStack.empty() && height < rectStack.top().height) {
				auto top = rectStack.top();
				Rect newRect(top.start, 0, pos - top.start, top.height);

				maxRect = Rect::max(maxRect, newRect);

				start = top.start;
				rectStack.pop();

				continue;
			}

			// height == rectStack.top().height goes here
			break;
		}
	}

	// pos++; Not needed version because unlike in the python the end of the for loop will increment it

	std::stack<StackEntry> revRectStack;

	while (!rectStack.empty()) {
		revRectStack.push(rectStack.top());
		rectStack.pop();
	}

	while (!revRectStack.empty()) {
		auto top = revRectStack.top();
		Rect newRect(top.start, 0, pos - top.start, top.height);

		maxRect = Rect::max(maxRect, newRect);
		
		revRectStack.pop();
	}

	return maxRect;
}
