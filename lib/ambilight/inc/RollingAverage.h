#ifndef __ROLLING_AVERAGE_H
#define __ROLLING_AVERAGE_H

#include <queue>

class RollingAverage {
private:
	int maxItemCount = 15;

	std::queue<int> items;
	int curItemSum = 0;
public:
	RollingAverage();
	RollingAverage(int averageItemCount);

	void addItem(int item);
	float getAverage();
};

#endif