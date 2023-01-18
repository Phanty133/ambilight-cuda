#include "RollingAverage.h"

RollingAverage::RollingAverage() {}

RollingAverage::RollingAverage(int avgCount) {
	this->maxItemCount = avgCount;
}

void RollingAverage::addItem(int item) {
	if (this->items.size() == this->maxItemCount) {
		this->curItemSum -= this->items.front();
		this->items.pop();
	}

	this->items.push(item);
	this->curItemSum += item;
}

float RollingAverage::getAverage() {
	if (this->items.size() == 0) return 0;

	return (float)this->curItemSum / this->items.size();
}