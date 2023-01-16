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