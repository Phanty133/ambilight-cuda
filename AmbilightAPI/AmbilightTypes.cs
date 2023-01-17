using System;
using System.Collections.Generic;

namespace AmbilightAPI;

public struct KernelParams {
	public int frameWidth;
	public int frameHeight;
	public int sectorCount;

	public CKernelParams ToCType() {
		var cParams = new CKernelParams();

		cParams.frameWidth = (uint)frameWidth;
		cParams.frameHeight = (uint)frameHeight;
		cParams.frameSize = cParams.frameHeight * cParams.frameWidth;
		cParams.sectorCount = (uint)sectorCount;

		return cParams;
	}
}

public struct Sector {
	public uint index;
	public uint x;
	public uint y;
	public uint width;
	public uint height;

	public CSector ToCType() {
		var cSector = new CSector();

		cSector.index = index;
		cSector.minX = x;
		cSector.minY = y;
		cSector.maxX = x + width;
		cSector.maxY = y + height;

		return cSector;
	}
}

public struct SectorMap {
	public List<Sector> sectors;

	public SectorMap(List<Sector> sectors) {
		this.sectors = sectors;
	}

	public CSector[] ToCType() {
		var cSectorMap = new CSector[sectors.Count];

		for (int i = 0; i < sectors.Count; i++) {
			cSectorMap[i] = sectors[i].ToCType();
		}

		return cSectorMap;
	}
}

public struct HSVPixel {
	public float h; // 0 - 1
	public float s; // 0 - 1
	public float v; // 0 - 1

	public static HSVPixel FromCType(CAveragedHSVPixel cPixel) {
		// cPixel has each value in the range from 0-32

		var pixel = new HSVPixel();
		pixel.h = cPixel.h / 32f;
		pixel.s = cPixel.s / 32f;
		pixel.v = cPixel.v / 32f;

		return pixel;
	}
}