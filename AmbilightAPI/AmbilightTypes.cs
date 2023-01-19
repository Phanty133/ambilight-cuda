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

public struct RGBPixel {
	public float r; // 0 - 1
	public float g; // 0 - 1
	public float b; // 0 - 1

	public static RGBPixel FromHSVPixel(HSVPixel p) {
		float c = p.v * p.s;
		float x = c * (1 - Math.Abs((p.h * 6) % 2 - 1));
		float m = p.v - c;

		float r = 0;
		float g = 0;
		float b = 0;

		if (p.h < 0.167f) {
			r = c;
			g = x;
		} else if (p.h < 0.333f) {
			r = x;
			g = c;
		} else if (p.h < 0.5f) {
			g = c;
			b = x;
		} else if (p.h < 0.667f) {
			g = x;
			b = c;
		} else if (p.h < 0.833f) {
			r = x;
			b = c;
		} else {
			r = c;
			b = x;
		}

		RGBPixel rgbP = new RGBPixel();
		rgbP.r = r + m;
		rgbP.g = g + m;
		rgbP.b = b + m;

		return rgbP;
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

public delegate void AmbilightFrameReadyCallback();