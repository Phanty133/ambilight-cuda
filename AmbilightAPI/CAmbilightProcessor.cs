using System;
using System.Runtime.InteropServices;

namespace AmbilightAPI;

public struct CKernelParams {
	public uint frameSize;
	public uint frameWidth;
	public uint frameHeight;
	public uint sectorCount;
}

public struct CSector {
	public uint index;
	public uint minX;
	public uint minY;
	public uint maxX;
	public uint maxY;
}

public struct CAveragedHSVPixel {
	public float h;
	public float s;
	public float v;
}

public static class CAmbilightProcessor {
	[DllImport("lib/ambilight/libambilight.so", CallingConvention=CallingConvention.Winapi)]
	static public extern IntPtr createProcessor(CKernelParams kParams, ref CSector sectorMap);
	
	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void destroyProcessor(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern bool pInitCUDA(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pAllocMemory(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pDeallocMemory(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pInitCapture(IntPtr processor);
	
	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pGrabFrame(IntPtr processor);
	
	[DllImport("lib/ambilight/libambilight.so")]
	static public extern int pGetFrameSize(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pGetFrame(IntPtr processor, ref CAveragedHSVPixel outData);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pStartContinuousGrabbing(IntPtr processor, int targetFPS);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pStopContinuousGrabbing(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void pSetTargetFPS(IntPtr processor, int targetFPS);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern int pGetTargetFPS(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern float pGetActualFPS(IntPtr processor);
}