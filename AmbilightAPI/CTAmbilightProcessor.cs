using System;
using System.Runtime.InteropServices;

using AmbilightAPI;

public static class CTAmbilightProcessor {
	[DllImport("lib/ambilight/libambilight.so", CallingConvention=CallingConvention.Winapi)]
	static public extern IntPtr createTProcessor(CKernelParams kParams, ref CSector sectorMap);
	
	[DllImport("lib/ambilight/libambilight.so")]
	static public extern IntPtr tpCreateThread(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern IntPtr tpGetThread(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern bool tpDetachThread(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpStart(IntPtr processor, int targetFPS, bool waitUntilReady);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpStop(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpKill(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpGetFrame(IntPtr processor, ref CAveragedHSVPixel outData);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern bool tpIsActive(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern int tpGetTargetFPS(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern float tpGetActualFPS(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpOnFrameReady(IntPtr processor, AmbilightFrameReadyCallback callbackPtr);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern bool tpIsFrameReady(IntPtr processor);

	[DllImport("lib/ambilight/libambilight.so")]
	static public extern void tpClearFrameReadyFlag(IntPtr processor);
}