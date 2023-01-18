using System;
using System.Collections;
using System.Collections.Generic;

namespace AmbilightAPI;

public class AmbilightProcessor {
	public KernelParams kParams;

	public SectorMap sectorMap;

	private IntPtr? cProcessor = null;

	private bool captureInitialized = false;

	private CKernelParams cParams;

	private CSector[] cSectorMap;

	public bool CaptureInitialized {
		get {
			return captureInitialized;
		}

		private set {
			captureInitialized = value;
		}
	}

	public AmbilightProcessor(KernelParams kParams, SectorMap sectorMap) {
		this.kParams = kParams;
		this.sectorMap = sectorMap;

		cParams = kParams.ToCType();
		cSectorMap = sectorMap.ToCType();

		cProcessor = CAmbilightProcessor.createProcessor(cParams, ref cSectorMap[0]);

		CAmbilightProcessor.pInitCUDA(cProcessor.Value);
		CAmbilightProcessor.pAllocMemory(cProcessor.Value);
	}

	~AmbilightProcessor() {
		if (!cProcessor.HasValue) return;

		DestroyProcessor();
	}

	public void DestroyProcessor() {
		if (!cProcessor.HasValue) return;

		CAmbilightProcessor.pDeallocMemory(cProcessor.Value);
		CAmbilightProcessor.destroyProcessor(cProcessor.Value);

		cProcessor = null;
	}

	public void InitCapture() {
		// TODO: Add warning when initializing with null processor
		if (cProcessor == null) return;
		if (CaptureInitialized) return;

		CAmbilightProcessor.pInitCapture(cProcessor.Value);

		captureInitialized = true;
	}

	// Grabs and gets the frame
	public List<HSVPixel>? FetchSingleFrame() {
		// TODO: Add warning when initializing with null processor
		if (cProcessor == null) return null;
		if (!CaptureInitialized) return null;

		CAveragedHSVPixel[] cOutput = new CAveragedHSVPixel[kParams.sectorCount];

		CAmbilightProcessor.pGrabFrame(cProcessor.Value);
		CAmbilightProcessor.pGetFrame(cProcessor.Value, ref cOutput[0]);

		var output = new List<HSVPixel>();

		foreach (CAveragedHSVPixel p in cOutput) {
			output.Add(HSVPixel.FromCType(p));
		}

		return output;
	}

	// Only gets the frame in memory
	public List<HSVPixel>? ReadFrame() {
		// TODO: Add warning when initializing with null processor
		if (cProcessor == null) return null;
		if (!CaptureInitialized) return null;

		CAveragedHSVPixel[] cOutput = new CAveragedHSVPixel[kParams.sectorCount];

		CAmbilightProcessor.pGetFrame(cProcessor.Value, ref cOutput[0]);

		var output = new List<HSVPixel>();

		foreach (CAveragedHSVPixel p in cOutput) {
			output.Add(HSVPixel.FromCType(p));
		}

		return output;
	}

	public bool StartContinousGrabbing(int targetFPS) {
		// TODO: Add warning when initializing with null processor
		if (cProcessor == null) return false;
		if (!CaptureInitialized) return false;

		CAmbilightProcessor.pStartContinuousGrabbing(cProcessor.Value, targetFPS);

		return true;
	}

	public void StopContinousGrabbing(int targetFPS) {
		// TODO: Add warning when initializing with null processor
		if (cProcessor == null) return;

		CAmbilightProcessor.pStopContinuousGrabbing(cProcessor.Value);
	}
}