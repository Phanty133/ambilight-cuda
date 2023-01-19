using System;
using System.Collections.Generic;

namespace AmbilightAPI;

public class TAmbilightProcessor {
	public KernelParams kParams;

	public SectorMap sectorMap;

	private IntPtr? cProcessor = null;

	private IntPtr? calcThread = null;

	private CKernelParams cParams;

	private CSector[] cSectorMap;

	public bool IsActive {
		get {
			if (!cProcessor.HasValue) return false;
			if (!calcThread.HasValue) return false;

			return CTAmbilightProcessor.tpIsActive(cProcessor.Value);
		}
	}

	public bool IsFrameReady {
		get {
			if (!IsActive) return false;

			return CTAmbilightProcessor.tpIsFrameReady(cProcessor!.Value);
		}
	}

	public TAmbilightProcessor(KernelParams kParams, SectorMap sectorMap) {
		this.kParams = kParams;
		this.sectorMap = sectorMap;

		cParams = kParams.ToCType();
		cSectorMap = sectorMap.ToCType();

		cProcessor = CTAmbilightProcessor.createTProcessor(cParams, ref cSectorMap[0]);
		calcThread = CTAmbilightProcessor.tpCreateThread(cProcessor.Value);
		CTAmbilightProcessor.tpDetachThread(cProcessor.Value);
	}

	~TAmbilightProcessor() {
		if (!cProcessor.HasValue) return;

		DestroyProcessor();
	}

	public void DestroyProcessor() {
		if (!cProcessor.HasValue) return;

		CTAmbilightProcessor.tpKill(cProcessor.Value);
		cProcessor = null;
	}

	public void StartCapture(int targetFPS, bool waitUntilReady = false) {
		if (!cProcessor.HasValue) return;
		if (!calcThread.HasValue) return;

		CTAmbilightProcessor.tpStart(cProcessor.Value, targetFPS, waitUntilReady);
	}

	public void StopCapture() {
		if (!IsActive) return;

		CTAmbilightProcessor.tpStop(cProcessor!.Value);
	}

	public float? getActualFPS() {
		if (!IsActive) return null;

		return CTAmbilightProcessor.tpGetActualFPS(cProcessor!.Value);
	}

	// Only gets the frame in memory
	public List<HSVPixel>? ReadFrame() {
		// TODO: Add warning when not active
		if (!IsActive) return null;

		CAveragedHSVPixel[] cOutput = new CAveragedHSVPixel[kParams.sectorCount];

		CTAmbilightProcessor.tpGetFrame(cProcessor!.Value, ref cOutput[0]);

		var output = new List<HSVPixel>();

		foreach (CAveragedHSVPixel p in cOutput) {
			output.Add(HSVPixel.FromCType(p));
		}

		return output;
	}

	public void OnFrameReady(AmbilightFrameReadyCallback cb) {
		if (!cProcessor.HasValue) return;

		CTAmbilightProcessor.tpOnFrameReady(cProcessor.Value, cb);
	}

	public void ClearFrameReadyFlag() {
		if (!cProcessor.HasValue) return;

		CTAmbilightProcessor.tpClearFrameReadyFlag(cProcessor.Value);
	}
}