using Avalonia.Controls;

using System;
using System.Timers;
using System.Collections.Generic;
using AmbilightAPI;
using HardwareAPI;

namespace YetAnotherAmbilightKnockoff;

public partial class MainWindow : Window
{
	public MainWindow()
	{
		InitializeComponent();
		TestAmbilight();
		// TestLEDController();
	}

	TAmbilightProcessor processor;

	// Makes sure the delegate isn't garbage collected
	static AmbilightFrameReadyCallback? cbDelegate;

	void TestLEDController() {
		var controller = LEDController.Instance;

		controller.OpenSerial("/dev/ttyACM0");
		controller.SetLED(0, new LEDColor(255, 0, 0));
		controller.SetLED(1, new LEDColor(0, 255, 0));
		controller.SetLED(2, new LEDColor(0, 0, 255));
		controller.SetLED(10, new LEDColor(255, 0, 255));
		controller.SetLED(11, new LEDColor(255, 255, 255));

		controller.SetLED(39, new LEDColor(255, 0, 0));

		controller.CommitFrame();
	}

	void TestAmbilight() {
		// Define kernel parameters
		var kParams = new KernelParams() {
			frameWidth = 2560,
			frameHeight = 1600,
			sectorCount = 40,
		};

		// Define display sectors
		var sectors = new List<Sector>();

		uint height = 100;
		int xSectors = 10;
		int ySectors = 10;

		uint xsWidth = (uint)(kParams.frameWidth / xSectors);
		uint ysHeight = (uint)((kParams.frameHeight - height * 2) / ySectors);

		// Top
		for (uint i = 0; i < xSectors; i++) {
			sectors.Add(new Sector() {
				x = i * xsWidth,
				y = 0,
				width = xsWidth,
				height = height
			});
		}

		// Right
		for (uint i = 0; i < xSectors; i++) {
			sectors.Add(new Sector() {
				x = (uint)kParams.frameWidth - height,
				y = 100 + i * ysHeight,
				width = height,
				height = ysHeight
			});
		}

		// Bottom
		for (uint i = 0; i < xSectors; i++) {
			sectors.Add(new Sector() {
				x = i * xsWidth,
				y = (uint)kParams.frameHeight - height,
				width = xsWidth,
				height = height
			});
		}

		// Left
		for (uint i = 0; i < xSectors; i++) {
			sectors.Add(new Sector() {
				x = 0,
				y = 100 + i * ysHeight,
				width = height,
				height = ysHeight
			});
		}

		var sectorMap = new SectorMap(sectors);

		LEDController.Instance.OpenSerial("/dev/ttyACM0");

		// Init processor
		processor = new TAmbilightProcessor(kParams, sectorMap);

		cbDelegate = new AmbilightFrameReadyCallback(ReadFrame);
		processor.OnFrameReady(cbDelegate);

		processor.StartCapture(60, true);
	}

	void ReadFrame() {
		Console.WriteLine(String.Format("{0} FPS", processor.getActualFPS()));
		var frame = processor.ReadFrame();

		if (frame != null) {
			for (int i = 0; i < frame.Count; i++) {
				var col = HSVPixelToLED(frame[i]);
				LEDController.Instance.SetLED((byte)i, col);
			}

			LEDController.Instance.CommitFrame();
		} else {
			Console.WriteLine("Frame is null????");
		}
	}

	LEDColor HSVPixelToLED(HSVPixel p) {
		RGBPixel rgbP = RGBPixel.FromHSVPixel(p);
		LEDColor led = new LEDColor(
			(byte)(rgbP.r * 255),
			(byte)(rgbP.g * 255),
			(byte)(rgbP.b * 255)
		);

		return led;
	}
}