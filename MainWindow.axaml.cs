using Avalonia.Controls;

using System;
using System.Collections.Generic;
using AmbilightAPI;

namespace YetAnotherAmbilightKnockoff;

public partial class MainWindow : Window
{
	public MainWindow()
	{
		InitializeComponent();
		TestAmbilight();
	}

	void TestAmbilight() {
		// Define kernel parameters
		var kParams = new KernelParams() {
			frameWidth = 2560,
			frameHeight = 1600,
			sectorCount = 10,
		};

		// Define display sectors
		var sectors = new List<Sector>();
		uint sWidth = (uint)(kParams.frameWidth / kParams.sectorCount);

		for (uint i = 0; i < kParams.sectorCount; i++) {
			sectors.Add(new Sector() {
				x = i * sWidth,
				y = 0,
				width = sWidth,
				height = 100
			});
		}

		var sectorMap = new SectorMap(sectors);

		// Init processor
		var processor = new AmbilightProcessor(kParams, sectorMap);
		processor.InitCapture();

		// Get frame
		var frame = processor.GetSingleFrame();

		if (frame != null) {
			foreach (HSVPixel p in frame) {
				Console.WriteLine(String.Format("({0}, {1}, {2})", p.h, p.s, p.v));
			}
		} else {
			Console.WriteLine("Frame is null????");
		}
	}
}