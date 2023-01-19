using System;
using System.IO.Ports;

namespace HardwareAPI;

public struct LEDColor {
	public byte r; // 0 - 255
	public byte g; // 0 - 255
	public byte b; // 0 - 255

	public LEDColor(byte r, byte g, byte b) {
		this.r = r;
		this.g = g;
		this.b = b;
	}
}

public class LEDController {
	static LEDController? instance = null;
	static SerialPort? serialPort = null;

	private LEDController() {}

	public void OpenSerial(string port, int baudrate = 2_000_000) {
		if (serialPort != null) {
			serialPort.Close();
		}

		serialPort = new SerialPort(port, baudrate);
		serialPort.Open();
	}

	public void SetLED(byte led, LEDColor color) {
		// TODO: Warning when null serialPort
		if (serialPort == null) return;

		byte[] buffer = new byte[4] {
			led,
			color.r,
			color.g,
			color.b
		};

		serialPort.Write(buffer, 0, buffer.Length);
	}
	
	public void CommitFrame() {
		// TODO: Warning when null serialPort
		if (serialPort == null) return;

		byte[] buffer = new byte[1] { 255 };

		serialPort.Write(buffer, 0, buffer.Length);
	}

	public static LEDController Instance {
		get {
			if (instance == null) {
				instance = new LEDController();
			}

			return instance;
		}
	}
}