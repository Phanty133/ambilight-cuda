#include <Arduino.h>
#include "FastLED.h"

#define BAUD_RATE 2000000
#define NUM_LEDS 40
#define DATA_PIN 6
#define REFRESH_RATE 2000

CRGB leds[NUM_LEDS];

void serial_flush() {
	while (Serial.available()) Serial.read();
}

void setup() {
	Serial.begin(BAUD_RATE);
	
	FastLED.addLeds<NEOPIXEL, DATA_PIN>(leds, NUM_LEDS);
	FastLED.setMaxRefreshRate(REFRESH_RATE);
	FastLED.show();

	serial_flush(); // just in case
}

void loop() {
	if (Serial.available()) {
		uint8_t id = Serial.read();

		if (id == 255) { // If the ID is 255, show the frame
			FastLED.show();
			serial_flush();
		} else if (id < NUM_LEDS) {
			leds[id].red = Serial.read();
			leds[id].green = Serial.read();
			leds[id].blue = Serial.read();
		}
	}
}