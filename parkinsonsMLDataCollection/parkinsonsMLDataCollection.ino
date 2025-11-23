/*
 * 30-Second Continuous Accelerometer Data Collection
 * For Adafruit ItsyBitsy nRF52840 with Python App Integration
 * 
 * Components:
 * - SparkFun ADXL362 Triple Axis Accelerometer (SPI)
 * - Push button on pin A3 (pulled high, press to GND)
 * 
 * Press the button to collect 30 seconds of acceleration data at 100Hz
 * Outputs JSON every second with 100 samples from that second
 * 
 * Libraries needed:
 * - SparkFun ADXL362 Arduino Library (SparkFun ADXL362)
 * 
 * Wiring:
 * ADXL362 (SPI):
 *   - VDD -> 3.3V
 *   - GND -> GND
 *   - CS -> Pin 2
 *   - MOSI -> MOSI (Pin 11 on ItsyBitsy)
 *   - MISO -> MISO (Pin 12 on ItsyBitsy)
 *   - SCLK -> SCK (Pin 13 on ItsyBitsy)
 * 
 * Button:
 *   - One side -> Pin A3
 *   - Other side -> GND
 *   - Internal pullup resistor enabled
 */

#include <SPI.h>
#include <ADXL362.h>

// Pin definitions
const int ADXL362_CS = 2;
const int BUTTON_PIN = A3;

// ADXL362 Accelerometer
ADXL362 adxl362(ADXL362_CS);

// Data collection parameters
const int SAMPLE_RATE_HZ = 100;
const int COLLECTION_DURATION_SEC = 10;  // Total collection time
const int SAMPLES_PER_SECOND = 100;      // Samples per JSON output
const unsigned long SAMPLE_INTERVAL_MS = 1000 / SAMPLE_RATE_HZ;  // 10ms

// Data storage arrays for one second worth of data
float xData[SAMPLES_PER_SECOND];
float yData[SAMPLES_PER_SECOND];
float zData[SAMPLES_PER_SECOND];

// State variables
bool isCollecting = false;
int sampleCount = 0;              // Current sample within the second
int secondsElapsed = 0;           // How many seconds have elapsed
unsigned long lastSampleTime = 0;
unsigned long collectionStartTime = 0;

// Button debouncing
unsigned long lastButtonPress = 0;
const unsigned long DEBOUNCE_DELAY = 200;  // 200ms debounce

void setup() {
  // Initialize Serial communication
  Serial.begin(115200);
  
  // Wait for serial port to connect (optional for nRF52840)
  delay(1000);
  
  Serial.println("Arduino Ready");
  
  // Initialize button pin with internal pullup
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  
  // Initialize SPI for ADXL362
  SPI.begin();
  
  if (adxl362.init() <= 0) {
    Serial.println("ERROR: ADXL362 init failed");
    while (1);
  }
  
  // Activate measurement mode (100Hz, ±2g range)
  if (adxl362.activateMeasure() <= 0) {
    Serial.println("ERROR: ADXL362 activate failed");
    while (1);
  }
  
  Serial.println("Press button to start 30-second data collection...");
}

void loop() {
  // Check for button press
  checkButton();
  
  // Collect data if in collection mode
  if (isCollecting) {
    collectData();
  }
}

void checkButton() {
  // Read button state (LOW when pressed due to pullup)
  if (digitalRead(BUTTON_PIN) == LOW && !isCollecting) {
    // Debounce check
    unsigned long currentTime = millis();
    if (currentTime - lastButtonPress > DEBOUNCE_DELAY) {
      lastButtonPress = currentTime;
      startCollection();
    }
  }
}

void startCollection() {
  // Send signals to Python app
  Serial.println("BUTTON_PRESSED");
  Serial.println("COLLECTING - 30 seconds");
  
  isCollecting = true;
  sampleCount = 0;
  secondsElapsed = 0;
  collectionStartTime = millis();
  lastSampleTime = millis();
  
  // Clear arrays
  clearArrays();
}

void clearArrays() {
  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    xData[i] = 0.0;
    yData[i] = 0.0;
    zData[i] = 0.0;
  }
}

void collectData() {
  unsigned long currentTime = millis();
  
  // Check if it's time to take a sample
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL_MS) {
    lastSampleTime = currentTime;
    
    // Read accelerometer data
    MeasurementInMg accelData = adxl362.getXYZ(ad_range_2G);
    
    // Convert from millig to g-force and store
    xData[sampleCount] = accelData.x / 1000.0;
    yData[sampleCount] = accelData.y / 1000.0;
    zData[sampleCount] = accelData.z / 1000.0;
    
    sampleCount++;
    
    // Check if we've collected 100 samples (1 second worth)
    if (sampleCount >= SAMPLES_PER_SECOND) {
      secondsElapsed++;
      
      // Output JSON for this second
      outputJSON();
      
      // Reset for next second
      sampleCount = 0;
      clearArrays();
      
      // Check if 30 seconds have elapsed
      if (secondsElapsed >= COLLECTION_DURATION_SEC) {
        finishCollection();
      }
    }
  }
}

void finishCollection() {
  isCollecting = false;
  
  unsigned long totalTime = millis() - collectionStartTime;
  
  Serial.println("=================================");
  Serial.println("Collection complete!");
  Serial.print("Total time: ");
  Serial.print(totalTime / 1000.0);
  Serial.println(" seconds");
  Serial.print("Total JSON outputs: ");
  Serial.println(secondsElapsed);
  Serial.println("=================================");
  Serial.println("Press button for another collection...");
}

void outputJSON() {
  Serial.println("{");
  Serial.println("  \"label\": \"unknown\",");
  Serial.println("  \"data\": {");
  
  // Output X data
  Serial.print("    \"x\": [");
  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    Serial.print(xData[i], 3);  // 3 decimal places for precision
    if (i < SAMPLES_PER_SECOND - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("],");
  
  // Output Y data
  Serial.print("    \"y\": [");
  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    Serial.print(yData[i], 3);
    if (i < SAMPLES_PER_SECOND - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("],");
  
  // Output Z data
  Serial.print("    \"z\": [");
  for (int i = 0; i < SAMPLES_PER_SECOND; i++) {
    Serial.print(zData[i], 3);
    if (i < SAMPLES_PER_SECOND - 1) {
      Serial.print(", ");
    }
  }
  Serial.println("]");
  
  Serial.println("  }");
  Serial.println("},");
}