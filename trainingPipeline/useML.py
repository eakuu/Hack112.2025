import serial
import time
import numpy as np
import pickle
from tensorflow.keras.models import load_model

class ParkinsonsDataCollector:
    def __init__(self, port, modelPath, encoderPath, baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.allData = [] 
        self.predictions = [] 
        
        print(f"Loading model from {modelPath}...")
        self.model = load_model(modelPath)
        
        print(f"Loading encoder from {encoderPath}...")
        self.encoder = self.loadEncoder(encoderPath)
    
    def loadEncoder(self, encoderPath):
        try:
            with open(encoderPath, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"Encoder file {encoderPath} not found.")
            return None
    
    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print("Connected to Arduino!")
    
    def readLine(self):
        return self.ser.readline().decode('utf-8').strip()
    
    def prepareModelInput(self, x, y, z, target_length=100):
        dataSequence = np.column_stack((x, y, z))
        
        # Pad or truncate
        length = len(dataSequence)
        if length < target_length:
            padding = target_length - length
            padList = np.zeros((padding, 3))
            dataSequence = np.vstack((dataSequence, padList))
        elif length > target_length:
            dataSequence = dataSequence[:target_length, :]
        
        # Add batch dimension
        return np.expand_dims(dataSequence, axis=0)
    
    def predictFromArrays(self, x, y, z):
        inputData = self.prepareModelInput(x, y, z)
        predictionProbs = self.model.predict(inputData, verbose=0)
        
        # Extract probabilities for each class
        results = {}
        for i, className in enumerate(self.encoder.classes_):
            probability = float(predictionProbs[0][i] * 100)
            results[className] = probability
        
        # Find predicted class
        predictedClass = max(results, key=results.get)
        
        return predictedClass, results
    
    def collectData(self):
        print("Waiting for button press on Arduino...")
        
        while True:
            line = self.readLine()
            
            # Wait for button press
            if line == "BUTTON_PRESSED":
                print("Button pressed! Collecting data...")
                self.allData = []
                self.predictions = []
                
                # Collect 10 seconds of data
                while True:
                    xLine = None
                    yLine = None
                    zLine = None
                    
                    # Read X, Y, Z lines
                    for _ in range(10):
                        line = self.readLine()
                        
                        if line[:2] == "X:":
                            xLine = line
                        elif line[:2] == "Y:":
                            yLine = line
                        elif line[:2] == "Z:":
                            zLine = line
                        elif line == "COLLECTION_COMPLETE":
                            print(f"\nCollection complete! Total samples: {len(self.allData)} seconds")
                            self.displayResults()
                            return
                        
                        # Got all three lines
                        if xLine and yLine and zLine:
                            # Parse data
                            xValues = np.array([float(v) for v in xLine[2:].split(',')])
                            yValues = np.array([float(v) for v in yLine[2:].split(',')])
                            zValues = np.array([float(v) for v in zLine[2:].split(',')])
                            
                            # Store raw data
                            self.allData.append((xValues, yValues, zValues))
                            
                            # Make prediction
                            predictedClass, results = self.predictFromArrays(xValues, yValues, zValues)
                            
                            # Store Parkinson's probability (True class)
                            parkinsonsProb = results.get('True', results.get(True, 0.0))
                            self.predictions.append(parkinsonsProb)
                            
                            print(f"Second {len(self.allData)}/10 - Prediction: {predictedClass} (Parkinsons: {parkinsonsProb:.1f}%)")
                            break
    
    def displayResults(self):
        if not self.predictions:
            print("No predictions made.")
            return
        
        avgParkinsonsProb = np.mean(self.predictions)
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Average Parkinson's Probability: {avgParkinsonsProb:.2f}%")
        print(f"Std Dev: {np.std(self.predictions):.2f}%")
        
        # Final classification
        if avgParkinsonsProb >= 50.0:
            print(f"\nRESULT: Parkinson's indicators detected")
        else:
            print(f"\nRESULT: Normal movement patterns")
        print("="*60)
    
    def getData(self):
        return self.allData
    
    def getPredictions(self):
        return self.predictions
    
    def close(self):
        if self.ser:
            self.ser.close()
            print("Connection closed.")


def main():
    
    collector = ParkinsonsDataCollector(
        port = '/dev/cu.usbmodem101',
        modelPath = 'parkinsonsEye.keras',
        encoderPath = 'labelEncoder.pkl'
    )
    
    collector.connect()
    
    try:
        collector.collectData()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
