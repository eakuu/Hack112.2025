from cmu_graphics import *
import threading
import serial
import time
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import random

class ParkinsonsDataCollector:
    def __init__(self, port, modelPath, encoderPath, baudrate=115200):
        self.port = port                # The chosen port by the person (we used usbmodem101).
        self.baudrate = baudrate        # This is the symbol frequency at which data is collected. 115200 is standard.
        self.ser = None                 # This is for the serial port connection.
        self.allData = []               # This is representative of all x,y,z samples for all 10 seconds. That is a total of 3000 samples!
        self.predictions = []           # This is representative of each percentage prediction of Parkinsons every second.
        self.isCollecting = False       # This is the variable that is dependent on the button press and enables a loop.
        self.currentSecond = 0          # This helps for displaying the images of what second out of 10 the user is on.
        

        # Here we load the model from the model we manually trained. load_model is builtin to tensorflow.
        print(f"Loading model from {modelPath}...")
        self.model = load_model(modelPath)
        
        # We also have to load the encoded symbols to understand True and False predictions. There is no builtin so manually open file.
        print(f"Loading encoder from {encoderPath}...")
        self.encoder = self.loadEncoder(encoderPath)
    
    def loadEncoder(self, encoderPath):
        try:
            with open(encoderPath, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            print(f"Encoder file {encoderPath} not found.")
            return None
    
    # To establish serial port connection, check port, analyze at symbole rate we defined, and ensure connection to Arduino.
    # The time.sleep and resent buffer are standard steps in connecting devices by serial port as buffers.
    def connect(self):
        self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        time.sleep(2)
        self.ser.reset_input_buffer()
        print("Connected to Arduino!")
    
    # This is a really important part, as this is what reads each line outputted as serial data from the arduino.
    # It'll help us find the start of the thing (when serial monitor says BUTTON_PRESSED), collect XYZ data, and end it.
    def readLine(self):
        return self.ser.readline().decode('utf-8').strip()
    
    # This is just some standard data processing to stack x y and z into one data sequence, 
    # and either pads or truncates around teh target length.
    def prepareModelInput(self, x, y, z, target_length=100):
        dataSequence = np.column_stack((x, y, z))
        
        length = len(dataSequence)
        if length < target_length:
            padding = target_length - length
            padList = np.zeros((padding, 3))
            dataSequence = np.vstack((dataSequence, padList))
        elif length > target_length:
            dataSequence = dataSequence[:target_length, :]
        
        return np.expand_dims(dataSequence, axis=0)
    
    # This is what uses the keras model to predict the probability of Parkinsons.
    # It adds all the True and False predictions to a dictionary with the probabilities.
    def predictFromArrays(self, x, y, z):

        # Gives predictions in form [False Probability, True Probability]
        inputData = self.prepareModelInput(x, y, z)
        predictionProbs = self.model.predict(inputData, verbose=0)  

        # Adds the probabilities to a dict under their respective label.
        results = {}
        for i, className in enumerate(self.encoder.classes_):
            probability = float(predictionProbs[0][i] * 100)
            results[className] = probability                        
        
        # Gives the most likely class that it is [False or True]
        mostLikelyClass = max(results, key=results.get) 
        return mostLikelyClass, results
    
    # AI Generated startCollection because we haven't used thread before.
    def startCollection(self, callback):
        self.isCollecting = True
        self.allData = []
        self.predictions = []
        self.currentSecond = 0
        
        thread = threading.Thread(target=self._collectDataThread, args=(callback,))
        thread.daemon = True
        thread.start()
    
    # This function collects the 10 seconds of data threads and adds them to the allData and predictions lists.
    def _collectDataThread(self, callback):
        print("Waiting for button press on Arduino...")
        callback('waiting', None)
        
        while True:
            # Uses the readLine function to check for indicators.
            line = self.readLine()
            
            # ReadLine sees a button press message from the arduino.
            if line == "BUTTON_PRESSED":
                print("Button pressed! Collecting data...")
                callback('collecting', None)
                
                # Starts collecting in a loop.
                while True:
                    xLine = None
                    yLine = None
                    zLine = None
                    
                    # The for loop of 10 here is to ensure no data loss.
                    # We were losing entire lines because of garbage serial communication,
                    # and slow laptops lol. But with this loop its almost guaranteed we read all the data as needed.
                    for _ in range(10):
                        line = self.readLine()
                        
                        # Checks the prefix of the line. For reference, the arduino outputs lines like this 'X:n1,n2,n3,n4...'
                        if line[:2] == "X:":
                            xLine = line
                        elif line[:2] == "Y:":
                            yLine = line
                        elif line[:2] == "Z:":
                            zLine = line
                        elif line == "COLLECTION_COMPLETE":
                            print(f"\nCollection complete!")
                            self.isCollecting = False
                            avgProb = np.mean(self.predictions) if self.predictions else 0
                            stdDev = np.std(self.predictions) if self.predictions else 0
                            callback('complete', {'avg': avgProb, 'std': stdDev})
                            return
                        
                        # This is when xLine yLine and zLine all finally exist without dataLoss of the Xs or the entire lines.
                        if xLine and yLine and zLine:
                            xValues = np.array([float(v) for v in xLine[2:].split(',')])
                            yValues = np.array([float(v) for v in yLine[2:].split(',')])
                            zValues = np.array([float(v) for v in zLine[2:].split(',')])
                            
                            self.allData.append((xValues, yValues, zValues))
                            
                            mostLikelyClass, results = self.predictFromArrays(xValues, yValues, zValues)
                            parkinsonsProb = results.get('True', results.get(True, 0.0))
                            self.predictions.append(parkinsonsProb)
                            self.currentSecond = len(self.allData)
                            
                            callback('update', {'second': self.currentSecond, 'prob': parkinsonsProb})
                            print(f"Second {self.currentSecond}/10 - Parkinsons: {parkinsonsProb:.1f}%")
                            break
                        
    # Close after 10 seconds so that a new one can be initiated.
    def close(self):
        if self.ser:
            self.ser.close()
            print("Connection closed.")

# ----------eye animation idle state----------
def setupEye(app):
    app.eyeX = app.width//2
    app.eyeY = 450
    
    app.pupilOffset = 0
    app.pupilTarget = 0
    app.lastChangeTime = time.time()
    app.nextChangeDelay = random.uniform(0.5, 2.5)

def updateEye(app):
    currentTime = time.time()
    if currentTime - app.lastChangeTime > app.nextChangeDelay:
        app.pupilTarget = random.randint(-40, 40)
        app.lastChangeTime = currentTime
        app.nextChangeDelay = random.uniform(0.5, 2.5)
    
    speed = 2
    if app.pupilOffset < app.pupilTarget:
        app.pupilOffset += speed
    elif app.pupilOffset > app.pupilTarget:
        app.pupilOffset -= speed

def drawEye(app):
    cx = app.eyeX
    cy = app.eyeY
    eyeWidth = 220
    eyeHeight = 90
    leftX = cx - eyeWidth//2
    rightX = cx + eyeWidth//2
    topY = cy - eyeHeight//2
    bottomY = cy + eyeHeight//2
    points = [
        leftX, cy, 
        (leftX + cx)//2, (cy + topY)//2 - 10,
        ((leftX + cx)//2 + cx)//2, (cy + topY)//2 - 18,
        cx, topY, 
        ((rightX + cx)//2 + cx)//2, (cy + topY)//2 - 18,
        (cx+rightX)//2, (topY+cy)//2 - 10, 
        rightX, cy,
        (cx+rightX)//2, (bottomY+cy)//2 + 10, 
        ((rightX + cx)//2 + cx)//2, (cy + bottomY)//2 + 20,
        cx, bottomY, 
        ((leftX + cx)//2 + cx)//2, (cy + bottomY)//2 + 20,
        (leftX + cx)//2, (cy + bottomY)//2 + 10, 
        leftX, cy
        ]
    drawPolygon(*points, fill=None, border='cyan', borderWidth=5)
    pupilX = cx + app.pupilOffset
    drawCircle(pupilX, cy, 28, fill=None, border='cyan', borderWidth=4)



# App
def onAppStart(app):
    app.width = 1200
    app.height = 700
    app.state = 'idle'  # The states include idle, waiting, collecting, complete
    app.currentSecond = 0
    app.currentProb = 0
    app.allData = [] # All XYZ data, similar to the class attribute we have
    app.predictions = [] # All predictions, similar to the class attribute we have
    app.avgProb = 0
    app.stdDev = 0
    app.stepsPerSecond = 10  # Check for updates 10 times per second. It does not have to be as fast as the device because it outputs once per second.
    app.pendingUpdate = None

    #eye app start
    setupEye(app)
    
    # This just initializes the collector. Claude.ai recommended we use the try/except notation for this.
    # app.collector is an instance of the ParkinsonsDataCollector class, with the port, model, and encoder all defined.
    try:
        app.collector = ParkinsonsDataCollector(
            port='/dev/cu.usbmodem101',
            modelPath='parkinsonsEye.keras',
            encoderPath='labelEncoder.pkl'
        )
        app.collector.connect()
        app.connected = True
    except Exception as e:
        print(f"Error initializing: {e}")
        app.connected = False

# AI Added this. Its specifically for whether there has been a change or not.
def dataCallback(app, state, data):
    app.pendingUpdate = (state, data)

# This processes the updates. It helps to initialize variables and add values to dictionaries.
def onStep(app):
    #update eye animation
    updateEye(app)
    if app.pendingUpdate:
        state, data = app.pendingUpdate
        app.pendingUpdate = None
        
        if state == 'waiting':
            app.state = 'waiting'
        elif state == 'collecting':
            app.state = 'collecting'
            app.predictions = []
            app.allData = []
        elif state == 'update':
            app.currentSecond = data['second']
            app.currentProb = data['prob']
            app.predictions.append(data['prob'])
        elif state == 'complete':
            app.state = 'complete'
            app.avgProb = data['avg']
            app.stdDev = data['std']
            app.allData = app.collector.allData 

# Checks for button presses within certain regions depending on the state of the app.
def onMousePress(app, mouseX, mouseY):
    if app.state in ['idle', 'complete']:
        if app.state == 'idle':
            buttonX, buttonY = app.width // 2, 150
            buttonWidth, buttonHeight = 200, 60
        else: 
            buttonX, buttonY = app.width // 2, app.height - 40
            buttonWidth, buttonHeight = 200, 50
        
        if (buttonX - buttonWidth//2 <= mouseX <= buttonX + buttonWidth//2 and
            buttonY - buttonHeight//2 <= mouseY <= buttonY + buttonHeight//2):
            if app.connected:
                app.state = 'waiting'
                app.currentSecond = 0
                app.currentProb = 0
                app.predictions = []
                app.allData = []
                app.collector.startCollection(lambda s, d: dataCallback(app, s, d))

# This draws the graph based on the hand movement. AI Recommended a lot of try/except notation in case of fault serial comms.
# There was solid AI use in this section, as the graph was incredibly hard to create. We did most of the parameterization,
# and AI added a lot of the try/except stuff after.
def drawMovementGraph(app):
    try:
        # Graph parameters
        graphX = 150
        graphY = 275
        graphWidth = app.width - 300
        graphHeight = 320 
        
        # Draw graph background
        drawRect(graphX, graphY, graphWidth, graphHeight, fill='black', 
                 border='cyan', borderWidth=2)
        
        drawLabel('Hand Movement Data (XYZ Accelerometer)', 
                  graphX + graphWidth//2, graphY - 20, size=18, bold=True, fill='cyan', font = 'monospace')
        
        if not app.allData:
            drawLabel('No data available', graphX + graphWidth//2, 
                     graphY + graphHeight//2, size=16, fill='gray', font = 'monospace')
            return
    except Exception as e:
        print(f"Error in graph setup: {e}")
        return
    
    try:
        # Flatten all data into continuous arrays - sample every 5 points
        allX = []
        allY = []
        allZ = []
        
        for xVals, yVals, zVals in app.allData:
            # Convert to list and sample every 5th point
            xList = list(xVals)
            yList = list(yVals)
            zList = list(zVals)
            
            allX.extend(xList[::5])  # Every 5th point
            allY.extend(yList[::5])
            allZ.extend(zList[::5])
        
        totalSamples = len(allX)
        
        if totalSamples == 0:
            return
        
        # Find min/max for scaling
        allValues = allX + allY + allZ
        minVal = min(allValues)
        maxVal = max(allValues)
        
        # Add padding to range
        valueRange = maxVal - minVal
        if valueRange == 0:
            valueRange = 1
        minVal -= valueRange * 0.1
        maxVal += valueRange * 0.1
        valueRange = maxVal - minVal
    except Exception as e:
        print(f"Error processing data: {e}")
        drawLabel(f'Error: {str(e)}', graphX + graphWidth//2, 
                 graphY + graphHeight//2, size=14, fill='red', font = 'monospace')
        return
    
    try:
        # Draw the grid lines
        for i in range(5):
            y = graphY + (i * graphHeight / 4)
            drawLine(graphX, y, graphX + graphWidth, y, fill='dimGray', lineWidth=1)
            value = maxVal - (i * valueRange / 4)
            drawLabel(f'{value:.1f}', graphX - 15, y, size=10, fill='lightGray', align='right', font = 'monospace')
        
        # Draw the bottom axis line
        drawLine(graphX, graphY + graphHeight, graphX + graphWidth, graphY + graphHeight, 
                fill='cyan', lineWidth=2)
        
        # Draw the markers
        for second in range(11):
            x = graphX + (second * graphWidth / 10)
            drawLine(x, graphY + graphHeight, x, graphY + graphHeight + 5, 
                    fill='cyan', lineWidth=1)
            drawLabel(f'{second}s', x, graphY + graphHeight + 15, 
                     size=10, fill='white', font = 'monospace')
    except Exception as e:
        print(f"Error drawing axes: {e}")
    
    try:
        # Helper function to convert data point to screen coordinates
        def toScreen(index, value):
            x = graphX + (index / totalSamples) * graphWidth
            normalizedValue = (value - minVal) / valueRange
            normalizedValue = max(0, min(1, normalizedValue))
            y = graphY + graphHeight - (normalizedValue * graphHeight)
            return x, y
        
        # Draw lines for X, Y, Z with different colors
        datasets = [
            ('red', allX, 'X'),
            ('lime', allY, 'Y'),
            ('deepSkyBlue', allZ, 'Z')
        ]
        
        for (color, data, label) in datasets:
            if len(data) > 1:
                for i in range(len(data) - 1):
                    try:
                        x1, y1 = toScreen(i, data[i])
                        x2, y2 = toScreen(i + 1, data[i + 1])
                        drawLine(x1, y1, x2, y2, fill=color, lineWidth=1.5)
                    except Exception as e:
                        print(f"Error drawing line for {label}: {e}")
                        pass  # Skip if there's any error with individual points
    except Exception as e:
        print(f"Error drawing data lines: {e}")
    
    try:
        # Draw legend
        legendX = graphX + graphWidth - 100
        legendY = graphY + 20
        legend_items = [
            ('red', 'X-axis'),
            ('lime', 'Y-axis'),
            ('deepSkyBlue', 'Z-axis')
        ]
        
        for idx, (color, label) in enumerate(legend_items):
            y = legendY + idx * 25
            drawLine(legendX, y, legendX + 30, y, fill=color, lineWidth=3)
            drawLabel(label, legendX + 50, y, size=12, fill='white', align='left', font = 'monospace')
        
        # Draw vertical lines to separate seconds (static)
        for second in range(1, 10):
            x = graphX + (second * graphWidth / 10)
            drawLine(x, graphY, x, graphY + graphHeight, 
                    fill='dimGray', lineWidth=1, dashes=True)
    except Exception as e:
        print(f"Error drawing legend/separators: {e}")

def redrawAll(app):
    # Background
    drawRect(0, 0, app.width, app.height, fill='black')
    
    # Title
    drawLabel("Parkinson's Disease Detection", app.width//2, 50, 
              size=32, bold=True, fill='cyan', font = 'monospace')
    
    # Begin Button
    buttonX, buttonY = app.width // 2, 150
    buttonWidth, buttonHeight = 200, 60
    
    if app.state in ['idle', 'complete']:
        if app.state == 'idle':
            drawEye(app)
            drawRect(buttonX, buttonY, buttonWidth, buttonHeight, 
                    fill='black', border='cyan', borderWidth=3, align='center')
            drawLabel('BEGIN', buttonX, buttonY, size=24, bold=True, fill='cyan', font = 'monospace')
    
    # Status Display
    if app.state == 'idle':
        if not app.connected:
            drawLabel('Error: Could not connect to device', app.width//2, 250, 
                     size=18, fill='red', font = 'monospace')
        else:
            drawLabel('Press BEGIN to start', app.width//2, 250, 
                     size=20, fill='white', font = 'monospace')
    
    elif app.state == 'waiting':
        drawLabel('Waiting for button press on device...', app.width//2, 250, 
                 size=22, fill='cyan', bold=True, font = 'monospace')
        drawLabel('Please press the button on your Arduino', app.width//2, 290, 
                 size=16, fill='white', font = 'monospace')
    
    elif app.state == 'collecting':
        drawLabel(f'Collecting Data: {app.currentSecond}/10 seconds', 
                 app.width//2, 250, size=22, fill='cyan', bold=True, font = 'monospace')
        
        # Progress bar for seconds
        barWidth = 500
        barHeight = 30
        barX = app.width//2 - barWidth//2
        barY = 290
        
        drawRect(barX, barY, barWidth, barHeight, fill='white', border='cyan')
        if app.currentSecond > 0:
            progress = app.currentSecond / 10
            drawRect(barX, barY, barWidth * progress, barHeight, fill='cyan')
        
        # Current probability
        if app.currentSecond > 0:
            drawLabel(f'Current Probability: {app.currentProb:.1f}%', 
                     app.width//2, 360, size=28, bold=True, fill='white', font = 'monospace')
            
            # Color indicator
            color = 'green' if app.currentProb < 50 else 'red' 
            drawCircle(app.width//2, 430, 40, fill=color, opacity=50)
        
        # Show all predictions so far
        if app.predictions:
            y = 500
            drawLabel('Second-by-Second Results:', app.width//2, y, 
                     size=16, bold=True, fill='white', font = 'monospace')
            
            predText = ' | '.join([f'{i+1}: {p:.1f}%' for i, p in enumerate(app.predictions)])
            drawLabel(predText, app.width//2, y + 30, size=14, fill='white', font = 'monospace')
    
    elif app.state == 'complete':
        
        drawRect(app.width//2, 150, 440, 120, fill='black', 
                border='cyan', borderWidth=2, align='center')
        
        drawLabel(f'Average Probability: {app.avgProb:.2f}%', 
                 app.width//2, 120, size=20, bold=True, fill='white', font = 'monospace')
        drawLabel(f'Standard Deviation: {app.stdDev:.2f}%', 
                 app.width//2, 150, size=16, fill='gray', font = 'monospace')
        
        # Final diagnosis
        if app.avgProb >= 50.0:
            result = "Parkinson's indicators detected"
            color = 'red'
        else:
            result = "Normal movement patterns"
            color = 'green'
        
        drawLabel(f'Result: {result}', app.width//2, 180, 
                 size=18, bold=True, fill=color, font = 'monospace')
        
        # Draw the graph
        drawMovementGraph(app)
        
        # Begin button at bottom
        buttonX, buttonY = app.width // 2, app.height - 40
        buttonWidth, buttonHeight = 200, 50
        drawRect(buttonX, buttonY, buttonWidth, buttonHeight, 
                fill='black', border='cyan', borderWidth=3, align='center')
        drawLabel('BEGIN AGAIN', buttonX, buttonY, size=18, bold=True, fill='cyan', font = 'monospace')

def onAppStop(app):
    if hasattr(app, 'collector'):
        app.collector.close()

runApp()
