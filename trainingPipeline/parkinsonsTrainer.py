# Imports: NumPy for data handling, JSON for files, TensorFlow/Keras for CNN, & SkLearn for data preprocessing/evaluation.

import copy
import math
import numpy
import json
import tensorflow
import pickle
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Here we collected the data from the JSON file and parsed it.
# If the data is in a JSON array structure, it outputs a list.
# Elif the data is in a JSON object, it outputs a dictionary.
# In this case, the data structure of the JSON file we input from arduino is a JSON object.
# The variable rawData is a dictionary.

def loadAccelerometerData(jsonFilePathway):
  try:
    with open(jsonFilePathway, 'r') as file:
      rawData = json.load(file)
      return rawData

  except FileNotFoundError:
    print(f"The file {jsonFilePathway} doesn't exist. Upload a file.")

  except json.JSONDecodeError:
    print(f'''Could not decode file {jsonFilePathway}.''')

# Here we convert the raw data into a NumPy array digestible for convolutional neural networks.
# First it iterates through the data, and extracts 3 values at a time because of x, y, and z acceleration values.
# We then return this array of x, y, and z sequences as a NumPy array for the CNN.

def extractData(rawData):
  xyzData = []
  xyzMetadata = []
  counter = 0

  for sample in rawData:
    metadata = sample.get('label', 'unclassified')
    data = sample.get('data', dict())
    xList = data.get('x', [])
    yList = data.get('y', [])
    zList = data.get('z', [])

    if not (len(xList) == len(yList) == len(zList)):
      print(f'Because entry #{counter} contains mismatched data list lengths, it is to be skipped.')
      counter += 1
      continue

    elif len(xList) == 0 or len(yList) == 0 or len(zList) == 0:
      print(f'Because entry #{counter} contains missing data, it is to be skipped.')
      counter += 1
      continue

    elif metadata == 'unclassified':
      print(f'Because entry #{counter} is unclassified, it is to be skipped.')
      counter += 1
      continue

    dataSequence = numpy.column_stack((xList, yList, zList))
    xyzData.append(dataSequence)
    xyzMetadata.append(metadata)
    counter += 1

  return numpy.array(xyzData, dtype=object), numpy.array(xyzMetadata)

# Now, to ensure the data is standardized and everything is the same length,
# we trimmed the data that is greater than the target length,
# or padded the data that is less than the target length.

def truncateOrPadIrregularities(xyzData, target):
  standardizedXYZData = []

  for xyz2DList in xyzData:
    lengthSequence = len(xyz2DList)

    if lengthSequence < target:
      paddingLength = target - len(xyz2DList)
      padList = numpy.zeros((paddingLength, 3))
      xyz2DList = numpy.vstack((xyz2DList, padList))

    elif lengthSequence > target:
      xyz2DList = xyz2DList[:target, :]

    standardizedXYZData.append(xyz2DList)

  return standardizedXYZData

# Now, the xyzMetadata labels (either 'correct' or 'incorrect') are encoded to integers through a label encoder
# for the tensorflow library, as ML algorithms prefer numerical input for classification tasks.

def encode(xyzMetadata):
  encoder = LabelEncoder()
  encodedXYZMetadata = encoder.fit_transform(xyzMetadata)
  categoricalVectorXYZMetadata = to_categorical(encodedXYZMetadata)

  # To print the map of what classification maps to what number:
  print("\nEncoder mapping:")
  for n, label in enumerate(encoder.classes_):
      print(f"  {label}: {n}")

  return encodedXYZMetadata, categoricalVectorXYZMetadata, encoder

# This is then followed by splitting the data into a training set and a test set.
# The test size is to be 20% of the dataset, while the other 80% goes to training the model.
# We also used stratification to make sure both datasets have equal splits of correct and incorrect.

def finalPreprocessing(standardizedXYZData, encodedXYZMetadata, categoricalVectorXYZMetadata):

  x = numpy.array(standardizedXYZData, dtype=float)
  y1 = numpy.array(categoricalVectorXYZMetadata)
  y2 = numpy.array(encodedXYZMetadata)

  trainX, testX, trainY, testY = train_test_split(x, y1, test_size = 0.2, random_state = 1, stratify = y2)

  return trainX, testX, trainY, testY

# Building the CNN model for bicep curl form classification.
# The architecture uses multiple Convolutional 1D layers to extract time-based patterns from accelerometer data,
# with batch normalization and dropout to prevent model overfitting.
# These numbers were determined by claude.ai by Anthropic for a more accurate starting point.

def buildCNN(inputShape, numClasses):
  model = Sequential([

    # First convolutional block
    Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=inputShape, padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Second convolutional block
    Conv1D(filters=128, kernel_size=5, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    # Third convolutional block
    Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    # Flatten and dense layers
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.4),

    # Output
    Dense(numClasses, activation='softmax')
  ])

  # Compile the model with categorical crossentropy for multi-class classification
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  print("\n=== CNN Architecture ===")
  model.summary()

  return model

# Here, we train the model with early stopping and model checkpointing to save the best version.
# The values here were also claude.ai-determined. We chopped epochs from 100 to 50 to prevent overfitting.

def trainModel(model, trainX, trainY, testX, testY, epochs=50, batchSize=32):

  # Early stopping to prevent overfitting
  earlyStop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)

  # Save the best model during training
  checkpoint = ModelCheckpoint('parkinsonsEye.keras', monitor='val_accuracy', save_best_only=True, verbose=1)

  print("\n=== Training the Model ===")
  print(f"Training samples: {len(trainX)}")
  print(f"Testing samples: {len(testX)}")
  print(f"Epochs: {epochs}, Batch size: {batchSize}\n")

  # Train the model
  history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=epochs, batch_size=batchSize, callbacks=[earlyStop, checkpoint], verbose=1)

  return model, history

# Evaluate the trained model on the test set and display metrics WITH PROBABILITIES.

def evaluateModel(model, testX, testY, encoder):
  print("\n=== Model Evaluation ===")

  # Get predictions (these are already probabilities from softmax!)
  predictions = model.predict(testX)
  predictedClasses = numpy.argmax(predictions, axis=1)
  trueClasses = numpy.argmax(testY, axis=1)

  # NEW: Display sample predictions with probabilities
  print("\n=== Sample Predictions with Probabilities ===")
  num_samples_to_show = min(10, len(predictions))
  for i in range(num_samples_to_show):
    print(f"\nSample {i}:")
    for j, className in enumerate(encoder.classes_):
      probability = predictions[i][j] * 100
      print(f"  {className}: {probability:.2f}%")
    print(f"  → Predicted: {encoder.classes_[predictedClasses[i]]}")
    print(f"  → True label: {encoder.classes_[trueClasses[i]]}")

  # Calculate test accuracy
  testLoss, testAccuracy = model.evaluate(testX, testY, verbose=0)
  print(f"\n=== Overall Performance ===")
  print(f"Test Loss: {testLoss:.4f}")
  print(f"Test Accuracy: {testAccuracy:.4f}")

  # Classification report
  print("\n=== Classification Report ===")
  print(classification_report(trueClasses, predictedClasses, target_names=encoder.classes_))

  # Confusion matrix
  print("\n=== Confusion Matrix ===")
  cm = confusion_matrix(trueClasses, predictedClasses)
  print(cm)
  print(f"\nRows: True labels, Columns: Predicted labels")
  print(f"Classes: {encoder.classes_}")

  # NEW: Show probability distribution statistics
  print("\n=== Probability Statistics ===")
  for i, className in enumerate(encoder.classes_):
    class_probs = predictions[:, i] * 100
    print(f"{className}:")
    print(f"  Mean probability: {numpy.mean(class_probs):.2f}%")
    print(f"  Min probability: {numpy.min(class_probs):.2f}%")
    print(f"  Max probability: {numpy.max(class_probs):.2f}%")

  return testAccuracy, predictedClasses, predictions

# NEW FUNCTION: Predict Parkinson's probability for new data
def predictParkinsons(model, newData, encoder, targetLength=100):
  """
  Predict Parkinson's probability for new accelerometer data

  Args:
    model: trained Keras model
    newData: dictionary with 'x', 'y', 'z' lists
    encoder: LabelEncoder used during training
    targetLength: length to standardize data to

  Returns:
    dictionary with probabilities for each class
  """
  # Extract and format the data
  xList = newData.get('x', [])
  yList = newData.get('y', [])
  zList = newData.get('z', [])

  # Validate data
  if not (len(xList) == len(yList) == len(zList)):
    raise ValueError("X, Y, and Z lists must have the same length")

  if len(xList) == 0:
    raise ValueError("Data lists cannot be empty")

  # Create sequence
  dataSequence = numpy.column_stack((xList, yList, zList))

  # Standardize length
  if len(dataSequence) < targetLength:
    paddingLength = targetLength - len(dataSequence)
    padList = numpy.zeros((paddingLength, 3))
    dataSequence = numpy.vstack((dataSequence, padList))
  elif len(dataSequence) > targetLength:
    dataSequence = dataSequence[:targetLength, :]

  # Reshape for model input
  dataSequence = numpy.array([dataSequence], dtype=float)

  # Get prediction
  prediction = model.predict(dataSequence, verbose=0)

  # Create results dictionary
  results = {}
  print(f"\n=== Prediction Results ===")
  for i, className in enumerate(encoder.classes_):
    probability = prediction[0][i] * 100
    results[className] = probability
    print(f"{className}: {probability:.2f}%")

  return results

# Main execution pipeline

def main():
  # Configuration
  jsonFilePath = 'parkinsonsData.json'
  targetLength = 100

  print("=== Parkinson's Eye - Likelihood of Parkinson's Disease ===\n")

  # Step 1: Load data
  print("Step 1: Loading accelerometer data...")
  rawData = loadAccelerometerData(jsonFilePath)
  print(f"Loaded {len(rawData)} samples\n")

  # Step 2: Extract data
  print("Step 2: Extracting x, y, z accelerometer data...")
  xyzData, xyzMetadata = extractData(rawData)
  print(f"Valid samples: {len(xyzData)}\n")

  # Step 3: Standardize sequence lengths
  print(f"Step 3: Standardizing data to length {targetLength}...")
  standardizedXYZData = truncateOrPadIrregularities(xyzData, targetLength)
  print("Data standardization complete\n")

  # Step 4: Encode labels
  print("Step 4: Encoding labels...")
  encodedXYZMetadata, categoricalVectorXYZMetadata, encoder = encode(xyzMetadata)

  # Step 5: Split data
  print("\nStep 5: Splitting data into training and test sets...")
  trainX, testX, trainY, testY = finalPreprocessing(standardizedXYZData, encodedXYZMetadata, categoricalVectorXYZMetadata)

  # Step 6: Build CNN
  inputShape = (trainX.shape[1], trainX.shape[2])
  numClasses = trainY.shape[1]
  print(f"\nStep 6: Building CNN with input shape {inputShape} and {numClasses} classes...")
  model = buildCNN(inputShape, numClasses)

  # Step 7: Train model
  model, history = trainModel(model, trainX, trainY, testX, testY, epochs=100, batchSize=32)

  # Step 8: Evaluate model with probability outputs
  testAccuracy, predictedClasses, predictions = evaluateModel(model, testX, testY, encoder)

  print("\n=== Training Complete ===")
  print(f"Final test accuracy: {testAccuracy:.4f}")
  print("Best model saved as 'parkinsonsEye.keras'")

  # NEW: Save the encoder for later use in predictions
  print("\nSaving label encoder for future predictions...")
  with open('labelEncoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
  print("Label encoder saved as 'labelEncoder.pkl'")

if __name__ == "__main__":
  main()
