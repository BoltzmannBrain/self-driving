#!/usr/bin/env python
"""
Vehicle speed prediction model.

Example invocations:

  # Run verbose training
  ./cnn_prediction.py --verbosity 2

  # Only run the test phase
  ./cnn_prediction.py --skipTraining --test

Some suggestions for improving the model are to tune the optmizer learning rate
and other params (currently using Keras standards), or more/larger layers
(perhaps like the model here: https://arxiv.org/abs/1604.07316). Depending on
the input video stream, further preprocessing can be done; there are good tools
in opencv.
"""
import argparse
import cv2
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.callbacks import Callback
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D



_DEFAULT_MODEL_DIR = "./outputs/speed_model"
_VIDEO_CHANNELS = 3
_VIDEO_HEIGHT = 480
_VIDEO_WIDTH = 640


logging.basicConfig(level=logging.INFO)

class LossHistory(Callback):
  """ Helper class for useful logging info; set logger to DEBUG.
  """
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_epoch_end(self, epoch, logs={}):
    self.losses.append(logs.get("val_loss"))
    logging.debug("-----------------------------------------------------------")
    logging.debug("Epoch {} \nValidation loss = {} \nAccuracy = {}".format(
        epoch, logs.get("val_loss"), logs.get("val_acc")))
    logging.debug("-----------------------------------------------------------")

  def on_batch_end(self,batch,logs={}):
    logging.debug("Batch {} -- loss = {}, accuracy = {}".format(
        batch, logs.get("loss"), logs.get("acc")))



def _frameGenerator(videoPath, dataPath, batchSize, epoch, verbosity=0):
  """
  Yield X and Y data when the batch is filled.

  Args:
    videoPath: (str) Drive video for inference.
    dataPath: (str) JSON of truth data, as a list of [time, speed] items.
    verbosity: (int) Plot or not.
  Yields:
    Two-tuple representing a batch of input video frames and target values.
  """
  camera = cv2.VideoCapture(videoPath)
  width = int(camera.get(3))
  height = int(camera.get(4))
  frameCount = int(camera.get(7))

  interpolatedDataSpeeds = _prepData(dataPath, frameCount)

  X = np.zeros((batchSize, 3, height, width))
  Y = np.zeros((batchSize, 1))

  batch = 0
  for frameIdx, speed in enumerate(interpolatedDataSpeeds):
    ret, frame = camera.read()
    if ret is False: continue

    batchIndex = frameIdx%batchSize

    X[batchIndex] = np.rollaxis(frame, 2)  # roll the RGB dimension to the front
    Y[batchIndex] = speed

    # Draw stuff; it's assumed the speed data is in m/s
    _drawString(frame, (20, 20), "{:.2f} mph".format(speed*2.237))
    _drawSpeed(frame, speed)
    cv2.imshow("driving camera", frame)

    if (camera.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
      # End of video file, or user quit with 'ESC'
      break

    if batchIndex == 0 and frameIdx != 0:
      if verbosity > 0:
        print "Now yielding batch {} of epoch {}".format(batch, epoch)
      batch += 1
      yield X, Y


def _prepData(dataPath, numTimesteps, normalizeData=False):
  """
  Get and preprocess the ground truth drive speeds data.

  Args:
    dataPath: (str) Path to video file.
    numTimesteps: (int) Number of timesteps to interpolate the data to.
    normalizeData: (bool) Normalize the data to [0,1].
  Returns:
    (list) Speeds, one for each timestep.
  """
  with open(dataPath, "rb") as infile:
    driveData = json.load(infile)

  # Prep data: make sure it's in order, and use relative position (b/c seconds
  # values may be incorrect).
  driveData.sort(key = lambda x: x[0])
  dataSpeeds = np.array([d[1] for d in driveData])
  dataTimes = np.array([d[0] for d in driveData])
  dataPositions = ( (dataTimes - dataTimes.min()) /
                   (dataTimes.max() - dataTimes.min()) )
  if normalizeData:
    dataSpeeds = normalize(dataSpeeds)

  # Linearly interpolate data to the number of video frames.
  return np.interp(np.arange(0.0, 1.0, 1.0 / numTimesteps),
                   dataPositions,
                   dataSpeeds)


def _drawString(image, target, string):
  x, y = target
  cv2.putText(
      image, string, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
      thickness=2, lineType=cv2.LINE_AA)
  cv2.putText(
      image, string, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
      lineType=cv2.LINE_AA)


def _drawSpeed(image, speed, frameHeight=480, frameWidth=640):
  maxBars = 8
  numBars = min(int(speed/6)+1, maxBars)  # 6 m/s per speed bar
  for i in xrange(maxBars):
    overlay = image.copy()
    color = (10, 42*i, 42*(maxBars-1-i))  # BGR
    cv2.rectangle(
        overlay, (i*20, frameHeight-i*10),
        (frameWidth-i*20, frameHeight-(i+1)*10), color, thickness=-1)
    opacity = 0.08
    cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)
    if i <= numBars:
      # Shade bars to represent the speed
      opacity = 0.4
      cv2.addWeighted(overlay, opacity, image, 1-opacity, 0, image)


def normalize(vector):
  nump = np.array(vector)
  return (nump - nump.min()) / (nump.max() - nump.min())


def standardize(array):
  nump = np.array(array)
  return (nump - nump.mean()) / nump.std()



def runTrain(args):
  """
  Builds model (with global defaults for frame specs), trains it on the
  specified video stream, and saves the model weights and architecture.
  The model description is in buildModel().

  Args:
    (See the script's command line arguments.)
  Out:
    JSON and keras files of the saved model.
  """
  if args.loadModel:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, loadModel+".json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
    model.compile(optimizer="adam", loss="mse")
    model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, loadModel+".keras"))
    print "Model loaded from", modelPath
  else:
    model = buildModel((_VIDEO_CHANNELS, _VIDEO_HEIGHT, _VIDEO_WIDTH))

  logging = [LossHistory()] if args.verbosity else []

  # Setup dir for model checkpointing
  if not os.path.exists(_DEFAULT_MODEL_DIR):
    os.makedirs(_DEFAULT_MODEL_DIR)
  checkpointConfig = os.path.join(_DEFAULT_MODEL_DIR, "speed_checkpoint.json")
  checkpointWeights = os.path.join(_DEFAULT_MODEL_DIR, "speed_checkpoint.keras")

  print "Starting training..."
  for epoch in xrange(args.epochs):
    if args.verbosity > 0:
      print "\nTraining epoch {} of {}".format(epoch, args.epochs-1)
    for XBatch, YBatch in _frameGenerator(args.videoPath,
                                          args.dataPath,
                                          args.batchSize,
                                          epoch,
                                          verbosity=args.verbosity):
      history = model.fit(XBatch,
                YBatch,
                batch_size=args.batchSize,
                nb_epoch=1,
                callbacks=logging,
                validation_split=args.validation,
                verbose=0)
      if args.verbosity:
        print "Training loss =", history.history["loss"][0]

    # Checkpoint the model in case training breaks early
    if args.verbosity > 0:
      print "\nEpoch {} complete, checkpointing the model.".format(epoch)
    model.save_weights(checkpointWeights, True)
    with open(checkpointConfig, "w") as outfile:
      json.dump(model.to_json(), outfile)

  print "\nTraining complete, saving model weights and configuration files."
  model.save_weights(os.path.join(_DEFAULT_MODEL_DIR, "speed.keras"), True)
  with open(os.path.join(_DEFAULT_MODEL_DIR, "speed.json"), "w") as outfile:
    json.dump(model.to_json(), outfile)
  print "Model saved as .keras and .json files in", _DEFAULT_MODEL_DIR
  os.remove(checkpointWeights)
  os.remove(checkpointConfig)



def buildModel(cameraFormat=(3, 480, 640)):
  """
  Build and return a CNN; details in the comments.
  The intent is a scaled down version of the model from "End to End Learning
  for Self-Driving Cars": https://arxiv.org/abs/1604.07316.

  Args:
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, rows, columns).
  Returns:
    A compiled Keras model.
  """
  print "Building model..."
  ch, row, col = cameraFormat

  model = Sequential()

  # Use a lambda layer to normalize the input data
  model.add(Lambda(
      lambda x: x/127.5 - 1.,
      input_shape=(ch, row, col),
      output_shape=(ch, row, col))
  )

  # Several convolutional layers, each followed by ELU activation
  # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
  model.add(ELU())
  # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  # Flatten the input to the next layer
  model.add(Flatten())
  # Apply dropout to reduce overfitting
  model.add(Dropout(.2))
  model.add(ELU())
  # Fully connected layer
  model.add(Dense(512))
  # More dropout
  model.add(Dropout(.5))
  model.add(ELU())
  # Fully connected layer with one output dimension (representing the speed).
  model.add(Dense(1))

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  model.compile(optimizer="adam", loss="mse")

  return model



def runTest(videoPath, dataPath, verbosity=1):
  """
  Load a serialized model, run inference on a video stream, write out and
  compare the predictions to the given data labels.

  Args:
    videoPath: (str) Drive video for inference.
    dataPath: (str) JSON of truth data, as a list of [time, speed] items.
    verbosity: (int) Plot or not.
  Out:
    JSON of prediction results, as a list of [time, speed] items.
    Displays line plot of predicted and truth speeds.
  """
  try:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, "speed.json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
  except:
    print "Could not load a saved model JSON from", modelPath
    raise

  # Serialized to JSON only preserves model archtecture, so we need to recompile
  model.compile(optimizer="adam", loss="mse")
  model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, "speed.keras"))

  camera = cv2.VideoCapture(videoPath)

  # Run inference on video stream
  predictedSpeeds = []
  while camera.isOpened():
    ret, frame = camera.read()
    if ret is False: continue

    X = np.rollaxis(frame, 2)[None, :, :, :]
    predictedSpeed = model.predict(X)[0][0]

    frameSeconds = camera.get(0)/1000.0
    predictedSpeeds.append([frameSeconds, float(predictedSpeed)])

    # Draw stuff; it's assumed the speed data is in m/s
    _drawString(frame, (20, 20), "{:.2f} mph".format(predictedSpeed*2.237))
    _drawSpeed(frame, predictedSpeed)
    cv2.imshow("driving camera", frame)

    if (camera.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
      # End of video file, or user quit with 'ESC'
      break

  resultsPath = os.path.join(_DEFAULT_MODEL_DIR, "speed_test.json")
  with open(resultsPath, "w") as outfile:
    json.dump(predictedSpeeds, outfile)
  print "Test results written to", resultsPath

  # Calculate the root mean squared error. We expect the data labels to cover
  # the full video that the model just ran prediction on.
  interpolatedDataSpeeds = _prepData(dataPath, len(predictedSpeeds))
  numValues = len(interpolatedDataSpeeds)
  times = np.zeros(numValues)
  predictions = np.zeros(numValues)
  for i, (time, prediction) in enumerate(predictedSpeeds):
    times[i] = time
    predictions[i] = prediction

  rmse = ( np.linalg.norm(predictions - interpolatedDataSpeeds) /
           np.sqrt(numValues) )
  print "Finished testing, with a RMSE =", rmse

  if verbosity > 0:
    # Show line plot of results
    # TODO: use plotly (much better for data viz, but didn't want to add another
    #     dependency for this first version)
    plt.plot(times, interpolatedDataSpeeds)
    plt.plot(times, predictions)
    plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--videoPath",
                      type=str,
                      default="drive.mp4",
                      help="Path to drive video.")
  parser.add_argument("-d", "--dataPath",
                      type=str,
                      default="drive.json",
                      help="Path to JSON of ground truth speeds.")
  parser.add_argument("--batchSize",
                      type=int,
                      default=200,
                      help="Batch size.")
  parser.add_argument("-e", "--epochs",
                      type=int,
                      default=10,
                      help="Number of epochs.")
  parser.add_argument("--test",
                      default=False,
                      action="store_true",
                      help="Run test phase (using saved model).")
  parser.add_argument("--skipTraining",
                      default=False,
                      action="store_true",
                      help="Bypass training phase.")
  parser.add_argument("--validation",
                      type=float,
                      default=0.0,
                      help="Portion of training data for validation split.")
  parser.add_argument("--loadModel",
                      type=str,
                      default="",
                      help="Load a model to train.")
  parser.add_argument("--verbosity",
                      type=int,
                      default=1,
                      help="Level of printing stuff to the console.")
  args = parser.parse_args()

  if not args.skipTraining:
    runTrain(args)

  if args.test:
    runTest(args.videoPath, args.dataPath, args.verbosity)
