#!/usr/bin/env python
"""
Vehicle speed prediction model.

Example invocations:

  # Run verbose training
  ./cnn_lstm_prediction.py --verbosity 2

  # Only run the test phase
  ./cnn_lstm_prediction.py --skipTraining --test

  # Continue training a checkpointed model
  ./cnn_lstm_prediction.py --loadModel speed_checkpoint

Some suggestions for improving the model are to tune the optmizer learning rate
and other params (currently using Keras standards), or more/larger layers
(perhaps like the model here: https://arxiv.org/abs/1604.07316). Depending on
the input video stream, further preprocessing can be done; there are good tools
in OpenCV.
"""
import argparse
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pprint import pprint
from prettytable import PrettyTable

from keras.models import model_from_json, Sequential
from keras.layers import Activation, Dense, Dropout, Flatten, Lambda, LSTM
from keras.layers.convolutional import Convolution2D
from keras.layers.wrappers import TimeDistributed

from support.utils import VideoStream, prepTruthData, LossHistory


_DEFAULT_MODEL_DIR = "./outputs/speed_model"

# Model architecture is based on video defaults:
_VIDEO_CHANNELS = 3
_VIDEO_HEIGHT = 480
_VIDEO_WIDTH = 640

logging.basicConfig(level=logging.INFO)



def setupExperiment(videoPath, dataPath):
  """ Setup the data -- create VideoStream object and preprocess the truth data.

  Args:
    videoPath: (str) Path to mp4 video file.
    dataPath: (str) Path to JSON of truth data.
  Returns:
    videoPath: (VideoStream)
    truthData: (list) Preprocessed truth data values.
  """
  # Open video stream
  videoStream = initVideoStream(videoPath)
  videoInfo = videoStream.getVideoInfo()

  # Get and prep training data
  truthData = prepTruthData(dataPath, videoInfo["frameCount"])

  return videoStream, truthData



def initVideoStream(videoIn):
  """
  Args:
    videoIn: (str) Path to mp4 video file.
  Returns:
    VideoStream object
  """
  videoStream = VideoStream(videoIn)

  if videoStream.width != _VIDEO_WIDTH or videoStream.height != _VIDEO_HEIGHT:
    raise ValueError(
        "Video stream dimensions do not match expected width and/or height")

  return videoStream



def buildModel(volumesPerBatch, timesteps, cameraFormat=(3, 480, 640), verbosity=0):
  """
  Build and return a CNN + LSTM model; details in the comments.

  The model expects batch_input_shape =
  (volumes per batch, timesteps per volume, (camera format 3-tuple))

  A "volume" is a video frame data struct extended in the time dimension.

  Args:
    volumesPerBatch: (int) batch size / timesteps
    timesteps: (int) Number of timesteps per volume.
    cameraFormat: (3-tuple) Ints to specify the input dimensions (color
        channels, height, width).
    verbosity: (int) Print model config.
  Returns:
    A compiled Keras model.
  """
  print "Building model..."
  ch, row, col = cameraFormat

  model = Sequential()

  if timesteps == 1:
    raise ValueError("Not supported w/ TimeDistributed layers")

  # Use a lambda layer to normalize the input data
  # It's necessary to specify batch_input_shape in the first layer in order to
  # have stateful recurrent layers later
  model.add(Lambda(
      lambda x: x/127.5 - 1.,
      batch_input_shape=(volumesPerBatch, timesteps, ch, row, col),
      )
  )

  # For CNN layers, weights are initialized with Gaussian scaled by fan-in and
  # activation is via ReLU units; this is current best practice (He et al., 2014)

  # Several convolutional layers, each followed by ELU activation
  # 8x8 convolution (kernel) with 4x4 stride over 16 output filters
  model.add(TimeDistributed(
      Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", init="he_normal")))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 32 output filters
  model.add(TimeDistributed(
      Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal")))
  model.add(Activation("relu"))
  # 5x5 convolution (kernel) with 2x2 stride over 64 output filters
  model.add(TimeDistributed(
      Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal")))
  # TODO: Add a max pooling layer?

  # Flatten the input to the next layer; output shape = (None, 76800)
  model.add(TimeDistributed(Flatten()))
  # Apply dropout to reduce overfitting
  model.add(Dropout(.2))
  model.add(Activation("relu"))

  # Fully connected layer
  model.add(TimeDistributed(Dense(512)))
  # More dropout
  model.add(Dropout(.2))
  model.add(Activation("relu"))

  # Add stacked (stateful) LSTM layers
  model.add(LSTM(512,
                 return_sequences=True,
                 batch_input_shape=(volumesPerBatch, timesteps, 512),
                 stateful=True))
  model.add(LSTM(512,
                 return_sequences=True,
                 stateful=True))

  # Fully connected layer with one output dimension (representing the predicted
  # value).
  model.add(TimeDistributed(Dense(1)))

  # Adam optimizer is a standard, efficient SGD optimization method
  # Loss function is mean squared error, standard for regression problems
  model.compile(optimizer="adam", loss="mse")

  if verbosity:
    printTemplate = PrettyTable(["Layer", "Input Shape", "Output Shape"])
    printTemplate.align = "l"
    printTemplate.header_style = "upper"
    for layer in model.layers:
      printTemplate.add_row([layer.name, layer.input_shape, layer.output_shape])
    print printTemplate

  if verbosity > 1:
    config = model.get_config()
    for layerSpecs in config:
      pprint(layerSpecs)

  return model



def _runTrain(args, videoStream, truthData):
  """
  Builds or loads a model (with global defaults for frame specs), trains it on
  the specified video stream, and saves the model weights and architecture.

  Args:
    (See the script's command line arguments.)
    videoStream: (VideoStream) The generator is used to yield X and Y training data batches.
    truthData: (list) Ground truth data expected by the generator.
  Out:
    JSON and keras files of the saved model.
  """
  if args.validation != 0.0 and args.validation != 0.5:
    raise ValueError("Only 0.0 and 0.5 validation splits are supported for "
        "stateful models (b/c the batch_input_shape changes between training "
        "and validation data.")

  volumesPerBatch = args.batchSize / args.timesteps
  trainingBatchSize = volumesPerBatch - (volumesPerBatch * args.validation)
  if args.loadModel:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, args.loadModel+".json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
    model.compile(optimizer="adam", loss="mse")
    model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, args.loadModel+".keras"))
    print "Model loaded from", modelPath
  else:
    model = buildModel(volumesPerBatch=trainingBatchSize,
                       timesteps=args.timesteps,
                       cameraFormat=(_VIDEO_CHANNELS, _VIDEO_HEIGHT, _VIDEO_WIDTH),
                       verbosity=args.verbosity)

  logging = [LossHistory()] if args.verbosity else []

  # Setup dir for model checkpointing
  if not os.path.exists(_DEFAULT_MODEL_DIR):
    os.makedirs(_DEFAULT_MODEL_DIR)
  checkpointConfig = os.path.join(_DEFAULT_MODEL_DIR, "speed_checkpoint.json")
  checkpointWeights = os.path.join(_DEFAULT_MODEL_DIR, "speed_checkpoint.keras")

  print "Starting training..."
  for epoch in xrange(args.epochs):
    # Iterate through epochs explicitly b/c Keras fit_generator doesn't yield data as expected
    if args.verbosity > 0:
      print "\nTraining epoch {} of {}".format(epoch, args.epochs-1)

    # Reset the video and generator for this epoch
    videoStream.setVideoPosition(0.0)
    frameGen = videoStream.robustFramesGenerator

    # Get video data; here batch size is the number of video frames in the batch
    for i, (XBatch, YBatch) in enumerate(frameGen(truthData,
                                                  args.batchSize,
                                                  timesteps=args.timesteps,
                                                  speedVisuals=True,
                                                  verbosity=args.verbosity)):

      # Train the model on this batch
      history = model.fit(  # TODO: decay learning rate
                XBatch,
                YBatch,
                batch_size=volumesPerBatch,
                shuffle=False,
                nb_epoch=1,
                callbacks=logging,
                validation_split=args.validation,
                verbose=0)
      if args.verbosity:
        # Not using Keras's training loss print out b/c it always shows epoch 0
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



def _runTest(args, frameGen, truthData):
  """
  Load a serialized model, run inference on a video stream, write out and
  compare the predictions to the given data labels.

  Args:
    (See the script's command line arguments.)
    frameGen: (generator) Yields X and Y data batches to train on.
    truthData: (list) Ground truth data expected by the generator.
  Out:
    JSON of prediction results, as a list of [time, speed] items.
    Displays line plot of predicted and truth speeds.
  """
  modelName = args.loadModel if args.loadModel else "speed"
  try:
    modelPath = os.path.join(_DEFAULT_MODEL_DIR, modelName+".json")
    with open(modelPath, "r") as infile:
      model = model_from_json(json.load(infile))
  except:
    print "Could not load a saved model JSON from", modelPath
    raise

  # Serialized to JSON only preserves model archtecture, so we need to recompile
  model.compile(optimizer="adam", loss="mse")
  model.load_weights(os.path.join(_DEFAULT_MODEL_DIR, modelName+".keras"))

  # Run inference
  predictedSpeeds = []  # TODO: preallocate arrays for len(truthData)
  truthSpeeds = []
  for i, (XBatch, YBatch) in enumerate(frameGen(truthData,
                                           args.batchSize,
                                           timesteps=args.timesteps,
                                           speedVisuals=True,
                                           verbosity=args.verbosity)):
    predictedSpeeds.extend(model.predict(XBatch).flatten().astype(float))
    truthSpeeds.extend(YBatch.flatten())

  resultsPath = os.path.join(_DEFAULT_MODEL_DIR, "speed_test.json")
  with open(resultsPath, "w") as outfile:
    json.dump(list(predictedSpeeds), outfile)
  print "Test results written to", resultsPath

  # Calculate the root mean squared error. We expect the data labels to cover
  # the full video that the model just ran prediction on.
  rmse = ( np.linalg.norm(predictedSpeeds - truthSpeeds) /
           np.sqrt(len(truthSpeeds)) )
  print "Finished testing, with a RMSE =", rmse

  if args.verbosity > 0:
    # Show line plot of results
    # TODO: use plotly
    plt.plot(truthSpeeds)
    plt.plot(predictedSpeeds)
    plt.show()



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("-v", "--videoPath",
                      type=str,
                      default="data/drive.mp4",
                      help="Path to drive video.")
  parser.add_argument("-d", "--dataPath",
                      type=str,
                      default="data/drive.json",
                      help="Path to JSON of ground truth speeds.")
  parser.add_argument("--batchSize",
                      type=int,
                      default=200,
                      help="Frames per batch yielded by the data generator.")
  parser.add_argument("-e", "--epochs",
                      type=int,
                      default=10,
                      help="Number of epochs.")
  parser.add_argument("-t", "--timesteps",
                      type=int,
                      default=10,
                      help="Number of consecutive video frames per CNN volume.")
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
                      help="Load a specific model to train or test.")
  parser.add_argument("--verbosity",
                      type=int,
                      default=1,
                      help="Level of printing stuff to the console.")
  args = parser.parse_args()

  # Train or test; experiment setup is done for both b/c we need to reset data
  # objects if running test immediately after training (on the same data).
  if not args.skipTraining:
    videoStream, truthData = setupExperiment(args.videoPath, args.dataPath)
    _runTrain(args, videoStream, truthData)
  if args.test:
    videoStream, truthData = setupExperiment(args.videoPath, args.dataPath)
    _runTest(args, videoStream.robustFramesGenerator, truthData)
