"""
Utility functions for the models.
"""
import cv2
import json
import logging
import numpy as np
import os

from keras.callbacks import Callback



class VideoStream(object):
  """ Utilities class for using video data with OpenCV.
  Docs: http://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html
  """
  def __init__(self, videoIn):
    """ Init OpenCV video stream.

    Args:
      videoIn: (str) Path to mp4 video file.
    """
    self.videoSource = cv2.VideoCapture(videoIn)
    self.width = self.videoSource.get(3)
    self.height = self.videoSource.get(4)


  def getVideoInfo(self):
    return {"msTime": self.videoSource.get(0),
            "frameIndex": self.videoSource.get(1),
            "relativePosition": self.videoSource.get(2),
            "frameRate": self.videoSource.get(5),
            "frameCount": self.videoSource.get(7)
    }


  def setVideoPosition(self, position=0.0):
    self.videoSource.set(1, position)


  def rawFramesGenerator(self):
    """ Yields next video frame on each call."""
    while self.videoSource.isOpened():
      ret, frame = self.videoSource.read()
      if ret is False: continue
      if (self.videoSource.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
        # End of video file, or user quit with 'ESC'
        break
      yield frame


  def robustFramesGenerator(self, truthData, batchSize,
                            timesteps=10, speedVisuals=False, verbosity=0):
    """ Yields X and Y data (in batches). Each yield is the frame data extended
    in the time dimension, or a "volume", where there are batchSize / timesteps
    volumes per batch.

    Args:
      truthData: (list) Truth data, as a list of [time in seconds, value] items.
          We assume that items correspond to successive video frames.
      batchSize: (int) Number of consecutive frames per batch.
      timesteps: (int) Number of timesteps to accumulate in a volume.
      speedVisuals: (bool) Draw visuals over the video.
      verbosity: (int) Print detailed info or not.
    Yields:
      Two-tuple representing a batch of input video frames and target values.
    """
    if timesteps < 1:
      raise ValueError("Need more than 0 timesteps.")
    if batchSize%timesteps:
      raise ValueError(
          "Batch size should be divisible by timesteps so we get an equal "
          "number of frames in each portion of the batch.")

    # A "volume" is a video frame data struct extended in the time dimension.
    volumesPerBatch = batchSize / timesteps

    if timesteps > 1:
      # For recurrent architectures
      X = np.zeros((volumesPerBatch, timesteps, 3, self.height, self.width), dtype="uint8")
      Y = np.zeros((volumesPerBatch, timesteps, 1), dtype="float32")
    else:
      # For static models (no time dimension)
      X = np.zeros((batchSize, 3, self.height, self.width), dtype="uint8")
      Y = np.zeros((batchSize, 1), dtype="float32")

    if verbosity:
      print "Data shapes from the generator:"
      print "  X =", X.shape
      print "  Y =", Y.shape

    # Loop through video and accumulate data (over time for each batch)
    batchCount = 0
    volumeIndex = -1
    for frameIdx, value in enumerate(truthData):
      ret, frame = self.videoSource.read()
      if ret is False: continue

      # Update counters so we know where to allocate this frame in the data structs
      timeIndex = frameIdx%timesteps
      if timeIndex == 0:
        volumeIndex += 1

      # Populate data structs; in the frame we roll the RGB dimension to the front
      if timesteps > 1:
        X[volumeIndex][timeIndex, :, :, :] = np.rollaxis(frame, 2)
        Y[volumeIndex][timeIndex] = value
      else:
        batchIndex = frameIdx%batchSize
        X[batchIndex] = np.rollaxis(frame, 2)
        Y[batchIndex] = value

      if speedVisuals:
        # Draw stuff; it's assumed the speed data is in m/s
        self._drawString(frame, (20, 20), "{:.2f} mph".format(value*2.237))
        self._drawSpeed(frame, value)
      cv2.imshow("driving camera", frame)

      if (self.videoSource.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
        # End of video file, or user quit with 'ESC'
        break

      if volumeIndex==volumesPerBatch-1 and timeIndex==timesteps-1:
        # End of this batch
        if verbosity:
          print "Now yielding batch", batchCount
        batchCount += 1
        volumeIndex = 0
        yield X, Y


  @staticmethod
  def _drawString(image, target, string):
    x, y = target
    cv2.putText(
        image, string, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0),
        thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(
        image, string, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255),
        lineType=cv2.LINE_AA)


  @staticmethod
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



def prepTruthData(dataPath, numFrames, normalizeData=False):
  """
  Get and preprocess the ground truth drive speeds data.

  Args:
    dataPath: (str) Path to JSON of ground truths.
    numFrames: (int) Number of timesteps to interpolate the data to.
    normalizeData: (bool) Normalize the data to [0,1].
  Returns:
    (list) Linearly interpolated truth values, one for each timestep.
  """
  with open(dataPath, "rb") as infile:
    driveData = json.load(infile)

  # Prep data: make sure it's in order, and use relative position (b/c seconds
  # values may be incorrect)
  driveData.sort(key = lambda x: x[0])
  times = np.zeros(len(driveData))
  speeds = np.zeros(len(driveData))
  for i, (time, speed) in enumerate(driveData):
    times[i] = time
    speeds[i] = speed
  positions = (times - times.min()) / (times.max() - times.min())

  if normalizeData:
    speeds = normalize(speeds)

  # Linearly interpolate the data to the number of video frames
  return np.interp(np.arange(0.0, 1.0, 1.0/numFrames), positions, speeds)



def normalize(vector):
  nump = np.array(vector)
  return (nump - nump.min()) / (nump.max() - nump.min())



def standardize(array):
  nump = np.array(array)
  return (nump - nump.mean()) / nump.std()



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


