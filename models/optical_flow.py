"""
Tools for optical flow tracking with OpenCV.
"""
import cv2
import numpy as np



class OpticalFlow(object):
  """ Module for optical flow tracking with feature point detection.
  """

  def __init__(self, videoSource, featurePtMask=None, verbosity=0):
    # cap the length of optical flow tracks
    self.maxTrackLength = 10

    # detect feature points in intervals of frames; adds robustness for
    # when feature points disappear.
    self.detectionInterval = 5

    # Params for Shi-Tomasi corner (feature point) detection
    self.featureParams = dict(
        maxCorners=500,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )
    # Params for Lucas-Kanade optical flow
    self.lkParams = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    # # Alternatively use a fast feature detector
    # self.fast = cv2.FastFeatureDetector_create(500)

    self.verbosity = verbosity

    (self.videoStream,
     self.width,
     self.height,
     self.featurePtMask) = self._initializeCamera(videoSource)


  def _initializeVideoStream(self, videoSource, featurePtMask=None):
    """
    Setup OpenCV VideoCapture object from the videoSource, and related
    member variables (including specifying the feature mask).
    """
    # Open the video stream and get the first frame.
    self.videoStream = cv2.VideoCapture(videoSource)
    ret, _ = self.videoStream.read()
    if ret is False:
      raise RuntimeError("Bad camera input! Unable to initialize.")

    print "Video stream opened, ready to run."

    self.width = self.videoStream.get(3)
    self.height = self.videoStream.get(4)

    # Use a mask to define where optical flow should be detected
    if featurePtMask:
      self.featurePtMask = featurePtMask
    else:
      # Defualt to the full frame
      self.featurePtMask = np.zeros((self.height, self.width), dtype=np.uint8)
      self.featurePtMask[:] = 255

    if self.verbosity:
      # Video stream stats
      print "Dimensions (w,h) = {},{}".format(self.width, self.height)
      print "FPS = {}".format(self.videoStream.get(5))


  def _rewind(self, relativePosition=0.0):
    """ Rewind video to a relative position 0-1.
    """
    self.videoStream.set(1, relativePosition)


  def runAll(self):
    """
    Run through the video stream. Must initialize the camera first.

    For each frame, we get the points of the optical flow tracks, calculate
    their spatial changes from the last frame ("deltas"), draw their tracking
    lines, and re-detect feature points (every few frames).

    NOTE: The "delta" logic is commented out b/c is accumulates a list of lists,
    which could be computationally burdensome. Uncomment it below if you'd like
    to return streamOfDeltas; you could also modify this to yield the delta
    lists at each frame step.
    """
    # A track is a list of (x,y) pixel coordinates that define the path of a
    # feature point over successive video frames.
    tracks = []

    # # "Deltas" track the location changes of feature pts over successive frames,
    # # and streamOfDeltas accumulates the (frame time, deltas) pairs.
    # streamOfDeltas = []

    prevGreyFrame = None
    frameIndex = 0
    while self.videoStream.isOpened():
      # Read in this frame, and get the grey mask of it
      ret, frame = self.videoStream.read()
      if ret is False:
        continue
      frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Use a frame copy for drawing
      frameVis = frame.copy()

      if len(tracks) > 0:
        # Points are the (x,y) pixel coords of the optical flow tracks. Use
        # the most recent track points (old) to calculate the (new) points
        # for this frame.
        pointsOld = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        pointsNew, statuses, _ = cv2.calcOpticalFlowPyrLK(
            prevGreyFrame, frameGrey, pointsOld, None, **self.lkParams)

        # Use back-tracking to verify feature point matches between frames, and
        # the status to ignore "lost" points.
        pointsOldReversed, _, _ = cv2.calcOpticalFlowPyrLK(
            frameGrey, prevGreyFrame, pointsNew, None, **self.lkParams)
        maxDiffs = abs(pointsOld-pointsOldReversed).reshape(-1, 2).max(-1)
        goodPoints = np.logical_and(maxDiffs < 1, statuses.reshape(maxDiffs.shape[0]))

        # Update optical flow tracks, track their changes (deltas)
        # deltas = []
        updatedTracks = []
        for track, newPt, good in zip(tracks, pointsNew.reshape(-1, 2), goodPoints):
          if not good:
            # Bad track, ignore it
            continue
          if not _verifyPtWithinDomain(np.uint32(track[-1])):
            continue
          # Calculate the delta btwn this pt and the last pt in the track
          # if len(track) > 0:
            # # Get this track's delta -- pixel distance from previous frame
            # deltas.append(np.linalg.norm(newPt-track[-1]))
          # Add this pt to the optical flow track
          x, y = newPt[0], newPt[1]
          track.append((x, y))
          if len(track) > self.maxTrackLength:
            # Only maintain the recent portion of the tracks (for drawing)
            del track[0]
          # Draw a circle for this optical flow pt.
          cv2.circle(frameVis, (x, y), 2, (0, 255, 0), -1)
          updatedTracks.append(track)

        tracks = updatedTracks
        # frameSeconds = self.videoStream.get(0)/1000.0
        # streamOfDeltas.append([frameSeconds, deltas])

        # Draw multiple lines
        cv2.polylines(frameVis, [np.int32(track) for track in tracks], False, (0, 255, 0))

      # Detect feature pts every few frames
      if frameIndex % self.detectionInterval == 0:
        # Draw a circle for each feature pt
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
          cv2.circle(self.featurePtMask, (x, y), 5, 0, -1)
        # Detect corners with Shi-Tomasi algorithm
        featurePts = cv2.goodFeaturesToTrack(
            frameGrey, mask=self.featurePtMask, **self.featureParams)
        # featurePts = self.fast.detect(frameGrey, None)
        if featurePts is not None:
          # Add feature points to the optical flow tracks

          for featureCoordinates in featurePts:  # TODO: do this better with itertools?
            x = featureCoordinates[0][0]  # ShiTomasi
            y = featureCoordinates[0][1]  # ShiTomasi
            # x = featureCoordinates.pt[0]  # fast features
            # y = featureCoordinates.pt[1]  # fast features
            tracks.append([(x, y)])

      frameIndex += 1
      prevGreyFrame = frameGrey
      cv2.imshow("tracked_frame", frameVis)

      if (self.videoSource.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
        # End of video file, or user quit with 'ESC'
        break

    self.videoStream.release()
    cv2.destroyAllWindows()

    # TODO: refactoring to be a generator would be better!
    # return streamOfDeltas


  def runGenerator(self):
    """ Iterate through the video stream, yielding an optical flow mask for each
    frame. For each frame, we get the points of the optical flow tracks,
    calculate their spatial changes from the last frame ("deltas"), draw their
    tracking lines, and re-detect feature points (every few frames).

    Yields:
      List of optical flow tracks, each a list of (x,y) coords.
    """
    tracks = []
    frameIndex = 0
    while self.videoStream.isOpened():
      # Read in this frame, and get the grey mask of it
      ret, frame = self.videoStream.read()
      if ret is False:
        continue
      frameGrey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # Use a frame copy for drawing
      frameVis = frame.copy()

      if len(tracks) > 0:
        # Points are the (x,y) pixel coords of the optical flow tracks. Use
        # the most recent track points (old) to calculate the (new) points
        # for this frame.
        pointsOld = np.float32([tr[-1] for tr in tracks]).reshape(-1, 1, 2)
        pointsNew, statuses, _ = cv2.calcOpticalFlowPyrLK(
            prevGreyFrame, frameGrey, pointsOld, None, **self.lkParams)

        # Use back-tracking to verify feature point matches between frames, and
        # the status to ignore "lost" points.
        pointsOldReversed, _, _ = cv2.calcOpticalFlowPyrLK(
            frameGrey, prevGreyFrame, pointsNew, None, **self.lkParams)
        maxDiffs = abs(pointsOld-pointsOldReversed).reshape(-1, 2).max(-1)
        goodPoints = np.logical_and(maxDiffs < 1, statuses.reshape(maxDiffs.shape[0]))

        # Update optical flow tracks, track their changes (deltas)
        # deltas = []
        updatedTracks = []
        for track, newPt, good in zip(tracks, pointsNew.reshape(-1, 2), goodPoints):
          if not good:
            # Bad track, ignore it
            continue
          if not _verifyPtWithinDomain(np.uint32(track[-1])):
            continue
          # Add this pt to the optical flow track
          x, y = newPt[0], newPt[1]
          track.append((x, y))
          if len(track) > self.maxTrackLength:
            # Only maintain the recent portion of the tracks (for drawing)
            del track[0]
          # Draw a circle for this optical flow pt.
          cv2.circle(frameVis, (x, y), 2, (0, 255, 0), -1)
          updatedTracks.append(track)

        tracks = updatedTracks

        # Draw multiple lines
        cv2.polylines(frameVis, [np.int32(track) for track in tracks], False, (0, 255, 0))

      # Detect feature pts every few frames
      if frameIndex % self.detectionInterval == 0:
        # Draw a circle for each feature pt
        for x, y in [np.int32(tr[-1]) for tr in tracks]:
          cv2.circle(self.featurePtMask, (x, y), 5, 0, -1)
        # Detect corners with Shi-Tomasi algorithm
        featurePts = cv2.goodFeaturesToTrack(
            frameGrey, mask=self.featurePtMask, **self.featureParams)
        # featurePts = self.fast.detect(frameGrey, None)
        if featurePts is not None:
          # Add feature points to the optical flow tracks

          for featureCoordinates in featurePts:  # TODO: do this better with itertools?
            x = featureCoordinates[0][0]  # ShiTomasi
            y = featureCoordinates[0][1]  # ShiTomasi
            # x = featureCoordinates.pt[0]  # fast features
            # y = featureCoordinates.pt[1]  # fast features
            tracks.append([(x, y)])

      frameIndex += 1
      prevGreyFrame = frameGrey
      cv2.imshow("tracked_frame", frameVis)

      if (self.videoSource.get(2) == 1) or (cv2.waitKey(1) & 0xFF==27):
        # End of video file, or user quit with 'ESC'
        break

      # TODO: better to yield just the most recent pt of each track
      yield tracks

    self.videoStream.release()
    cv2.destroyAllWindows()


  def _verifyPtWithinDomain(self, trackPt):
    """ Returns False if the track point is (projected) out of the video frame
    or the domain mask.
    """
    if (trackPt[0]>self.height-1 or trackPt[0]<0) or
       (trackPt[1]>self.width-1 or trackPt[1]<0):
      # Out of bounds
      return False
    # (Below is commented out b/c we may want to allow pts to move beyond the domain.)
    # if self.featurePtMask[(trackPt[0], trackPt[1])] == 0:
    #   # Track is outside the predefined feature point domain
    #   return False
    return True
