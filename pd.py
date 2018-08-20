import os
import re
import time
from   datetime import datetime
from   pytz import timezone
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from   skimage import data, color
from   skimage.transform import resize
import imageio
import dapy
from io  import BytesIO
from PIL import Image
import logging
import configparser

if not os.path.isfile('config.ini'):
  print("\n\n\nFile config.ini is missing\n")
  exit(1)

config = configparser.ConfigParser()
config.read('config.ini')
if 'main' not in config.sections():
  print("\n\nno main section in config.ini\n")
  exit(1)

requiredParams = {'ipaddr', 'password', 'timezone', 'username'}
missingParams  = requiredParams - set(config.options('main'))
if missingParams:
  print("\n\n=== The following parameters are missing from config.ini:")
  print(*missingParams)
  print("\n")
  exit(1)

ipAddr   = config.get('main', 'ipAddr')
tz       = config.get('main', 'timezone')
username = config.get('main', 'username')
password = config.get('main', 'password')

logging.basicConfig(filename='det.log', level=logging.DEBUG)

# This will be used to create image output filenames based on current date/time
fmt = '%Y-%m-%d_%H:%M:%S.%f'
pacific = timezone(tz)

stream = 'rtsp://' + username + ':' + password + '@' + ipAddr + ':554:/cam/realmonitor?channel=1&subtype=1'
logging.debug('stream address=' + stream)
cap = cv2.VideoCapture(stream)
fps = cap.get(cv2.CAP_PROP_FPS)
logging.debug("fps=" + str(fps))
frameTime = 1/fps
minFrameTime = (0.8 * frameTime) * 1000

# This will filter out false positives specific to this camera. The flower in the front
# is sometimes classified as a person. Filter out very short objects that don't start below the
# image
def myFilter(box):
    height = box[2] - box[0]
    bottom = box[2]
    return bottom > 700 or height > 150

def drawRectangle(img, coordinates, color):
    (y1, x1, y2, x2) = coordinates
    img[y1:y1+3, x1:x2] = color
    img[y2:y2+3, x1:x2] = color
    img[y1:y2, x1:x1+3] = color
    img[y1:y2, x2:x2+3] = color

def readImageFromCamera():
  global cap
  # Flush frame buffer from old frames. A new frame takes at least 1/fps time to retrieve.
  while True:
    try:
      tick = time.time()
      r = cap.grab()
      tock = time.time()
    except:
      logging.info("bad response from camera")
      return False
    delta = (tock - tick) * 1000
    logging.debug("r=" + str(r) + " delta=" + str(delta))
    if delta > minFrameTime:
      break
    r, img = cap.retrieve()
  if not r:
    return False
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  logging.debug(img.shape)      # (1296, 2304, 3)
  img = cv2.resize(img, (1280, 720))

  return img

def makeOutputFileNames():
  global fmt
  global pacific
  loc_dt = datetime.now(pacific)
  fn = loc_dt.strftime(fmt) + '.jpg'
  positive = os.path.join('images', 'positive', fn)
  marked   = os.path.join('images', 'marked', fn)
  return positive, marked
  
def processImage():
  global odapi
  threshold = 0.7

  img = readImageFromCamera()
  if type(img) == bool and img == False:
    return

  tick = time.time()
  boxes, scores, classes, num = odapi.processFrame(img)
  tock = time.time()
  logging.debug("processFrame Time:" + str(tock-tick))
  found = [boxes[i] for i in range(len(boxes)) if classes[i] == 1 and scores[i] > threshold]
  logging.debug("found           =" + str(found))
  found = list(filter(myFilter, found))
  logging.debug("found (filtered)=" + str(found))
  if len(found):
    positive, marked = makeOutputFileNames()
    logging.debug(positive)
    logging.debug(marked)
    imageio.imwrite(positive, img)

    for i in range(len(boxes)):
      if classes[i] == 1 and scores[i] > threshold:
        box = boxes[i]
        drawRectangle(img, box, (255,0,0))

    imageio.imwrite(marked, img)
  
model_path = 'frozen_inference_graph.pb'
odapi = dapy.DetectorAPI(path_to_ckpt=model_path)

#for _ in range(20):
while True:
  processImage()


