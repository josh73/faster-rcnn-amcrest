# faster-rcnn-amcrest
* Retrieve rtsp stream from an Amcrest IP camera
* Performs people detection on the frames
* Saves frames containing people in image/positive
* Saves the same frames with bounding boxes in images/marked

## preparation
* install tensorflow and cv2
* Create directories images/positive and images/marked in the current working directory
* Copy config.ini.template to config.ini and edit the 4 parameters.

## running
python pd.py
