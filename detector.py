import numpy as np
import argparse
import time
import cv2
import os
import copy
import imutils

class Detector:
    def __init__(self,yoloFolder,confidence,threshold):
        # derive the paths to the YOLO weights and model configuration
        weightsPath = os.path.sep.join([yoloFolder, "yolov3-tiny.weights"])
        configPath = os.path.sep.join([yoloFolder, "yolov3-tiny.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        #print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        self.confidence=confidence
        self.threshold=threshold

    def detect(self,image):
        (H, W) = image.shape[:2]

        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i - 1] for i in self.net.getUnconnectedOutLayers().flatten()]

        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=False, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(ln)

        end = time.time()

        boxes = []
        confidences = []
        classIDs = []
        max_confidence = 0.0
        box = []

        # loop over each of the layer outputs
        for output in layerOutputs:
	        # loop over each of the detections
	        for detection in output:
		        # extract the class ID and confidence (i.e., probability) of
		        # the current object detection
		        scores = detection[5:]
		        classID = np.argmax(scores)
		        confidence = scores[classID]
		        # filter out weak predictions by ensuring the detected
		        # probability is greater than the minimum probability
		        if confidence > self.confidence:
			        # scale the bounding box coordinates back relative to the
			        # size of the image, keeping in mind that YOLO actually
			        # returns the center (x, y)-coordinates of the bounding
			        # box followed by the boxes' width and height
			        box = detection[0:4] * np.array([W, H, W, H])
			        (centerX, centerY, width, height) = box.astype("int")
			        # use the center (x, y)-coordinates to derive the top and
			        # and left corner of the bounding box
			        x = int(centerX - (width / 2))
			        y = int(centerY - (height / 2))
			        # update our list of bounding box coordinates, confidences,
			        # and class IDs
			        boxes.append([x, y, int(width), int(height)])
			        confidences.append(float(confidence))
			        classIDs.append(classID)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
        return boxes, idxs, classIDs, confidences


