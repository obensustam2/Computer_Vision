import cv2
import numpy as np

thres = 0.6 # Threshold to detect object
nms_threshold = 0.3 #Non-Max suppression algorithm, lower value higher filter

# Camera data
cap = cv2.VideoCapture(0)
cap.set(3, 720) # Width
cap.set(4, 720) # Height
cap.set(10, 150) # Brightness

# Read coco file
classNames= []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Detection Model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Removing Dublicates NMS
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    bbox = list(bbox)
    confs = list(np.array(confs).reshape(1, -1)[0])
    confs = list(map(float, confs)) # confs class = float
    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold) # NMS Filtering

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # Rectangle & Text
        cv2.rectangle(img, (x, y), (x+w, h+y), color=(0, 255, 0), thickness=2)
        cv2.putText(img, classNames[classIds[i][0]-1].upper(), (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, str(confs * 100), (box[0] + 300, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Output", img)
    cv2.waitKey(1)
