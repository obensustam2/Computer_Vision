
import pyrealsense2 as rs
import numpy as np
import cv2

# DNN Initialization
classNames = []
classFile = '/home/oben/Computer Vision/Computer_Vision/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = '/home/oben/Computer Vision/Computer_Vision/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = '/home/oben/Computer Vision/Computer_Vision/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


# Object Detection Function
def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            x, y, w, h = box[0], box[1], box[2], box[3] # NEW
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (x + 200, y + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(color_image, (center_x, center_y), 4, (0, 0, 255))
                    distance = depth_image[center_y, center_x]
                    cv2.putText(color_image, "{}m".format(distance/1000), (box[0], box[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
    
    return img, objectInfo


if __name__ == "__main__":
    # Realsense Camera Initialization
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            depth_frame = frames.get_depth_frame()
            depth_image = np.asanyarray(depth_frame.get_data())
            result, objectInfo = getObjects(color_image, 0.6, 0.2, True, objects=['person'])
            print(objectInfo)
            cv2.imshow('RealSense', color_image)
            cv2.waitKey(1)

    finally:
        pipeline.stop()



