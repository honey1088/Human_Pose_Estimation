import cv2
import numpy as np
import matplotlib.pyplot as plt

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9, "RAnkle": 10,
    "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14, "LEye": 15, "REar": 16, "LEar": 17,
    "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"], ["RElbow", "RWrist"],
    ["LShoulder", "LElbow"], ["LElbow", "LWrist"], ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"],
    ["Neck", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"], ["REye", "REar"],
    ["Nose", "LEye"], ["LEye", "LEar"]
]

width, height = 368, 368

# Load TensorFlow Model
net = cv2.dnn.readNetFromTensorflow("models/openpose/models/pose/coco/graph_opt.pb") 

thres = 0.2  # Confidence threshold

def detect_pose(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),
                                    (127.5, 127.5, 127.5), swapRB=True, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    points = []
    for i in range(len(BODY_PARTS) - 1):  # Ignoring "Background"
        heatMap = output[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / output.shape[3])
        y = int((frameHeight * point[1]) / output.shape[2])

        if conf > thres:
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
        else:
            points.append(None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 2, cv2.LINE_AA)

    return frame


# if __name__ == "__main__":
#     test_img = cv2.imread("stand.jpg") 
#     output_img = detect_pose(test_img)

#     cv2.imshow("Pose Estimation", output_img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()