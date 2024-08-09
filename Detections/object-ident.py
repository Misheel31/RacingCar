import cv2
import numpy as np

# Load class names from file
classNames = []
classFile = "C:/Users/acer/Desktop/Detections/Detections/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# Configuration paths for the model
configPath = "C:/Users/acer/Desktop/Detections/Detections/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "C:/Users/acer/Desktop/Detections/Detections/frozen_inference_graph.pb"

# Load the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def getObjects(img, thres, nms, draw=True, objects=[]):
    """
    Perform object detection on the input image and return the image with bounding boxes and object information.

    Parameters:
    - img: Input image array (numpy array).
    - thres: Confidence threshold to filter detections.
    - nms: Non-maximum suppression threshold.
    - draw: Flag to draw bounding boxes and labels on the image.
    - objects: List of specific object classes to detect (empty list to detect all classes).

    Returns:
    - img: Image with bounding boxes and labels drawn.
    - objectInfo: List of detected objects with their bounding boxes and class names.
    """
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if len(objects) == 0 or className in objects:
                objectInfo.append([box, className])

                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    return img, objectInfo


def get_distance(bbox1, bbox2):
    """
    Calculate the Euclidean distance between the centers of two bounding boxes.

    Parameters:
    - bbox1: First bounding box coordinates in format [xmin, ymin, xmax, ymax].
    - bbox2: Second bounding box coordinates in format [xmin, ymin, xmax, ymax].

    Returns:
    - distance: Euclidean distance between the centers of the bounding boxes.
    """
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    distance = np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)
    return distance


def add_warning(img, bbox, text):
    """
    Add a warning message near the bounding box on the image.

    Parameters:
    - img: Input image array (numpy array).
    - bbox: Bounding box coordinates in format [xmin, ymin, xmax, ymax].
    - text: Warning text to display.

    Returns:
    - img: Image with warning text added near the bounding box.
    """
    cv2.putText(img, text, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    while True:
        success, img = cap.read()

        # Perform object detection
        result, objectInfo = getObjects(img, 0.45, 0.2)

        # Measure distance between detected objects and add warnings
        if len(objectInfo) >= 2:
            for i in range(len(objectInfo)):
                for j in range(i + 1, len(objectInfo)):
                    bbox1 = objectInfo[i][0]
                    bbox2 = objectInfo[j][0]
                    dist = get_distance(bbox1, bbox2)

                    # Add warning if distance is too close (adjust threshold as needed)
                    if dist < 100:  # Example threshold for warning
                        img = add_warning(img, bbox1, 'WARNING!')
                        img = add_warning(img, bbox2, 'WARNING!')

                    cv2.putText(img, f'{dist:.2f} units', (bbox1[0], bbox1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Display the image with detections and distances
        cv2.imshow("Output", img)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
