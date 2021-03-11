import cv2
import argparse
import numpy as np
import distance_to_camera
from centroidtracker import CentroidTracker

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=False,
                help='path to input image')
ap.add_argument('-i', '--image', required=False,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()

ct = CentroidTracker()

def draw_arrow(image, start_point, end_point):
    # Start coordinate, here (0, 0)
    # represents the top left corner of image

    # Green color in BGR
    color = (0, 255, 0)

    # Line thickness of 9 px
    thickness = 6

    # Using cv2.arrowedLine() method
    # Draw a diagonal green arrow line
    # with thickness of 9 px
    image = cv2.arrowedLine(image, start_point, end_point,
                            color, thickness)
    return image


def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    if (class_id == 0):
        color = (0, 255, 255)
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def analyzeFrame(image):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)

                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    old = ct.objects.copy()
    ct.update(boxes)

    for i in indices:
        i = i[0]

        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        distance_to_camera.find_distance(image, box, focalLength)

    for key, value in ct.objects.items():
        newBox = ct.objects.get(key)
        oldBox = old.get(key)
        directionX = "-"
        directionY = "-"
        speed = 0
        cv2.circle(image, (newBox[0], newBox[1]), 3, (0,0,255), 6)
        if oldBox is not None and newBox is not None:

            if newBox[0] > oldBox[0]:
                directionX = "RIGHT"
                speed = 1
            if newBox[0] < oldBox[0]:
                speed = 1
                directionX = "LEFT"

            if newBox[1] > oldBox[1]:
                directionY = "DOWN"
                speed = 1
            if newBox[1] < oldBox[1]:
                directionY = "UP"
                speed = 1

            cv2.line(image, (oldBox[0], oldBox[1]), (newBox[0], newBox[1]), (0, 255, 0), 10)

        cv2.putText(image, "Direction: " + directionX+"/"+directionY, (round(newBox[0]), 25 + round(newBox[1] + int(newBox[3] * 0.5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Speed: " + str(speed), (round(newBox[0]), 40 + round(newBox[1] + int(newBox[3] * 0.5))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

if (args.video == None and args.image == None):
    print("Error: --image or --video argument is not present")
    exit(1)

focalLength = distance_to_camera.initialize()

if (args.video != None):
    videoStream = cv2.VideoCapture(args.video)
    totalFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
    if not videoStream.isOpened():
        print("Could not open steam")
        exit()

    frameCount = 0
    img_array = []
    # loop through frames
    while videoStream.isOpened():
        status, image = videoStream.read()
        if not status:
            print("Next frame status == false")
            break
        analyzed_image = analyzeFrame(image)
        cv2.imshow("Frame", analyzed_image)
        img_array.append(analyzed_image)
        height, width, layers = analyzed_image.shape
        size = (width, height)
        print("Processed: " + str(frameCount) + "/" + str(totalFrames) + " frames")
        frameCount += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()

    videoStream.release()
elif (args.image != None):
    image = cv2.imread(args.image)
    analyzed_image = analyzeFrame(image)
    cv2.imshow("Frame", analyzed_image)
    cv2.imwrite("output.jpg", analyzed_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
