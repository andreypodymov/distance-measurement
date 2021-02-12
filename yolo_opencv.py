import cv2
import argparse
import numpy as np
import distance_to_camera

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=False,
                help='path to input image')
ap.add_argument('-c', '--config', required=True,
                help='path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help='path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help='path to text file containing class names')
args = ap.parse_args()


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


classes = None

with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# open webcam
videoStream = cv2.VideoCapture(args.video)
totalFrames = int(videoStream.get(cv2.CAP_PROP_FRAME_COUNT))
print(totalFrames)

if not videoStream.isOpened():
    print("Could not open steam")
    exit()

focalLength = distance_to_camera.initialize()
frameCount = 0
img_array = []

# loop through frames
while videoStream.isOpened():

    # read frame from webcam
    status, image = videoStream.read()

    if not status:
        print("Next frame status == false")
        break

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

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        distance_to_camera.find_distance(image, box, focalLength)

    height, width, layers = image.shape
    size = (width, height)
    img_array.append(image)
    print("Processed: " + str(frameCount) + "/" + str(totalFrames) + " frames")
    frameCount += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

videoStream.release()
cv2.destroyAllWindows()
# release resources


