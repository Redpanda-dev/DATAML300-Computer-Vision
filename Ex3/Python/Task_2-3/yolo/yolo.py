import cv2
import numpy as np
import time
import os

# Load YOLO, pretrained with COCO dataset

# If you have GPU available, change "yolov3-tiny" to "yolov3" in both lines. 
# That way you get to use the heavier and more accurate version of YOLO.
# You have to also download the weights for this model, which can be found from 
# https://pjreddie.com/media/files/yolov3.weights
# Tiny YOLO model weights are already in the exercise folder.

cwd = os.getcwd()
weights = os.path.join(cwd, "yolov3-tiny.weights")
model_dir = os.path.join(cwd, "cfg/yolov3-tiny.cfg")

net = cv2.dnn.readNet(weights, model_dir)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
# Tiny-YOLOv3 network is not very good at detecting objects in unideal lighting conditions, 
# so using a video feed over webcam feed might be wise. 

# Use webcam feed with cv2.VideoCapture(0).
cap = cv2.VideoCapture("walking.mp4")
#cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame_id += 1
    
    # Image size
    height, width, channels = frame.shape 

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            #detection-array has the bound
            scores = detection[5:]
            # Create a new variable called class_id which takes in the index of the highest value 
            # from scores-array. 
            # Then set the highest value of the array to variable "confidence".
            
            ##-your-code-starts-here-##
            class_id = 0 # replace me
            confidence = 0 # replace me 
            ##your-code-ends-here
            
            if confidence > 0.5:
                pass ##-REMOVE-THIS-LINE-ONCE-YOU-HAVE-EVERYTHING-READY-##
                
                # Object detected
                # Examine the detection array. Its first four arguments include information 
                # about the bounding box limits, but the problem is that these values are between
                # 0 and 1. You need to scale these values up with width and height -arguments, which specify 
                # the size of the whole video image. The values of the four elements are in order 
                # (x-coordinate of the center of bounding box, y-coordinate of the center of bounding box, 
                # width of the bounding box, height of bounding box)
                # Create four variables, center_x, center_y, w, and h that include the scaled values. 
                # Round them to the nearest integer using int()
                
                ##-your-code-starts-here-##

                ##-your-code-ends-here-##
                
                # Using the values you calculated earlier, calculate the coordinates
                # for the top-left corner location of the bounding box
                # and store x-coordinate to variable x and y-coordinate to variable y.
                
                ##-your-code-starts-here-##

                ##-your-code-ends-here-##
                
                # Store x, y, w and h to boxes-list as a four element list,
                # confidence to confidences list and class_id to class_ids list.
                # All of these lists have already been declared above.
                
                ##-your-code-starts-here-##

                ##-your-code-ends-here-##          
                

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            # Draw the properly sized and colored bounding box with a thickness of 2 
            # using cv2.rectangle()-function and x, y, w, h.
            
            ##-your-code-starts-here-##
            
            ##-your-code-ends-here-##
            
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)


    elapsed_time = time.time() - starting_time
    fps = "{:.0f} FPS".format(frame_id/elapsed_time)
    cv2.putText(frame, fps, (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))
    cv2.putText(frame, 'Press q to quit', (1000, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    
    if key & 0xFF == ord('q'):
        break  # q to quit

cap.release()
cv2.destroyAllWindows()