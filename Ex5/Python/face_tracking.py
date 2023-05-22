import cv2
import time
import traceback
import numpy as np


def get_delay(start_time, fps=30):
    if (time.time() - start_time) > (1 / float(fps)):
        return 1
    else:
        return max(int((1 / float(fps)) * 1000 - (time.time() - start) * 1000), 1)


# Instantiate cascade classifiers for finding faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Camera instance
#cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture('visionface.avi')  # uncomment if you want to use a video file instead

# Check if instantiation was successful
if not cam.isOpened():
    raise Exception("Could not open camera/file")
    

# USE OPENCV DOCUMENTATION TO FIND OUT HOW CERTAIN FUNCTIONS WORK.
# Your task is to implement real-time face point tracking.
# A few tips:
#  You should start by implementing the detection part first.
#  Try drawing the trackable points in the detection part without saving them
#  to p0 so you're able to see if the point coordinates are correct.
#  When finding the good points in the tracking part, use isFound as an index
#  for telling if the point is valid. (you may have to convert this to a boolean array first).

gray_prev = None  # previous frame
p0 = []  # previous points

while True:
    start = time.time()
    try:            
        # Get a single frame
        ret_val, img = cam.read()
        if not ret_val:
            cam.set(cv2.CAP_PROP_POS_FRAMES, 0)  # restart video
            gray_prev = None  # previous frame
            p0 = []  # previous points
            continue

        else:
            # Mirror
            img = cv2.flip(img, 1)
            
            # Grayscale copy
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if len(p0) <= 10:
                # Detection
                img = cv2.putText(img, 'Detection', (0,20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,255,255))

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                
                # Take the first face and get trackable points.
                if len(faces) != 0:
                    # Extract ROI (face) from the grayscale frame
                    # Detections are in the form
                    # (x_upperleft, y_upperleft, width, height)
                    # You can also crop this ROI even more to make sure only 
                    # the face area is considered in the tracking.
                    
                    ##-your-code-starts-here-##
                    (x, y, w, h) = faces[0]
                    roi_gray = gray[y : y+h, x :  x+w]  # replaced
                    ##-your-code-ends-here-##

                    # Get trackable points
                    p0 = cv2.goodFeaturesToTrack(roi_gray, 
                                                 maxCorners=70,
                                                 qualityLevel=0.001,
                                                 minDistance=5)

                    # Convert points to form (point_id, coordinates)
                    p0 = p0[:, 0, :] if p0 is not None else []
                    
                    # Convert from ROI to image coordinates
                    ##-your-code-starts-here-##
                    for i in range(len(p0)):
                        p0[i][0] += x
                        p0[i][1] += y
                    ##-your-code-ends-here-##

                # Save grayscale copy for next iteration
                gray_prev = gray.copy()
                
            else:
                # Tracking
                img = cv2.putText(img, 'Tracking', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0, 255, 255))

                # Calculate optical flow using calcOpticalFlowPyrLK
                p1, isFound, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray, p0, 
                                                            None,
                                                            winSize=(31,31),
                                                            maxLevel=10,
                                                            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03),
                                                            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
                                                            minEigThreshold=0.00025)
                
                # Select good points. Use isFound to select valid found points from p1
                ##-your-code-starts-here-#
                points = p1[isFound[:, 0] == 1, :]
                good_points = [(int(element[0]), int(element[1])) for element in points] # Convert float tuples to float int
                ##-your-code-starts-here-##
                
                # Draw points using e.g. cv2.drawMarker
                ##-your-code-starts-here-##
                for point in good_points:
                    cv2.drawMarker(img, point, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=10)
                ##-your-code-ends-here-##q
                
                # Update p0 (which points should be kept?) and gray_prev for 
                # next iteration
                ##-your-code-starts-here-##
                p0 = points
                gray_prev = gray.copy()
                ##-your-code-ends-here-##

            # Quit text
            img = cv2.putText(img, 'Press q to quit', (440, 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color=(0,0,255))
            cv2.imshow('Video feed', img)

        # Limit FPS to ~30 (if detector is fast enough)
        if cv2.waitKey(get_delay(start, fps=30)) & 0xFF == ord('q'):
            break  # q to quit
        
    # Catch exceptions in order to close camera and video feed window properly
    except:
        traceback.print_exc()  # display for user
        break
            
# Close camera and video feed window
cam.release()   
cv2.destroyAllWindows()

