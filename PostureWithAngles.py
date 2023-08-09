# -*- coding: utf-8 -*-
"""
Created on Sun Jul 2 14:16:39 2023
@author: ahmad.rifai
"""

# TechVidvan Human pose estimator
# import necessary packages

import cv2
import mediapipe as mp
import math as m

# Calculate angle.
def findAngle(x1, y1, x2, y2, x3, y3):
    theta1 = m.atan2(y1 - y2, x1 - x2)
    theta2 = m.atan2(y3 - y2, x3 - x2)
    angle = abs(theta1 - theta2) * (180 / m.pi)
    return angle


# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# create capture object
cap = cv2.VideoCapture(0)  # 0 represents the default webcam

while cap.isOpened():
    # read frame from capture object
    ret, frame = cap.read()

    if not ret:
        break

    try:
        # convert the frame to RGB format
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # process the RGB frame to get the result
        results = pose.process(RGB)

        # draw detected skeleton on the frame
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Fetch angles between landmarks
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            num_landmarks = len(landmarks)
            for i in range(1, num_landmarks - 1):
                x1 = int(landmarks[i - 1].x * frame.shape[1])
                y1 = int(landmarks[i - 1].y * frame.shape[0])
                x2 = int(landmarks[i].x * frame.shape[1])
                y2 = int(landmarks[i].y * frame.shape[0])
                x3 = int(landmarks[i + 1].x * frame.shape[1])
                y3 = int(landmarks[i + 1].y * frame.shape[0])

                angle = findAngle(x1, y1, x2, y2, x3, y3)
                cv2.putText(frame, str(round(angle, 2)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # show the final output
        cv2.imshow('Output', frame)
    except:
        break
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
