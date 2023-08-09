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

# Compare posture with reference posture.
def comparePosture(angles, reference_angles, threshold=10):
    num_angles = len(angles)
    if num_angles != len(reference_angles):
        return False
    for i in range(num_angles):
        if abs(angles[i] - reference_angles[i]) > threshold:
            return False
    return True

# initialize Pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Load the reference posture image
reference_image = cv2.imread('PostureReference.jpg')
reference_RGB = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# process the reference image to get the reference pose landmarks
reference_results = pose.process(reference_RGB)

# Fetch angles between landmarks for reference posture
if reference_results.pose_landmarks:
    reference_landmarks = reference_results.pose_landmarks.landmark
    reference_num_landmarks = len(reference_landmarks)
    reference_angles = []
    for i in range(1, reference_num_landmarks - 1):
        x1 = int(reference_landmarks[i - 1].x * reference_image.shape[1])
        y1 = int(reference_landmarks[i - 1].y * reference_image.shape[0])
        x2 = int(reference_landmarks[i].x * reference_image.shape[1])
        y2 = int(reference_landmarks[i].y * reference_image.shape[0])
        x3 = int(reference_landmarks[i + 1].x * reference_image.shape[1])
        y3 = int(reference_landmarks[i + 1].y * reference_image.shape[0])

        angle = findAngle(x1, y1, x2, y2, x3, y3)
        reference_angles.append(angle)

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

        # Fetch angles between landmarks for detected posture
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            num_landmarks = len(landmarks)
            angles = []
            for i in range(1, num_landmarks - 1):
                x1 = int(landmarks[i - 1].x * frame.shape[1])
                y1 = int(landmarks[i - 1].y * frame.shape[0])
                x2 = int(landmarks[i].x * frame.shape[1])
                y2 = int(landmarks[i].y * frame.shape[0])
                x3 = int(landmarks[i + 1].x * frame.shape[1])
                y3 = int(landmarks[i + 1].y * frame.shape[0])

                angle = findAngle(x1, y1, x2, y2, x3, y3)
                angles.append(angle)

            # Compare detected posture with reference posture
            if comparePosture(angles, reference_angles, threshold=10):
                cv2.putText(frame, "Matching Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Different Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Draw reference pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, reference_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # show the final output
        cv2.imshow('Output', frame)
    except:
        break

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
