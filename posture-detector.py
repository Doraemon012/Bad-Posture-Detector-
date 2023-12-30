import cv2 
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# find angle between two lines
def find_angle(point1, point2) -> float:
    # Find the angle between two lines
    angle = np.arctan((point1[0] - point2[0]) / (point2[1] - point1[1]))
    # convert to degrees
    angle = angle * 180 / np.pi
    # keep it always positive
    angle = np.abs(angle)
    return angle

# find euclidean distance between two points
def find_distance(point1, point2) -> float:
    # Find the distance between two points
    distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    return distance

# find mid point between two points
def find_mid_point(point1, point2) -> tuple:
    # Find the mid point between two points
    mid_point = (int((point1[0] + point2[0])/2), int((point1[1] + point2[1])/2))
    return mid_point

# detect key points
def detect_keypoints(img, results) -> dict:

    # Extract shoulder, ear, and hip points -> scale them according to image size
    points = {}

    image_shape = img.shape  # Assuming 'img' is the loaded image

    landmarks = results.pose_landmarks.landmark

    # scaled shoulder points
    points['shoulder'] = (
        (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_shape[0])),
        (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_shape[0]))
    )

    # scaled ear points
    points['ear'] = (
        (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * image_shape[0])),
        (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR].y * image_shape[0]))
    )
    
    # scaled hip points
    points['hip'] = (
        (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * image_shape[0])),
        (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * image_shape[1]),
        int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * image_shape[0]))
    )

    return points

# check neck posture
def check_neck_posture(points) -> tuple:

    mid_ear = find_mid_point(points['ear'][0], points['ear'][1])

    mid_shoulder = find_mid_point(points['shoulder'][0], points['shoulder'][1])

    # Calculate the angle between the neck and the y axis
    neck_angle = find_angle(mid_ear, mid_shoulder)

    # calculate the distance between hips
    neck_distance = find_distance(points["shoulder"][1], points["shoulder"][0])

    # check if the angle is less than 10 degrees
    if neck_angle < 10 and neck_distance < 20: # ! change 
        return (True, neck_angle, neck_distance)
    else:
        return (False, neck_angle, neck_distance)

def check_hip_posture(points) -> tuple:

    # sample points :
    # points = {
    #     'ear': ((0, 0), (0, 0)),
    #

    mid_hip = find_mid_point(points['hip'][0], points['hip'][1])

    mid_shoulder = find_mid_point(points['shoulder'][0], points['shoulder'][1])

    # Calculate the angle between the neck and the y axis
    hip_angle = find_angle(mid_hip, mid_shoulder)

    # calculate the distance between hips
    hip_distance = find_distance(points["hip"][1], points["hip"][0])

    # check if the angle is less than 10 degrees
    if hip_angle < 10 and hip_distance < 20: # ! change 
        return (True, hip_angle, hip_distance)
    else:
        return (False, hip_angle, hip_distance)

def draw_on_image(img, points) -> np.ndarray:

    # find mid point between two ear scaled points
    mid_ear = find_mid_point(points['ear'][0], points['ear'][1])

    # find mid point between two hip scaled points
    mid_hip = find_mid_point(points['hip'][0], points['hip'][1])

    # find mid point between two shoulder scaled shoulder points
    mid_shoulder = find_mid_point(points['shoulder'][0], points['shoulder'][1])

    # find distance between two shoulder scaled points
    shoulder_distance = find_distance(points['shoulder'][0], points['shoulder'][1])

    # fins distance between two hip scaled points
    hip_distance = find_distance(points['hip'][0], points['hip'][1])

   
    # join ear and shoulder mid points
    cv2.line(img, mid_ear, mid_shoulder, (0, 237, 255), 2)

    # join hip and shoulder mid points
    cv2.line(img, mid_hip, mid_shoulder, (0, 189, 255), 2)

    # draw ear mid point on image
    cv2.circle(img, mid_ear, 5, (255, 0, 0), -1)

    # ! draw scaled ear points on image
    # cv2.circle(img, points['ear'][0], 5, (0, 255, 0), -1)
    # cv2.circle(img, points['ear'][1], 5, (0, 255, 0), -1)

    # draw hip mid point on image
    cv2.circle(img, mid_hip, 5, (255, 0, 0), -1)

    # ! draw scaled hip points on image
    cv2.circle(img, points['hip'][0], 3, (0, 255, 0), -1)
    cv2.circle(img, points['hip'][1], 3, (0, 255, 0), -1)

    # draw shoulder mid point on image
    cv2.circle(img, mid_shoulder, 5, (255, 0, 0), -1)

    # ! draw scaled shoulder points on imagef
    # cv2.circle(img, points['shoulder'][0], 5, (0, 255, 0), -1)
    # cv2.circle(img, points['shoulder'][1], 5, (0, 255, 0), -1)

    # # draw translucent line joining two shoulder scaled points
    # cv2.line(img, points['shoulder'][0], points['shoulder'][1], (0, 0, 255), 2)

    # # draw translucent line joining two hip scaled points
    # cv2.line(img, points['hip'][0], points['hip'][1], (54, 54, 51, 0.71), 1)

    # draw small vertical line on shoulder mid point
    cv2.line(img, (mid_shoulder[0], mid_shoulder[1] - 50), (mid_shoulder[0], mid_shoulder[1] + 50), (0, 0, 255, 0.5), 1)

    # draw small vertical line on hip mid point
    cv2.line(img, (mid_hip[0], mid_hip[1] - 50), (mid_hip[0], mid_hip[1] + 50), (0, 0, 255, 0.5), 1)

    # write hip posture on image
    cv2.putText(img, "Hip Posture: " + str(check_hip_posture(points)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # write neck posture on image
    cv2.putText(img, "Neck Posture: " + str(check_neck_posture(points)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    return img

# main function
def main():
    # image path
    # image_path = r'C:\Users\neyat\Desktop\mediapp\chat\sit1.jpg'

    # img = cv2.imread(image_path)

    # check with live video
    cap = cv2.VideoCapture(0)

    # downsample the image to make it faster
    downsample_factor = 0.5  # Adjust 

    while cap.isOpened():
        success, img = cap.read()
        # image = img.copy()

        if not success:
            print("Ignoring empty camera frame.")
            continue

        results =  pose.process(img)

        if results.pose_landmarks:
            
            # detect keypoints
            points = detect_keypoints(img, results )

            # draw on image
            img = draw_on_image(img, points)


        cv2.imshow('MediaPipe Pose', img)

        # exit
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break

# main function call
if __name__ == '__main__':
    main()