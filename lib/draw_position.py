import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
def draw_position(name,results,image,put_text = False,color = (0, 255, 0,255)):
    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5, color, -1)
    if (put_text):
         cv2.putText(image, f"{x}:{y}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

    x = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    y = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    cv2.circle(image, (x, y), 5,color, -1)
    if (put_text):
        cv2.putText(image, f"{x}:{y}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)