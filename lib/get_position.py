import mediapipe as mp
mp_pose = mp.solutions.pose

def get_position(name,results,image):
    xl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].x * image.shape[1])
    yl = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['LEFT_'+name]].y * image.shape[0])
    xr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].x * image.shape[1])
    yr = int(results.pose_landmarks.landmark[mp_pose.PoseLandmark['RIGHT_'+name]].y * image.shape[0])
    return (xl,yl),(xr,yr)