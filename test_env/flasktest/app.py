import time
import cv2 
from flask import Flask, render_template, Response
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy import spatial

from utils.test import *


app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2() 

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -------------------
# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

def gen():
    CAMERA_DEVICE_ID = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    
    path = 'static/^^7.jpg'
    x = extractKeypoint(path)
    
    cv2.imshow('target', x[3])
    angle_target = x[2]
    point_target = x[1]
    
    while (cap.isOpened()):
        ret, img = cap.read()
        if not ret:
            print("카메라 연결을 확인해 주세요.")
            break
        else:
            # 유저 편의성을 위해 카메라 영상 좌우 반전
            img = cv2.flip(img, 1)   
            # 랜드마크 정보 추출
            results = pose.process(img)         
            # 카메라 해상도를 가져와 세로길이 720px 창으로 리사이즈
            image_height, image_width, _ = img.shape
            img = cv2.resize(img, (int(image_width * (720 / image_height)), 720))
            # Landmark 그리기
            mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=4),
                                mp_draw.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=3)
                                )
            
            try:
                landmarks = results.pose_landmarks.landmark

                # 관절 각도 계산에 사용될 포인트 랜드마크 좌표 추출
                angle_point = []

                right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                angle_point.append(right_elbow)

                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                angle_point.append(left_elbow)

                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                angle_point.append(right_shoulder)

                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                angle_point.append(left_shoulder)

                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                angle_point.append(right_hip)

                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                angle_point.append(left_hip)

                right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                angle_point.append(right_knee)

                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                angle_point.append(left_knee)
                right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # 포즈 유사도를 검사하기 위한 랜드마크 정보 배열에 저장
                keypoints = []
                for point in landmarks:
                    keypoints.append({
                        'X': point.x,
                        'Y': point.y,
                        'Z': point.z,
                    })

                # 유사도 비교 함수로 저장한 랜드마크 정보와 정답 사진에서 뽑은 point_target 배열 전달하여 점수 판정
                p_score = dif_compare(keypoints, point_target)

                # 계산한 관절 각도를 저장할 배열 선언
                angle = []

                angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
                angle.append(int(angle1))

                angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
                angle.append(int(angle2))

                angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
                angle.append(int(angle3))

                angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
                angle.append(int(angle4))

                angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
                angle.append(int(angle5))

                angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
                angle.append(int(angle6))

                angle7 = calculateAngle(right_hip, right_knee, right_ankle)
                angle.append(int(angle7))

                angle8 = calculateAngle(left_hip, left_knee, left_ankle)
                angle.append(int(angle8))

                compare_pose(img, angle_point, angle, angle_target)
                a_score = diff_compare_angle(angle, angle_target)

                # 실시간 계산되는 관절 각도 영상에 오버레이
                cv2.putText(img, 'ID', (1100, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
                cv2.putText(img, str(1), (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(2), (1100, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(3), (1100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(4), (1100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(5), (1100, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(6), (1100, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(7), (1100, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(8), (1100, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

                cv2.putText(img, 'Angle', (1200, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle1)), (1200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle2)), (1200, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle3)), (1200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle4)), (1200, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle5)), (1200, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle6)), (1200, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle7)), (1200, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
                cv2.putText(img, str(int(angle8)), (1200, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

                if (p_score >= a_score):
                    cv2.putText(img, str(int((1 - a_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                                cv2.LINE_AA)

                else:
                    cv2.putText(img, str(int((1 - p_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                                cv2.LINE_AA)

            except:
                pass

            # cv2.imshow('MediaPipe Feed', img)
            frame = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            if cv2.waitKey(1) & 0xFF == 27:
                break