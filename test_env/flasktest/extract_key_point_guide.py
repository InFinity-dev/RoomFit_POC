import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy import spatial
import os
from PIL import Image
from io import BytesIO

# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 3개의 2차원 랜드마크 좌표를 순서대로 받아 각도를 계산하여 반환하는 함수
def calculateAngle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


# 정답자세가 될 사진으로 부터 랜드마크를 추출하고 관절 각도를 계산해서 img와 함께 리턴
def extractKeypoint(path):
    IMAGE_FILES = [path]
    stage = None
    joint_list_video = pd.DataFrame([])
    count = 0


    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image = cv2.flip(image,1)

        results = pose.process(image)
        image_h, image_w, _ = image.shape

        try:

            landmarks = results.pose_landmarks.landmark

            # print(landmarks)

            # 관절 각도 계산에 사용될 랜드마크 추출

            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

            left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            right_ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

            # 추출한 랜드마크 DataFrame으로 저장
            joint_list = pd.DataFrame([])

            for i, data_point in zip(range(len(landmarks)), landmarks):
                joints = pd.DataFrame({
                    'frame': count,
                    'id': i,
                    'x': data_point.x,
                    'y': data_point.y,
                    'z': data_point.z,
                    'vis': data_point.visibility
                }, index=[0])
                joint_list = pd.concat([joint_list, joints], ignore_index=True)

            keypoints = []
            for point in landmarks:
                keypoints.append({
                    'X': point.x,
                    'Y': point.y,
                    'Z': point.z,
                })

            # 8개 관절 각도를 저장할 배열 선언
            # [오른쪽 팔꿈치, 왼쪽 팔꿈치, 오른팔 들어올린 각도, 왼팔 들어올린 각도,
            # 오른다리 들어올린 각도, 왼다리 들어올린 각도, 오른쪽 무릎 굽힘 각도, 왼쪽 무릎 굽힘 각도]
            angle = []

            # angle1 [오른쪽 팔꿈치] : 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목 랜드마크 좌표를 이용하여 오른쪽 팔꿈치 각도 계산
            angle1 = calculateAngle(right_shoulder, right_elbow, right_wrist)
            angle.append(int(angle1))

            # angle2 [왼쪽 팔꿈치] : 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목 랜드마크 좌표를 이용하여 팔꿈치 각도 계산
            angle2 = calculateAngle(left_shoulder, left_elbow, left_wrist)
            angle.append(int(angle2))

            # angle3 [오른팔 들어올린 각도] : 오른쪽 팔꿈치 - 오른쪽 어깨 - 오른쪽 고관절 랜드마크 좌표를 이용하여 오른팔의 들어올림 각도 계산
            angle3 = calculateAngle(right_elbow, right_shoulder, right_hip)
            angle.append(int(angle3))

            # angle4 [왼팔 들어올린 각도] : 왼쪽 팔꿈치 - 왼쪽 어깨 - 왼쪽 고관절 랜드마크 좌표를 이용하여 왼팔의 들어올림 각도 계산
            angle4 = calculateAngle(left_elbow, left_shoulder, left_hip)
            angle.append(int(angle4))

            # angle5 [오른다리 들어올린 각도] : 오른쪽 어깨 - 오른쪽 고관절 - 오른쪽 무릎 랜드마크 좌표를 이용하여 오른다리의 들어올림 각도를 계산
            angle5 = calculateAngle(right_shoulder, right_hip, right_knee)
            angle.append(int(angle5))

            # angle6 [왼다리 들어올린 각도] : 왼쪽 어깨 - 왼쪽 고관절 - 왼쪽 무릎 랜드마크 좌표를 이용하여 왼다리의 들어올림 각도를 계산
            angle6 = calculateAngle(left_shoulder, left_hip, left_knee)
            angle.append(int(angle6))

            # angle7 [오른쪽 무릎 굽힘 각도] : 오른쪽 고관절 - 오른쪽 무릎 - 오른쪽 발목 좌표를 이용하여 오른쪽 무릎의 굽힘 각도를 계산
            angle7 = calculateAngle(right_hip, right_knee, right_ankle)
            angle.append(int(angle7))

            # angle8 [왼쪽 무릎 굽힘 각도] : 왼쪽 고관절 - 왼쪽 무릎 - 왼쪽 발목 좌표를 이용하여 왼쪽 무릎의 굽힘 각도를 계산
            angle8 = calculateAngle(left_hip, left_knee, left_ankle)
            angle.append(int(angle8))

            # 사진 상에 판별된 관절 랜드마크에 번호 오버레이
            cv2.putText(image,
                        str(1),
                        tuple(np.multiply(right_elbow, [image_w, image_h, ]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(2),
                        tuple(np.multiply(left_elbow, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(3),
                        tuple(np.multiply(right_shoulder, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(4),
                        tuple(np.multiply(left_shoulder, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(5),
                        tuple(np.multiply(right_hip, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(6),
                        tuple(np.multiply(left_hip, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(7),
                        tuple(np.multiply(right_knee, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
            cv2.putText(image,
                        str(8),
                        tuple(np.multiply(left_knee, [image_w, image_h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        [255, 255, 0],
                        2,
                        cv2.LINE_AA
                        )
        except:
            pass


        joint_list_video = pd.concat([joint_list_video, joint_list], ignore_index=True)
        cv2.rectangle(image, (0, 0), (100, 255), (255, 255, 255), -1)

        cv2.putText(image, 'ID', (10, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
        cv2.putText(image, str(1), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(2), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(3), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(4), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(5), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(6), (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(7), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(8), (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

        cv2.putText(image, 'Angle', (40, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0, 0, 255], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle1)), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle2)), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle3)), (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle4)), (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle5)), (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle6)), (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle7)), (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
        cv2.putText(image, str(int(angle8)), (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)

        # Render detections
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_draw.DrawingSpec(color=(0, 0, 255), thickness=4, circle_radius=2),
                                  mp_draw.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=2)
                                  )

        # cv2.imshow('MediaPipe Feed',image)

        # ESC키 입력시 break
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    return landmarks, keypoints, angle, image

def ret_image():
    folder_path = '../../target_pose'
    target_pose_image = "dab.jpg"

    image_path = os.path.join(folder_path, target_pose_image)

    extracted_img = extractKeypoint(image_path)[3]
    image_height, image_width, _ = extracted_img.shape
    img = cv2.resize(extracted_img, (int(image_width * (720 / image_height)), 720))
    buffer = cv2.imencode('.png', img)[1].tobytes()
    ret_img = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + buffer + b'\r\n')

    return ret_img