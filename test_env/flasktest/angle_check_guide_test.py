import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy import spatial
import os

# 현재 점수를 얻기 위한 전역변수
score = 0

# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# list 매개변수로 입력받은 리스트의 목록을 출력하고 인덱스를 입력받아 해당 인덱스의 원소를 문자열로 반환하는 함수.
# msg 에는 선택시 띄울 메세지를 입력
def list_selection(list, msg):
    for index, elem in enumerate(list):
        print(f'[{index}] 입력시 [{elem}] 선택')
    sel = int(input(f'\n{msg}'))
    return str(list[sel])


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

def compare_pose(image, angle_point, angle_user, angle_target):
    angle_user = np.array(angle_user)
    angle_target = np.array(angle_target)
    angle_point = np.array(angle_point)
    stage = 0
    # cv2.rectangle(image, (0, 0), (370, 40), (255, 255, 255), -1)
    # cv2.rectangle(image, (0, 40), (370, 370), (255, 255, 255), -1)
    cv2.putText(image, str("Score:"), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2, cv2.LINE_AA)
    height, width, _ = image.shape

    # 캠 화면 반전 처리로 Left - Right 랜드마크 바꿔 처리함

    # angle1 [오른쪽 팔꿈치] : 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목 랜드마크 좌표를 이용하여 오른쪽 팔꿈치 각도 계산
    if angle_user[0] < (angle_target[0] - 15):
        # print(">>> 오른쪽 팔꿈치를 더 뻗으세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the left arm at elbow"), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

    # angle1 [오른쪽 팔꿈치] : 오른쪽 어깨 - 오른쪽 팔꿈치 - 오른쪽 손목 랜드마크 좌표를 이용하여 오른쪽 팔꿈치 각도 계산
    if angle_user[0] > (angle_target[0] + 15):
        # print(">>> 오른쪽 팔꿈치를 더 구부리세요")
        stage = stage + 1
        cv2.putText(image, str("Fold the left arm at elbow"), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[0][0] * width), int(angle_point[0][1] * height)), 30, (0, 0, 255), 5)

    # angle2 [왼쪽 팔꿈치] : 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목 랜드마크 좌표를 이용하여 팔꿈치 각도 계산
    if angle_user[1] < (angle_target[1] - 15):
        # print(">>> 왼쪽 팔꿈치를 더 뻗으세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the right arm at elbow"), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

    # angle2 [왼쪽 팔꿈치] : 왼쪽 어깨 - 왼쪽 팔꿈치 - 왼쪽 손목 랜드마크 좌표를 이용하여 팔꿈치 각도 계산
    if angle_user[1] > (angle_target[1] + 15):
        # print(">>> 왼쪽 팔꿈치를 더 구부리세요")
        stage = stage + 1
        cv2.putText(image, str("Fold the right arm at elbow"), (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[1][0] * width), int(angle_point[1][1] * height)), 30, (0, 0, 255), 5)

    # angle3 [오른팔 들어올린 각도] : 오른쪽 팔꿈치 - 오른쪽 어깨 - 오른쪽 고관절 랜드마크 좌표를 이용하여 오른팔의 들어올림 각도 계산
    if angle_user[2] < (angle_target[2] - 15):
        # print(">>> 오른팔을 더 들어올리세요")
        stage = stage + 1
        cv2.putText(image, str("Lift your left arm"), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

    # angle3 [오른팔 들어올린 각도] : 오른쪽 팔꿈치 - 오른쪽 어깨 - 오른쪽 고관절 랜드마크 좌표를 이용하여 오른팔의 들어올림 각도 계산
    if angle_user[2] > (angle_target[2] + 15):
        # print(">>> 오른팔을 더 내리세요")
        stage = stage + 1
        cv2.putText(image, str("Put your left arm down a little"), (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[2][0] * width), int(angle_point[2][1] * height)), 30, (0, 0, 255), 5)

    # angle4 [왼팔 들어올린 각도] : 왼쪽 팔꿈치 - 왼쪽 어깨 - 왼쪽 고관절 랜드마크 좌표를 이용하여 왼팔의 들어올림 각도 계산
    if angle_user[3] < (angle_target[3] - 15):
        # print(">>> 왼팔을 더 올리세요")
        stage = stage + 1
        cv2.putText(image, str("Lift your right arm"), (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

    # angle4 [왼팔 들어올린 각도] : 왼쪽 팔꿈치 - 왼쪽 어깨 - 왼쪽 고관절 랜드마크 좌표를 이용하여 왼팔의 들어올림 각도 계산
    if angle_user[3] > (angle_target[3] + 15):
        # print(">>> 왼팔을 더 내리세요")
        stage = stage + 1
        cv2.putText(image, str("Put your right arm down a little"), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0], 2,
                    cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[3][0] * width), int(angle_point[3][1] * height)), 30, (0, 0, 255), 5)

    # angle5 [오른다리 들어올린 각도] : 오른쪽 어깨 - 오른쪽 고관절 - 오른쪽 무릎 랜드마크 좌표를 이용하여 오른다리의 들어올림 각도를 계산
    if angle_user[4] < (angle_target[4] - 15):
        # print(">>> 오른다리를 더 들어올리세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at left hip"), (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

    # angle5 [오른다리 들어올린 각도] : 오른쪽 어깨 - 오른쪽 고관절 - 오른쪽 무릎 랜드마크 좌표를 이용하여 오른다리의 들어올림 각도를 계산
    if angle_user[4] > (angle_target[4] + 15):
        # print(">>> 오른다리를 더 내리세요")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle of at left hip"), (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    [0, 153, 0], 2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[4][0] * width), int(angle_point[4][1] * height)), 30, (0, 0, 255), 5)

    # angle6 [왼다리 들어올린 각도] : 왼쪽 어깨 - 왼쪽 고관절 - 왼쪽 무릎 랜드마크 좌표를 이용하여 왼다리의 들어올림 각도를 계산
    if angle_user[5] < (angle_target[5] - 15):
        # print(">>> 왼다리를 더 들어올리세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at right hip"), (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

    # angle6 [왼다리 들어올린 각도] : 왼쪽 어깨 - 왼쪽 고관절 - 왼쪽 무릎 랜드마크 좌표를 이용하여 왼다리의 들어올림 각도를 계산
    if angle_user[5] > (angle_target[5] + 15):
        # print(">>> 왼다리를 더 내리세요")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at right hip"), (10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[5][0] * width), int(angle_point[5][1] * height)), 30, (0, 0, 255), 5)

    # angle7 [오른쪽 무릎 굽힘 각도] : 오른쪽 고관절 - 오른쪽 무릎 - 오른쪽 발목 좌표를 이용하여 오른쪽 무릎의 굽힘 각도를 계산
    if angle_user[6] < (angle_target[6] - 15):
        # print(">>> 오른쪽 무릎을 더 펴세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle of left knee"), (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

    # angle7 [오른쪽 무릎 굽힘 각도] : 오른쪽 고관절 - 오른쪽 무릎 - 오른쪽 발목 좌표를 이용하여 오른쪽 무릎의 굽힘 각도를 계산
    if angle_user[6] > (angle_target[6] + 15):
        # print(">>> 오른쪽 무릎을 더 구부리세요")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at left knee"), (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[6][0] * width), int(angle_point[6][1] * height)), 30, (0, 0, 255), 5)

    # angle8 [왼쪽 무릎 굽힘 각도] : 왼쪽 고관절 - 왼쪽 무릎 - 왼쪽 발목 좌표를 이용하여 왼쪽 무릎의 굽힘 각도를 계산
    if angle_user[7] < (angle_target[7] - 15):
        # print(">>> 왼쪽 무릎을 더 펴세요")
        stage = stage + 1
        cv2.putText(image, str("Extend the angle at right knee"), (10, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)

    # angle8 [왼쪽 무릎 굽힘 각도] : 왼쪽 고관절 - 왼쪽 무릎 - 왼쪽 발목 좌표를 이용하여 왼쪽 무릎의 굽힘 각도를 계산
    if angle_user[7] > (angle_target[7] + 15):
        # print(">>> 왼쪽 무릎을 더 구부리세요")
        stage = stage + 1
        cv2.putText(image, str("Reduce the angle at right knee"), (10, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0, 153, 0],
                    2, cv2.LINE_AA)
        cv2.circle(image, (int(angle_point[7][0] * width), int(angle_point[7][1] * height)), 30, (0, 0, 255), 5)

    if stage != 0:
        # print("화이팅!")
        cv2.putText(image, str("FIGHTING!"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)

        pass

    else:
        # print("완벽합니다!")
        cv2.putText(image, str("PERFECT"), (170, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2, cv2.LINE_AA)


# 리스트를 매개변수로 받아 리스트의 원소 개수와 총합으로 리스트 평균을 구하는 함수
def calc_average(lst):
    return sum(lst) / len(lst)


# 코사인 유사도를 사용하여 리스트 x, 리스트 y간 유사도를 판별하는 함수
def diff_compare(x, y):
    average = []
    for i, j in zip(range(len(list(x))), range(len(list(y)))):
        result = 1 - spatial.distance.cosine(list(x[i].values()), list(y[j].values()))
        average.append(result)
    score = math.sqrt(2 * (1 - round(calc_average(average), 2)))
    # print(calc_average(average))
    return score


def diff_compare_angle(x, y):
    new_x = []
    for i, j in zip(range(len(x)), range(len(y))):
        z = np.abs(x[i] - y[j]) / ((x[i] + y[j]) / 2)
        new_x.append(z)
        # print(new_x[i])
    return calc_average(new_x)

# 현재 점수 return
def get_score():
    global score
    return str(score)

# 레거시 함수, 더 이상 안씀
# def convert_data(landmarks):
#     df = pd.DataFrame(columns=['x', 'y', 'z', 'vis'])
#     for i in range(len(landmarks)):
#         df = df.append({"x": landmarks[i].x,
#                         "y": landmarks[i].y,
#                         "z": landmarks[i].z,
#                         "vis": landmarks[i].visibility
#                         }, ignore_index=True)
#     return df

def run(folder_path, seq_num):
    CAMERA_DEVICE_ID = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

    # file_list = os.listdir(folder_path)
    # file_list_image = [file for file in file_list if file.endswith(('.jpg', 'jpeg', '.png'))]
    # file_list_image.sort()
    # print(f'Target Pose 이미지 목록 : {file_list_image}\n')

    target_pose_image = f"pose_{seq_num}.jpg"
    print("!!!!!!!!!!!!!!!!!!!")
    print(folder_path)
    print(target_pose_image)
    image_path = os.path.join(folder_path, target_pose_image)

    x = extractKeypoint(image_path)
    # cv2.imshow('target', x[3])
    angle_target = x[2]
    point_target = x[1]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라 연결을 확인해 주세요.")
            continue

        # 유저 편의성을 위해 카메라 영상 좌우 반전
        frame = cv2.flip(frame, 1)

        # 랜드마크 정보 추출
        results = pose.process(frame)

        # 카메라 해상도를 가져와 세로길이 720px 창으로 리사이즈
        image_height, image_width, _ = frame.shape
        img = cv2.resize(frame, (int(image_width * (720 / image_height)), 720))

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
            p_score = diff_compare(keypoints, point_target)

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

            # Target Pose 이미지와 실시간 유저 영상으로 부터 추출한 Landmark를 통해 계산한 angle_point를 비교, 점수화
            compare_pose(img, angle_point, angle, angle_target)
            a_score = diff_compare_angle(angle, angle_target)

            # 실시간 계산되는 관절 각도 영상에 오버레이
            cv2.putText(img, 'ID', (1100,14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            cv2.putText(img, str(1), (1100,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(2), (1100,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(3), (1100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(4), (1100,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(5), (1100,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(6), (1100,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(7), (1100,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(8), (1100,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)

            cv2.putText(img, 'Angle', (1200,12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, [0,0,255], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle1)), (1200,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle2)), (1200,70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle3)), (1200,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle4)), (1200,130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle5)), (1200,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle6)), (1200,190), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle7)), (1200,220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)
            cv2.putText(img, str(int(angle8)), (1200,250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, [0,153,0], 2, cv2.LINE_AA)

            global score
            if (p_score >= a_score):
                # print('p_score >= a_score')
                cv2.putText(img, str(int((1 - a_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                            cv2.LINE_AA)
                score = int((1 - a_score) * 100)
            else:
                # print('else 조건 분기 : p_score < a_score')
                cv2.putText(img, str(int((1 - p_score) * 100)), (80, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 0, 255], 2,
                            cv2.LINE_AA)
                score = int((1 - p_score) * 100)

        except:
            pass

        # cv2.imshow('User Cam Feed', img)
        img = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        if cv2.waitKey(1) & 0xFF == 27:
            break
