import os, jwt
import cv2
from flask import Flask, render_template, Response, jsonify, send_file, request, redirect, flash, url_for, Blueprint
from flask import current_app as current_app, session
from flask import render_template_string
from werkzeug.utils import secure_filename
import hashlib
from datetime import datetime, timedelta
from functools import wraps
import time

from PIL import Image
import pandas as pd
import numpy as np
import mediapipe as mp
import numpy as np
import pandas as pd
import math
from scipy import spatial

import angle_check_guide_test
import extract_key_point_guide
import extract_landmark_for_flask
import diff_extract_visualize_for_flask
import video_slice_check_for_flask
from module import dbModule
from test_db import test_db

app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()  # create background subtractor

UPLOAD_FOLDER = 'static/target_video/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.register_blueprint(test_db)

db_class = dbModule.Database()

# 현재 점수를 얻기 위한 전역변수
score = 0

# 포즈 끝 감지를 위한 전역변수
detect_change = False

# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# 비디오 스트리밍 예제 start

# @app.route('/')
# def index():
#     """Video streaming home page."""
#     return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def gen():
    """Video streaming generator function."""
    cap = cv2.VideoCapture('768x576.avi')

    # Read until video is completed
    while(cap.isOpened()):
        ret, frame = cap.read()  # import image
        if not ret: #if vid finish repeat
            frame = cv2.VideoCapture("768x576.avi")
            continue
        if ret:  # if there is a frame continue with code
            image = cv2.resize(frame, (0, 0), None, 1, 1)  # resize image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # converts image to gray
            fgmask = sub.apply(gray)  # uses the background subtraction
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # kernel to apply to the morphology
            closing = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(opening, kernel)
            retvalbin, bins = cv2.threshold(dilation, 220, 255, cv2.THRESH_BINARY)  # removes the shadows
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            minarea = 400
            maxarea = 50000
            for i in range(len(contours)):  # cycles through all contours in current frame
                if hierarchy[0, i, 3] == -1:  # using hierarchy to only count parent contours (contours not within others)
                    area = cv2.contourArea(contours[i])  # area of contour
                    if minarea < area < maxarea:  # area threshold for contour
                        # calculating centroids of contours
                        cnt = contours[i]
                        M = cv2.moments(cnt)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        # gets bounding points of contour to create rectangle
                        # x,y is top left corner and w,h is width and height
                        x, y, w, h = cv2.boundingRect(cnt)
                        # creates a rectangle around contour
                        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        # Prints centroid text in order to double check later on
                        cv2.putText(image, str(cx) + "," + str(cy), (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX,.3, (0, 0, 255), 1)
                        cv2.drawMarker(image, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, markerSize=8, thickness=3,line_type=cv2.LINE_8)
        #cv2.imshow("countours", image)
        frame = cv2.imencode('.jpg', image)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        #time.sleep(0.1)
        key = cv2.waitKey(20)
        if key == 27:
            break

# 비디오 스트리밍 예제 end

# 자세 일치도 예제 start
@app.route('/test_angle')
def test_angle_get():
    """Video streaming home page."""
    given_model_id = int(request.args.get('model_id'))
    
    return render_template('test_angle.html', model_id = given_model_id,)

@app.route('/test_angle_db')
def test_angle_db():
    given_model_id = int(request.args.get('model_id'))

    return render_template('test_angle.html', model_id = given_model_id)

@app.route('/test_angle_video')
def test_angle_video():
    """angle check guide."""
    given_model_id = int(request.args.get('model_id'))
    sql      = "SELECT POSE_ID, MODEL_ID, SEQ_NUM, POSE_DUR, FILE_SOURCE \
                FROM ROOMFIT_DB.ROUTINE_MODEL_POSE WHERE MODEL_ID = %d" % (given_model_id)
    poses   = db_class.executeAll(sql)
    # folder_path = result[0]['FILE_SOURCE'].split("/")
    # folder_path = f"{folder_path[0]}/{folder_path[1]}/{folder_path[2]}/{folder_path[3]}"
    print(poses)

    return Response(angle_check_gen(poses),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/extracted_current_pose_guide_img')
def extracted_current_pose_guide_img():
    return Response(extract_key_point_guide.ret_image(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/extracted_next_pose_guide_img')
def extracted_next_pose_guide_img():
    return Response(extract_key_point_guide.ret_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 포즈 끝 감지를 위한 전역변수
detect_seq_num = -1
cur_img_path = ""
next_img_path = ""

@app.route('/detect_seq_num')
def detect_seq_function():
    return jsonify({'value': detect_seq_num, 'cur_img_path': cur_img_path, 'next_img_path': next_img_path})

def angle_check_gen(poses):
    CAMERA_DEVICE_ID = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    webcam_fps = cap.get(cv2.CAP_PROP_FPS)
    # file_list = os.listdir(folder_path)
    # file_list_image = [file for file in file_list if file.endswith(('.jpg', 'jpeg', '.png'))]
    # file_list_image.sort()
    # print(f'Target Pose 이미지 목록 : {file_list_image}\n')
    global detect_seq_num
    global cur_img_path
    global next_img_path
    for i in range(0, len(poses)):
        detect_seq_num = i

        if i != len(poses) - 1:
            cur_img_path = poses[i]['FILE_SOURCE']
            next_img_path = poses[i + 1]['FILE_SOURCE']
        else:
            cur_img_path = poses[i]['FILE_SOURCE']
            next_img_path = './static/pose_end.png'

        print(poses[i])
        count_time = int(poses[i]['POSE_DUR']) * webcam_fps
        print(count_time)

        # folder_path = poses[i]['FILE_SOURCE'].split("/")
        # folder_path = f"{folder_path[0]}/{folder_path[1]}/{folder_path[2]}/{folder_path[3]}"
        # folder_path = "./static/target_pose/phw_clip/"
        # target_pose_image = "pose_1.jpg"
        # print("!!!!!!!!!!!!!!!!!!!")
        # print(folder_path)
        # print(target_pose_image)
        # image_path = os.path.join(folder_path, target_pose_image)

        x = extractKeypoint(poses[i]['FILE_SOURCE'])
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

                # cv2.putText(img, str("DURATION:"), (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255], 2,cv2.LINE_AA)
                if score >= 50:
                    count_time -= 1
                    cv2.putText(img, str("DURATION: ") + str(count_time // webcam_fps), (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255], 2,cv2.LINE_AA)
                else:
                    cv2.putText(img, str("DURATION:") + str(count_time // webcam_fps), (700, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 255], 2,cv2.LINE_AA)
                if count_time <= 0:
                    break
            except:
                pass

            # 카메라 FPS 출력
            cv2.putText(img, str("FPS : " + str(webcam_fps)), (30, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 2, cv2.LINE_AA)

            # cv2.imshow('User Cam Feed', img)
            img = cv2.imencode('.jpg', img)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

            if score >= 50:
                count_time -= 1

            if count_time <= 0:
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

        time.sleep(2)

@app.route('/test_angle_score')
def test_angle_score():
    score = angle_check_guide_test.get_score()
    return jsonify({'value': score})

@app.route('/extracted_pose_guide_img')
def extracted_pose_guide_img():
    """showing extracted guide img."""
    guide_img = extract_key_point_guide.ret_image()

    return Response(guide_img, mimetype='multipart/x-mixed-replace; boundary=frame')

# 자세 일치도 예제 end

# 비디오 업로드 및 자세 추출 start
@app.route('/upload_video', methods=['GET'])
def upload_form():
    return render_template('upload.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    model_name = request.form['model_name']

    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_video filename: ' + filename)
        flash('Video successfully uploaded and displayed below')

        file_name = file.filename.split(".")[0]
        extract_landmark_for_flask.run()
        diff_extract_visualize_for_flask.run(file_name)
        vid_slice_info = video_slice_check_for_flask.run(file_name)

        total_pose_cnt = len(vid_slice_info)
        total_time = sum(pose[2] for pose in vid_slice_info)
        thumbnail = "./static/target_pose/" + file_name + "/pose_1.jpg"
        sql      = "INSERT INTO ROOMFIT_DB.ROUTINE_MODEL(MODEL_NAME, TOTAL_POSE_CNT, TOTAL_TIME, THUMBNAIL) \
                    VALUES('%s', '%d', '%d', '%s')" % (model_name, total_pose_cnt, total_time, thumbnail)
        inserted_id = db_class.execute(sql)
        print(inserted_id)
        seq_num = 0
        for pose in vid_slice_info:
            seq_num += 1
            pose_dur = pose[2]
            file_path = "./static/target_pose/" + file_name + "/pose_" + str(seq_num) + ".jpg"
            sql = "INSERT INTO ROOMFIT_DB.ROUTINE_MODEL_POSE(MODEL_ID, SEQ_NUM, POSE_DUR, FILE_SOURCE) \
                    VALUES('%d', '%d', '%d', '%s')" % (int(inserted_id), int(seq_num), int(pose_dur), file_path)
            db_class.execute(sql)

        db_class.commit()

        return render_template('my_model_list.html', filename=filename)

# 비디오 업로드 및 자세 추출 end

# 내 모델 리스트 출력 화면 start
@app.route('/my_model_list')
def my_model_list():
    """my model list page."""
    return render_template('my_model_list.html')

@app.route('/models', methods=['GET'])
def read_models():
    sql      = "SELECT MODEL_ID, MODEL_NAME, TOTAL_POSE_CNT, TOTAL_TIME, THUMBNAIL \
                FROM ROOMFIT_DB.ROUTINE_MODEL"
    result      = db_class.executeAll(sql)
    print(result)
    return jsonify({'result': 'success', 'models': result})

# 내 모델 리스트 출력 화면 end

# 회원가입 및 얼굴인식 start
#------
# 1.회원가입
# 웹캠으로 유저 데이터 뽑아서 모델 학습시키기
# 모델 기기 local에 저장
# id,pw 서버로 올리기
@app.route('/register', methods =['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST':
        if 'idd' in request.form and 'password' in request.form:
            id_receive = request.form['idd']
            pw_receive = request.form['password']
            pw_hash=hashlib.sha256(pw_receive.encode('utf-8')).hexdigest()

            sql = "SELECT USER_NAME, PASSWORD FROM ROOMFIT_DB.ROOMFIT_USER WHERE USER_NAME = %s"
            result = db_class.executeOne(sql, (id_receive))

            if result:
                msg = 'It is a registerd ID'
            else:
                sql = "INSERT INTO ROOMFIT_DB.ROOMFIT_USER(USER_NAME, PASSWORD) VALUES ('%s', '%s')" % (id_receive, pw_hash)
                result = db_class.execute(sql)
                db_class.commit()

                msg = '성공적으로 가입되었습니다!'
                return render_template('face_register.html', msg = msg, idd=id_receive) # home으로 연동

        else:
            msg = '모든 항목을 기입해 주세요!'
    return render_template('register.html', msg = msg)

# 회원가입 및 얼굴인식 end

# id/pw 로그인 start
@app.route('/login', methods =['GET', 'POST'])
def login():
    auth = request.authorization
    msg = ''
    if request.method == 'POST':
        if 'id' in request.form and 'password' in request.form:
            id_receive = request.form['id']
            pw_receive = request.form['password']
            pw_hash = hashlib.sha256(pw_receive.encode('utf-8')).hexdigest()

            sql = "SELECT USER_NAME, PASSWORD FROM ROOMFIT_DB.ROOMFIT_USER WHERE USER_NAME = %s AND PASSWORD = %s"
            result = db_class.executeOne(sql, (id_receive, pw_hash))

            if result:
                session['loggedin'] = True
                session['USER_NAME'] = result['USER_NAME']

                token = jwt.encode({
                    'id':id_receive,
                    'expiration': str(datetime.utcnow() + timedelta(seconds=60 * 60 * 24))
                },
                    app.config['SECRET_KEY'],
                    algorithm='HS256')

                msg = '성공적으로 로그인 되었습니다!'
                return render_template('my_model_list.html', msg=msg)

            else:
                msg = '휴대폰 번호 / 비밀번호가 일치하지 않습니다!'

    return render_template('login.html', msg = msg)

# id/pw 로그인 end

# 얼굴 인식 등록 start
isFaceR=False
@app.route('/face_register', methods =['GET', 'POST'])
def face_register():
    global isFaceR
    isFaceR=False
    while not isFaceR:
        return render_template('face_register.html')
    return render_template('login.html')

@app.route('/get_face_data', methods =['GET', 'POST'])
def get_face_data():
    """Video streaming route. Put this in the src attribute of an img tag."""
    un = request.args.get('user_name', default = 'ns-abc-aaa', type = str)
    return Response(face_data(un),mimetype='multipart/x-mixed-replace; boundary=frame')

def face_data(user_name):
    global isFaceR

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    save_path = f'./static/face_training/dataset/{user_name}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    count=0
    # face_detector = cv2.CascadeClassifier('./static/face_training/haarcascade_frontalface_default.xml')
    face_detector = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    # Read until video is completed
    while(cap.isOpened()):
        ret, img = cap.read()  # import image
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            # 얼굴이 인식되면 카운터 증가
            count += 1
            cv2.imwrite(f'{save_path}/' + str(count).zfill(3) + ".jpg", gray[y:y+h,x:x+w])

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        if count >= 30:
            face_training()
            print(isFaceR)
            time.sleep(1)
            isFaceR=True
            break

def face_training():
    dataset_path = './static/face_training/dataset'
    user_list = os.listdir(dataset_path)
    user_list = [user for user in user_list if
                 os.path.isdir(os.path.join(dataset_path, user))]
    user_list.sort()
    user_list_pd = pd.DataFrame(user_list, columns=['user'])
    user_list_pd.to_csv('./static/face_training/model_user_list.csv', index=True)

    trainset_path = './static/face_training/train_data'
    model_exist_check = os.path.join(trainset_path, 'user_face_model.yml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

    target_images = []
    facesamples = []
    id = []

    img_count = 0

    for user in user_list:
        img_path = os.path.join(dataset_path,user)
    img_list = os.listdir(img_path)
    print(f'\n>>> 유저 인덱스 [{user_list.index(user)}] = {user} ')

    for img in img_list:
        img_to_path = os.path.join(img_path,img)
        target_images.append(img_to_path)
        print(img_to_path)
        img_count += 1
        image = cv2.imread(img_to_path)

        # 그레이 스케일로 변환
        PIL_img = Image.open(img_to_path).convert('L')
        img_numpy = np.array(PIL_img,'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            facesamples.append(img_numpy[y:y+h,x:x+w])
            id.append(user_list.index(user))

    recognizer.train(facesamples, np.array(id))
    # trainset_path 에 유저 이름으로 인식 모델 저장
    recognizer.write(f'{trainset_path}/user_face_model.yml')
    return
# 얼굴 인식 등록 end

# 얼굴 인식 로그인 start
# 4. faceID 로그인
# 정해진 시간안에 일치율이 넘으면 그냥 로그인
# 안되면 id/pw 로그인 창으로 넘어가기
isFaceL,FaceLtimeout=False, False
@app.route('/face_login')
def face_login():
    global isFaceL,FaceLtimeout
    isFaceL,FaceLtimeout=False, False
    while not isFaceL:
        if FaceLtimeout:
            return render_template('login.html')
        return render_template('face_login.html',msg="확인")
    return render_template('my_model_list.html')

@app.route('/face_cognize', methods =['GET', 'POST'])
def face_cognize():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(face_model_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def face_model_gen():
    global isFaceL,FaceLtimeout
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./static/face_training/train_data/user_face_model.yml')

    cascadePath = "./haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    user_list = pd.read_csv('./static/face_training/model_user_list.csv')
    names = user_list['user'].tolist()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    font = cv2.FONT_HERSHEY_SIMPLEX
    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)

    frame_cnt=0
    confidence_cnt=0
    while (cap.isOpened()): # 제한시간설정
        frame_cnt+=1
        if confidence_cnt > fps*frame_count*30/10:
            time.sleep(1)
            isFaceL=True
            break

        ret, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            if (confidence < 100):
                id = names[id]
                if (confidence <90):
                    confidence_cnt+=1
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    time.sleep(1)
    FaceLtimeout=True
# 얼굴 인식 로그인 end

# angle_check 필요 함수 start

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

# angle check 필요 함수 end

# 데이터 스트림 테스트
# generator 함수
def generate():
    for i in range(100):
        # 1초 딜레이
        time.sleep(1)
        yield f'<p>{i}</p>\n'

# TODO : 위 변수를 받아서 페이지 리로딩 없이 HTML에 렌더링 하는게 목표
# 1. 변수로 받는 방법 - HTML 요소로 바로 되는지? 비디오 스트림 img src로 받아온것 처럼
#                 - 자바스크립트 변수로 받아야 되는지?
# 2. 대안 : iframe 으로 리로딩 되어야 할 부분만 만들어서 거기다가 넣고 리로딩
@app.route('/stream')
def stream():
    return Response(generate(), mimetype='text/html')

# 요것도 안됨
@app.route('/stream_data_page')
def stream_page():
    return render_template('stream_data.html')

@app.route('/stream_data_flask')
def stream_flask():
    return render_template_string('''<p>This is the current value: <span id="latest_value"></span></p>
<script>

    var latest = document.getElementById('latest_value');

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '{{ url_for('stream') }}');

    xhr.onreadystatechange = function() {
        var all_lines = xhr.responseText.split('\\n');
        last_line = all_lines.length - 2
        latest.textContent = all_lines[last_line]

        if (xhr.readyState == XMLHttpRequest.DONE) {
            /*alert("The End of Stream");*/
            latest.textContent = "The End of Stream"
        }
    }

    xhr.send();

</script>''')

# AJAX Examples
@app.route('/stream_time')
def stream_time():
    def generate_timestream():
        while True:
            current_time = time.strftime("%H:%M:%S\n")
            print(current_time)
            yield current_time
            time.sleep(1)

    return Response(generate_timestream(), mimetype='text/plain')


# HTML에 AJAX로 뿌려주는건 안됨
@app.route('/stream_time_page')
def stream_time_page():
    return render_template('stream_time.html')

# flask에서 렌더링된 템플릿에 뿌려주는건 됨
@app.route('/stream_time_flask')
def stream_time_flask():
    return render_template_string('''<p>This is the current value: <span id="latest_value"></span></p>
<script>

    var latest = document.getElementById('latest_value');

    var xhr = new XMLHttpRequest();
    xhr.open('GET', '{{ url_for('stream_time') }}');

    xhr.onreadystatechange = function() {
        var all_lines = xhr.responseText.split('\\n');
        last_line = all_lines.length - 2
        latest.textContent = all_lines[last_line]

        if (xhr.readyState == XMLHttpRequest.DONE) {
            /*alert("The End of Stream");*/
            latest.textContent = "The End of Stream"
        }
    }

    xhr.send();

</script>''')
