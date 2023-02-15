import os, jwt
import cv2 
from flask import Flask, render_template, Response, jsonify, send_file, request, redirect, flash, url_for, Blueprint
from flask import current_app as current_app, session
from werkzeug.utils import secure_filename
import hashlib
from datetime import datetime, timedelta
from functools import wraps
from PIL import Image
import pandas as pd
import numpy as np

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

# 비디오 스트리밍 예제 start

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

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
    sql      = "SELECT POSE_ID, MODEL_ID, SEQ_NUM, POSE_DUR, FILE_SOURCE \
                FROM ROOMFIT_DB.ROUTINE_MODEL_POSE WHERE MODEL_ID = %d" % (given_model_id)
    result   = db_class.executeAll(sql)
    print("!!!!!!!!!!")
    print(result)
    return render_template('test_angle.html',len = len(result), poses = result)

@app.route('/test_angle_video')
def test_angle_video():
    """angle check guide."""
    
    return Response(angle_check_guide_test.run(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

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

        seq_num = 0
        for pose in vid_slice_info:
            seq_num += 1
            pose_dur = pose[2]
            file_path = "./static/target_pose/" + file_name + "/pose_" + str(seq_num) + ".jpg"
            sql = "INSERT INTO ROOMFIT_DB.ROUTINE_MODEL_POSE(MODEL_ID, SEQ_NUM, POSE_DUR, FILE_SOURCE) \
                    VALUES('%d', '%d', '%d', '%s')" % (inserted_id, seq_num, pose_dur, file_path)
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
                return render_template('login.html', msg = msg, idd=id_receive) # home으로 연동
            
        else:
            msg = '모든 항목을 기입해 주세요!'
    return render_template('register.html', msg = msg)

# 회원가입 및 얼굴인식 end

# id/pw 로그인 start

# id/pw 로그인
@app.route('/idpw_login', methods =['GET', 'POST'])
def idpw_login():
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

# Face ID 등록
@app.route('/face_register', methods =['GET', 'POST'])
def face_register():
    """Video streaming route. Put this in the src attribute of an img tag."""
    un = request.args.get('user_name', default = 'ns-abc-aaa', type = str)
    return Response(face_data(un),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def face_data(user_name):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    save_path = f'./static/face_training/dataset/{user_name}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    face_detector = cv2.CascadeClassifier('./static/face_training/haarcascade_frontalface_default.xml')
    count = 0
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
            print("얼굴등록 완료")
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
    detector = cv2.CascadeClassifier("./static/face_training/haarcascade_frontalface_default.xml")
    
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
        cv2.imshow('training face',image)
        cv2.waitKey(100)
        cv2.destroyAllWindows()

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

# 얼굴 인식 등록 end

# 얼굴 인식 로그인 start

# 4. faceID 로그인
# 정해진 시간안에 일치율이 넘으면 그냥 로그인
# 안되면 id/pw 로그인 창으로 넘어가기
@app.route('/face_login')
def face_login():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(face_model_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def face_model_gen():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./static/face_training/train_data/user_face_model.yml')
    
    cascadePath = "./static/face_training/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    
    user_list = pd.read_csv('./static/face_training/model_user_list.csv')
    names = user_list['user'].tolist()
    
    CAMERA_DEVICE_ID = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
    
    # video width
    cap.set(3, 640)
    # video height
    cap.set(4, 480)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    font = cv2.FONT_HERSHEY_SIMPLEX
    minW = 0.1*cap.get(3)
    minH = 0.1*cap.get(4)
    
    frame_cnt=0
    confidence_cnt=0
    while (frame_cnt<=fps*180): # 제한시간설정
        frame_cnt+=1
        if confidence_cnt > fps*180/3:
            print("login 성공")
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

# 얼굴 인식 로그인 end