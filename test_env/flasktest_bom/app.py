from flask import Flask, request, render_template,jsonify,session,Response
from flask_mysqldb import MySQL
from datetime import datetime, timedelta
from functools import wraps
from PIL import Image
import pandas as pd
import numpy as np
import MySQLdb.cursors
import hashlib
import jwt, os, cv2

#----Flask 선언, mongodb 연결 ----
app=Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'TEST'

app.config["SECRET_KEY"]="jungle"
mysql = MySQL(app)

#----decorate----
def token_required(func):
    @wraps(func)
    def decorated(*args, **kwargs) :
        token = request.args.get('token')
        if not token:
            return jsonify({'token' : 'token is missing'})
        try:
            payload = jwt.decode(token,app.config['SECRET_KEY'])
        except:
            return jsonify({'alert':'invalid token'})
    return decorated

#---- Test ----
@app.route('/auth')
@token_required
def auth():
    return 'jwt is verified'

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
            
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user WHERE id = % s', (id_receive, ))
            account = cursor.fetchone()
            
            if account:
                msg = 'It is a registerd ID'
            else:
                cursor.execute('INSERT INTO user VALUES (% s, % s)', (id_receive, pw_receive, ))
                mysql.connection.commit()
                msg = '성공적으로 가입되었습니다!'
                return render_template('face_register.html', msg = msg, idd=id_receive) # home으로 연동
            
        else:
            msg = '모든 항목을 기입해 주세요!'
    return render_template('register.html', msg = msg)


# 2. id/pw 로그인
@app.route('/idpw_login', methods =['GET', 'POST'])
def idpw_login():
    auth = request.authorization
    msg = ''
    if request.method == 'POST':
        if 'id' in request.form and 'password' in request.form:
            id_receive = request.form['id']
            pw_receive = request.form['password']
            pw_hash = hashlib.sha256(pw_receive.encode('utf-8')).hexdigest()
            
            cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            cursor.execute('SELECT * FROM user WHERE id = % s AND pw_hashed = % s', (id_receive, pw_receive, ))
            account = cursor.fetchone()

            if account:
                session['loggedin'] = True
                session['id'] = account['id']
                
                token = jwt.encode({
                    'id':id_receive,
                    'expiration': str(datetime.utcnow() + timedelta(seconds=60 * 60 * 24))
                },
                    app.config['SECRET_KEY'],
                    algorithm='HS256')
                
                msg = '성공적으로 로그인 되었습니다!'
                return render_template('login.html', msg=msg)
                
            else:
                msg = '휴대폰 번호 / 비밀번호가 일치하지 않습니다!'
                
    return render_template('login.html', msg = msg)


# 3. Face ID 등록
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
    
    save_path = f'./dataset/{user_name}'
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
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
    dataset_path = './dataset'
    user_list = os.listdir(dataset_path)
    user_list = [user for user in user_list if
                        os.path.isdir(os.path.join(dataset_path, user))]
    user_list.sort()
    user_list_pd = pd.DataFrame(user_list, columns=['user'])
    user_list_pd.to_csv('./model_user_list.csv', index=True)
    
    trainset_path = './train_data'
    model_exist_check = os.path.join(trainset_path, 'user_face_model.yml')
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
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
    recognizer.read('train_data/user_face_model.yml')
    
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)
    
    user_list = pd.read_csv('model_user_list.csv')
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
            


if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)