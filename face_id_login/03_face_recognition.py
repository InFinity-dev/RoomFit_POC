import cv2
import pandas as pd

# 1. LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. Eigen Face Recognizer
#recognizer = cv2.face.EigenFaceRecognizer_create()

# 3. Fisher Face Recognizer
#recognizer = cv2.face.FisherFaceRecognizer_create()

# 모델 경로 설정
recognizer.read('train_data/user_face_model.yml')

# cascade 설정
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# # 모델 유저 리스트
# id = 0

# model_user_list.csv 로 부터 얼굴인식 모델 생성시 기여된 유저 정보 리스트로 변환
user_list = pd.read_csv('model_user_list.csv')

names = user_list['user'].tolist()
print(names)


CAMERA_DEVICE_ID = 0
cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

# video width
cap.set(3, 640)
# video height
cap.set(4, 480)

# 오버레이 폰트 설정
font = cv2.FONT_HERSHEY_SIMPLEX

# 얼굴로 인식할 최소 크기 설정
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

while True:
    ret, img = cap.read()

    # 카메라 좌우 반전
    img = cv2.flip(img, 1)

    # 그레이 스케일로 변경
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

        # confidence = 0이면 완벽 매칭
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            print(f'{id} 인식됨 : 인식 정확도 {confidence}')
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
            print(f'등록되지 않은 사용자')

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera', img)

    # ESC키 입력시 break
    if cv2.waitKey(1) & 0xFF == 27:
        break

print("\n프로그램 종료")
cap.release()
cv2.destroyAllWindows()
