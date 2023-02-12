import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 웹상에서는 가입시 user_id로 받아올것
user_name = input('\n>>> 사용자 ID를 입력하세요 : ')

# 사용자 이름으로 폴더 생성
save_path = f'./dataset/{user_name}'

if os.path.isdir(save_path):
    print(f'이미 존재하는 유저 입니다.')
else:
    os.mkdir(save_path)

print("\n>>> 얼굴 인식 데이터를 수집하는 중입니다. 카메라를 바라봐 주세요.")

# 얼굴 데이터 수집 카운터
count = 0

while(True):

    ret, img = cam.read()

    # 카메라 좌우 반전
    img = cv2.flip(img, 1)

    # 그레이 스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        # 얼굴이 인식되면 카운터 증가
        count += 1

        # 얼굴 캡쳐하여 받아온 유저ID 폴더 하위에 유저ID로 순차 저장
        # cv2.imwrite(f'{save_path}/' + str(user_name) + '.' + str(count).zfill(3) + ".jpg", gray[y:y+h,x:x+w])
        cv2.imwrite(f'{save_path}/' + str(count).zfill(3) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('image', img)

    # ESC키 입력시 break
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # 얼굴 사진 30장 다 모으면 break
    elif count >= 30:
         break


print("\n>>> 얼굴 인식 데이터가 수집되었습니다.")
cam.release()
cv2.destroyAllWindows()


