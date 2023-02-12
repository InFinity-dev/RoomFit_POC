import cv2
import numpy as np
from PIL import Image
import os
import pandas as pd

# 디렉토리 재귀 탐색으로 폴더만 리스트로 반환해주는 함수 : 레거시 함수
# def folder_only(dirname):
#     dir_list = []
#     for filename in os.listdir(dirname):
#         file_path = os.path.join(dirname,filename)
#         if os.path.isdir(file_path):
#             dir_list.append(os.path.basename(file_path))
#             folder_only(file_path)
#
#     return dir_list

# prefix 'user_' 제거해주는 함수 : 레거시 함수
# def labelize(list):
#     labelized = []
#     for elem in list:
#         elem = str(elem.replace("user_", ""))
#         print(elem)
#         labelized.append(elem)
#
#     return labelized

# 경로 설정
dataset_path = './dataset'
user_list = os.listdir(dataset_path)
user_list = [user for user in user_list if
                    os.path.isdir(os.path.join(dataset_path, user))]
# user_list = folder_only(dataset_path)
user_list.sort()
print(user_list)

user_list_pd = pd.DataFrame(user_list, columns=['user'])
user_list_pd.to_csv('./model_user_list.csv', index=True)
print(f'>>> 유저 리스트가 "model_user_list.csv" 로 저장되었습니다.')

# 얼굴 인식 모델 저장 경로 세팅
trainset_path = './train_data'

# 모델 존재 여부 판별
model_exist_check = os.path.join(trainset_path, 'user_face_model.yml')

if os.path.isfile(model_exist_check):
    prompt = input(f'\n모델을 재학습 하시겠습니까 (y/n) : ')
    if prompt == 'y':
        print(f'기존 모델을 덮어 씁니다.')
    elif prompt == 'n':
        print(f'모델 생성을 종료 합니다.')
        exit()


# 1. LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# 2. Eigen Face Recognizer
#recognizer = cv2.face.EigenFaceRecognizer_create()

# 3. Fisher Face Recognizer
#recognizer = cv2.face.FisherFaceRecognizer_create()

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

# 모델 생성에 사용된 유저 수
cat_count = len(np.unique(id))
valid_count = len(id)
model_ratio = round((valid_count/img_count) * 100,2)

print(f'\n>>> {img_count} 개의 얼굴을 처리하였습니다.')
print(f'\n>>> {cat_count} 명에 대한 얼굴 인식 모델이 생성 되었습니다.')
print(f'\n>>> {valid_count} 개의 패턴을 대상으로 모델이 생성 되었습니다.')
print(f'\n>>> 모델 정확도 {model_ratio}%')