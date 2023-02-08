import cv2
import mediapipe as mp
import numpy as np
import os

# list 매개변수로 입력받은 리스트의 목록을 출력하고 인덱스를 입력받아 해당 인덱스의 원소를 문자열로 반환하는 함수.
# msg 에는 선택시 띄울 메세지를 입력
def list_selection(list, msg):
    for index, elem in enumerate(list):
        print(f'[{index}] 입력시 [{elem}] 선택')
    sel = int(input(f'\n{msg}'))
    return str(list[sel])


# 소스 선택
print('사용할 소스를 선택하세요.')
print('1 : 카메라')
print('2 : 영상')
sel_source = int(input('>> 소스 : '))
print()

# 카메라 선택 분기
if sel_source == 1:
    print('\n카메라를 선택하였습니다. ESC키 입력시 종료.')
    CAMERA_DEVICE_ID = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
# 영상 선택 분기
else:
    folder_path = './target_video'
    file_list = os.listdir(folder_path)
    file_list_mp4 = [file for file in file_list if file.endswith(".mp4")]
    file_list_mp4.sort()
    # print(f'비디오 목록 : {file_list_mp4}\n')
    # video_name = input('>> 오버레이를 출력할 비디오 이름을 입력하세요 (확장자 포함 *.mp4) : ')
    video_name = list_selection(file_list_mp4, '>> 오버레이를 출력할 비디오를 선택하세요. (확장자 *.mp4) : ')
    target_file = os.path.join(folder_path, video_name)
    print(f'\n{video_name} 영상으로 부터 랜드마크 오버레이를 실시합니다. ESC키 입력시 종료.')
    cap = cv2.VideoCapture(target_file)

# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

while cap.isOpened():
    try:
        ret, img = cap.read()
        if not ret:
            print("영상 스트림이 종료되었습니다.")
            break

        # 이미지 리사이즈
        img = cv2.resize(img, (600, 400))

        # 해당 이미지로 부터 랜드마크 추출
        results = pose.process(img)

        # 추출된 랜드마크 그리기 DrawingSpec((Blue,Green,Red))

        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        # 그려진 랜드마크 원본 영상/스트림에 오버레이

        cv2.imshow("Pose Estimation", img)

        # 추출된 랜드마크 별도 창에 그리기

        h, w, c = img.shape   # get shape of original frame
        opImg = np.zeros([h, w, c])  # create blank image with original frame size
        opImg.fill(255)  # set white background. put 0 if you want to make it black

        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2)
                               )

        cv2.imshow("Extracted Pose", opImg)

    except:
        pass

    # ESC키 입력시 break
    if cv2.waitKey(1) & 0xFF == 27:
        break
