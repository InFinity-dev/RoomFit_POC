import pandas as pd
import cv2
import mediapipe as mp
import ffmpeg
from time import sleep
import os

# 영상 폴더 경로 설정
folder_path = './target_video'
file_list = os.listdir(folder_path)
file_list_mp4 = [file for file in file_list if file.endswith(".mp4")]

# 타겟 폴더 내 mp4 파일 리스트 오름차순 정렬 후 출력
file_list_mp4.sort()
print(f'처리할 영상 목록 : {file_list_mp4}\n')

# 분석 비디오 이름 입력받기
target_mp4 = input('>>> 분석을 수행할 영상 이름을 입력하세요 (확장자 포함 *.mp4) : ')
video_path = os.path.join(folder_path, target_mp4)
print(video_path)

# 분석 비디오와 일치하는 데이터 폴더 경로 세팅
extracted_folder_path = './extracted'
extracted_files = os.listdir(extracted_folder_path)
extracted_data_path = os.path.join(extracted_folder_path, target_mp4.rsplit('.')[0])
print(extracted_data_path)

if os.path.exists(extracted_data_path):
    # 분석할 비디오에 필요한 데이터 가져오기
    file_list = os.listdir(extracted_data_path)
    file_list_csv = [file for file in file_list if file.endswith(".csv")]
    print(f'\n{extracted_data_path} 경로에 다음과 같은 csv 파일들이 존재합니다.\n{file_list_csv}\n')
    target_data = input(f'>>> 비디오에 적용할 데이터 파일을 입력하세요. (확장자 포함 *.csv) : ')
else:
    print(f'비디오에 대한 분석데이터가 존재하지 않습니다. diff_extract_visualize.py를 먼저 실행하세요.')
    exit()

df = pd.read_csv(f'{extracted_data_path}/{target_data}')
print(f'[데이터 개요]\n{df}\n')

print(f'{video_path} 영상 스트림을 실행합니다.\n')

cap = cv2.VideoCapture(video_path)

# 비디오 정보 가져오기
info = ffmpeg.probe(video_path)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 영상 정보 출력하기
print(f'>>> 파일 정보')
print(f"{info['format']['duration']} (초) 의 재생시간을 가지는 비디오 입니다.")
print(f'{frame_count} 개의 프레임을 가지는 비디오 입니다.')
print(f'{width} x {height}의 해상도를 가지는 비디오 입니다.')
print('FPS:', fps)
print('...')

before_bg_than_mean = False
check_bool = False
temp_frame = 0.0
below_mean_frames_list = []

# 가져온 csv 데이터로부터 mean값 구하기
mean = float(df.mean())

for idx, row in df.iterrows():
    frame_idx = int(idx)
    frame_diff_amount = float(row[0])

    if before_bg_than_mean and frame_diff_amount < mean:
        if not check_bool:
            temp_frame = frame_idx
            check_bool = True

    if not before_bg_than_mean and frame_diff_amount >= mean:
        if check_bool:
            div_len = abs(frame_idx - temp_frame)
            check_bool = False
            if div_len < fps * 3:
                continue

            below_mean_frames_list.append([int(temp_frame + (div_len * 0.2)), int(frame_idx - (div_len * 0.2))])

    if check_bool and int(df.index[-1]) == frame_idx:
        div_len = abs(frame_idx - temp_frame)
        check_bool = False
        if div_len < fps * 3:
            continue

        below_mean_frames_list.append([int(temp_frame + (div_len * 0.2)), int(frame_idx - (div_len * 0.2))])

    if frame_diff_amount > mean:
        before_bg_than_mean = True
    else:
        before_bg_than_mean = False

print(f'분리된 동작 개수 : {len(below_mean_frames_list)} 개의 동작이 분리되었습니다.')
print(f'mean 값({mean}) 이하 기준으로 분리된 프레임 구간')


for elem in below_mean_frames_list:

    # 프레임을 분,초로 변환
    sec_start = int(elem[0]/fps)
    sec_end = int(elem[1]/fps)
    min_start = sec_start // 60
    sec_start %= 60
    min_end = sec_end // 60
    sec_end %= 60

    print(f'프레임 구간 : {elem} -> {min_start}분 {sec_start}초 ~ {min_start}분 {sec_start}초')

# mediapipe 변수 설정
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

if not cap.isOpened():
    print("다음의 파일을 여는데 문제가 생겼습니다.")
    exit()

frame_idx = 0
while cap.isOpened():
    try:
        ret, img = cap.read()
        if not ret:
            print("\n영상 스트림이 종료되었습니다.")
            break

        now_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if below_mean_frames_list[frame_idx][0] <= now_frame <= below_mean_frames_list[frame_idx][1]:
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
            cv2.putText(img, str(frame_idx + 1), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            cv2.imshow("Pose Estimation", img)


        if now_frame >= below_mean_frames_list[frame_idx][1]:
            frame_idx += 1
            sleep(2)
            continue

    except:
        pass

    cv2.waitKey(10)