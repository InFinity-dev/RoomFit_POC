import cv2
import mediapipe as mp
import pandas as pd
import ffmpeg
import datetime
import os

def run():
    print("추출 시작")
    # mediapipe 변수 설정
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    # model_complexity customized
    pose = mp_pose.Pose(model_complexity=2)

    # 카메라 피드 사용시
    # CAMERA_DEVICE_ID = 0
    # cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

    # 비디오 스트림 사용시
    # 타겟 폴더 경로 설정
    folder_path = './static/target_video'
    file_list = os.listdir(folder_path)
    file_list_mp4 = [file for file in file_list if file.endswith(".mp4")]

    # 타겟 폴더 내 mp4 파일 리스트 오름차순 정렬 후 출력
    file_list_mp4.sort()
    print(f'>>> static/target_video 폴더 내의 모든 .mp4 영상의 랜드마크를 추출합니다.')
    print(f'처리할 영상 목록 : {file_list_mp4}\n')

    # 결과 폴더 내 존재하는 폴더 목록 가져오기
    extracted_folder_path = './static/extracted'
    extracted_files = os.listdir(extracted_folder_path)

    # 타겟 경로에 있는 영상에 대해 순차 처리
    for_counter = 1
    for target_mp4 in file_list_mp4:
        # 경로명 + 파일 이름 담을 변수 선언
        target_file = os.path.join(folder_path, target_mp4)
        print(f'>>>>> [{for_counter}/{len(file_list_mp4)} 번째 작업.] <<<<<<')
        print(f'{target_file} 파일을 불러옵니다.')

        # 이미 처리한 영상 예외 처리
        vid_name = target_mp4.rsplit('.')[0]
        if vid_name in extracted_files:
            print(f'{target_mp4}는 이미 처리된 영상 입니다. 다음 작업으로 넘어갑니다.\n')
            for_counter += 1
            continue

        # 추출 시작 시간 타임스탬프 출력
        start = datetime.datetime.now()
        print(f'>>> {target_mp4}에 대한 처리를 시작합니다. \n시작시간 : {start}')

        # cap 변수에 타켓 영상 바인딩
        cap = cv2.VideoCapture(target_file)

        # 영상 열기 오류시 예외처리
        if not cap.isOpened():
            print(f'{target_file} 파일을 여는데 문제가 생겼습니다.')
            exit()

        # 영상 파일 정보 가져오기
        info = ffmpeg.probe(target_file)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 영상 정보 출력하기
        print(f'>>> 파일 정보')
        print(f"{info['format']['duration']} (초) 의 재생시간을 가지는 비디오 입니다.")
        print(f'{frame_count} 개의 프레임을 가지는 비디오 입니다.')
        print(f'{width} x {height}의 해상도를 가지는 비디오 입니다.')
        print('FPS:', fps)
        print('...')

        # 랜드마크 저장할 리스트 선언
        # mediapipe 랜드마크 개수, 컬럼 명 추출
        columns_py = []
        for i in range(len(mp_pose.PoseLandmark)):
            columns_py.append(mp_pose.PoseLandmark(i).name)
        # 랜드마크 x,y,z,vis 요소 담을 리스트 선언
        total_x = []
        total_y = []
        total_z = []
        total_vis = []

        # 프레임 카운트 변수 선언
        count = 1

        # 타겟 영상 프레임 단위로 읽어오기
        while cap.isOpened():
            # 영상 비정상 종료 예외 처리
            try:
                ret, img = cap.read()
                if not ret:
                    print(f"{target_mp4} 영상 스트림이 종료되었습니다.")
                    break

                # 진행 사항 콘솔 로그 필요할 때 사용. 인터럽트로 인한 지연 심하므로 신중히 사용할 것.
                # now_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                # print(f'{target_mp4} 영상의 {now_frame} 번째 프레임을 처리중 입니다.')

                # 영상 이미지 리사이즈
                img = cv2.resize(img, (600, 400))

                # 해당 이미지로 처리하여 랜드마크 추출
                results = pose.process(img)

                # 랜드마크 콘솔에 출력. 인터럽트로 인한 지연 심하므로 신중히 사용할 것.
                # print(results.pose_landmarks)

            except:
                pass

            # 프레임 당 추출된 랜드마크 담을 변수 선언. 루프 순환 마다 초기화.
            per_frame_value_x = []
            per_frame_value_y = []
            per_frame_value_z = []
            per_frame_value_vis = []

            # 추출한 랜드마크 프레임 당 리스트에 담아서 전역 리스트에 append 처리.
            for i in range(len(mp_pose.PoseLandmark)):
                if results.pose_landmarks:
                    per_frame_value_x.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x)
                    per_frame_value_y.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y)
                    per_frame_value_z.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z)
                    per_frame_value_vis.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility)
                else:
                    per_frame_value_x.append(None)
                    per_frame_value_y.append(None)
                    per_frame_value_z.append(None)
                    per_frame_value_vis.append(None)

            total_x.append(per_frame_value_x)
            total_y.append(per_frame_value_y)
            total_z.append(per_frame_value_z)
            total_vis.append(per_frame_value_vis)

            # 처리한 프레임 카운트
            count += 1

        print('>>> 처리 결과')
        print(f'{target_file} 에 대해 총 {count}개의 프레임이 처리 되었습니다.')
        fin = datetime.datetime.now()
        print(f'종료 시간 : {fin}')
        print(f'걸린 시간 : {fin - start}')

        # 저장한 랜드마크 파일로 쓰기
        dx = pd.DataFrame(total_x, columns=columns_py)
        dy = pd.DataFrame(total_y, columns=columns_py)
        dz = pd.DataFrame(total_z, columns=columns_py)
        dv = pd.DataFrame(total_vis, columns=columns_py)

        save_path = f'./static/extracted/{vid_name}'
        os.mkdir(save_path)

        dx.to_csv(f'{save_path}/data_x.csv', index=False)
        dy.to_csv(f'{save_path}/data_y.csv', index=False)
        dz.to_csv(f'{save_path}/data_z.csv', index=False)
        dv.to_csv(f'{save_path}/data_v.csv', index=False)
        print(f'{vid_name}에 대한 Landmark 데이터가 csv 파일로 저장되었습니다.\n')

        for_counter += 1
