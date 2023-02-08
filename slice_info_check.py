import ffmpeg
import os
import pandas as pd

# list 매개변수로 입력받은 리스트의 목록을 출력하고 인덱스를 입력받아 해당 인덱스의 원소를 문자열로 반환하는 함수.
# msg 에는 선택시 띄울 메세지를 입력
def list_selection(list, msg):
    for index, elem in enumerate(list):
        print(f'[{index}] 입력시 [{elem}] 선택')
    sel = int(input(f'\n{msg}'))
    return str(list[sel])

# 결과 폴더 내 존재하는 폴더 목록 가져오기
# 결과 폴더 경로 세팅
extracted_folder_path = './extracted'

# 결과 폴더 경로내 존재하는 하위 폴더 목록 가져오기
extracted_folders = os.listdir(extracted_folder_path)
file_list_folder = [folder for folder in extracted_folders if os.path.isdir(os.path.join(extracted_folder_path, folder))]
file_list_folder.sort()
print(f'분석한 비디오 데이터 폴더 목록 : {file_list_folder}\n')

# 분석할 데이터 폴더 입력받아 경로 세팅
folder_name = list_selection(file_list_folder, '>>> slice info 분석을 수행할 데이터 폴더 이름을 입력하세요 : ')
target_file_path = os.path.join(extracted_folder_path, folder_name, 'frame_slice_info.csv')

if os.path.isfile(target_file_path):
    # 분석할 비디오에 필요한 데이터 가져오기
    print(f'\n{target_file_path} 경로에서 frame_slice_info.csv파일을 읽어옵니다.\n')
else:
    print(f'경로에 frame_slice_info.csv 파일이 존재하지 않습니다. video_slice_check.py를 먼저 실행하세요.')
    exit()

df = pd.read_csv(f'{target_file_path}')
print(f"[데이터 개요] : {df['duration'].count()} 개의 포즈가 분류되었습니다.\n{df}\n...")

while 1:
    time_threshold = int(input('\n>>> 필터링 할 기준 시간(초)값을 입력하세요. (0 입력시 종료) : '))
    if time_threshold == 0:
        break

    print(f'>>> {time_threshold} 초 이상 유지된 동작만 카운트 합니다.\n')

    test_df = df[df['duration']>=time_threshold]
    print(f"[데이터 개요] : {test_df['duration'].count()} 개의 포즈가 분류되었습니다.\n{test_df}\n...")