from matplotlib import pyplot as plt
import pandas as pd
import os
# import numpy as np
# import pylab as p

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
file_list_folder = [folder for folder in extracted_folders if
                    os.path.isdir(os.path.join(extracted_folder_path, folder))]
file_list_folder.sort()
print(f'분석한 비디오 Landmark 데이터 목록 : {file_list_folder}\n')

# 분석할 데이터 폴더 입력받아 경로 세팅
# folder_name = input('>>> diff 분석을 수행할 데이터 폴더 이름을 입력하세요 : ')
folder_name = list_selection(file_list_folder, '>>> Visualization을 수행할 데이터 폴더 이름을 입력하세요 : ')
target_file_path = os.path.join(extracted_folder_path, folder_name)
print(f'\n{target_file_path} 경로에서 다음의 csv파일을 읽어옵니다.')

# 입력 받은 데이터 폴더에 존재 하는 csv데이터 목록 출력
file_list = os.listdir(target_file_path)
file_list_csv = [file for file in file_list if file.endswith(".csv")]
print(f'{file_list_csv}\n')

if os.path.isfile(f'{target_file_path}/data_x.csv'):
    df_x = pd.read_csv(f'{target_file_path}/data_x.csv')
    df_x.plot(title = 'x-factor', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/data_x.png')

if os.path.isfile(f'{target_file_path}/data_y.csv'):
    df_y = pd.read_csv(f'{target_file_path}/data_y.csv')
    df_y.plot(title = 'y-factor', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/data_y.png')

if os.path.isfile(f'{target_file_path}/data_z.csv'):
    df_z = pd.read_csv(f'{target_file_path}/data_z.csv')
    df_z.plot(title = 'z-factor', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/data_z.png')

if os.path.isfile(f'{target_file_path}/data_v.csv'):
    df_v = pd.read_csv(f'{target_file_path}/data_v.csv')
    df_v.plot(title = 'v-factor', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/data_v.png')

if os.path.isfile(f'{target_file_path}/euclidean_diff.csv'):
    df_euclidean_diff = pd.read_csv(f'{target_file_path}/euclidean_diff.csv')
    df_euclidean_diff.plot(title = 'vector_graph std_norm', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/euclidean_diff.png')

if os.path.isfile(f'{target_file_path}/savgol_diff.csv'):
    df_savgol_diff = pd.read_csv(f'{target_file_path}/savgol_diff.csv')
    df_savgol_diff.plot(title = 'vector_graph filtered : SAVGOL', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/savgol_diff.png')

if os.path.isfile(f'{target_file_path}/ema_diff.csv'):
    df_ema_diff = pd.read_csv(f'{target_file_path}/ema_diff.csv')
    df_ema_diff.plot(title = 'vector_graph filtered : EMA', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/ema_diff.png')

if os.path.isfile(f'{target_file_path}/sma_diff.csv'):
    df_sma_diff = pd.read_csv(f'{target_file_path}/sma_diff.csv')
    df_sma_diff.plot(title = 'vector_graph filtered : SMA', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/sma_diff.png')

if os.path.isfile(f'{target_file_path}/cma_diff.csv'):
    df_cma_diff = pd.read_csv(f'{target_file_path}/cma_diff.csv')
    df_cma_diff.plot(title = 'vector_graph filtered : CMA', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/cma_diff.png')

if os.path.isfile(f'{target_file_path}/gau_diff.csv'):
    df_gau_diff = pd.read_csv(f'{target_file_path}/gau_diff.csv')
    df_gau_diff.plot(title = 'vector_graph filtered : GAUSSIAN', kind = 'line', figsize=(50,20))
    plt.savefig(f'{target_file_path}/gau_diff.png')

# if os.path.isfile(f'{target_file_path}/pose_sections.csv'):
#     df_gau_diff = pd.read_csv(f'{target_file_path}/pose_sections.csv')
#     df_gau_diff.plot(title = 'pose_sections.csv', kind = 'line', figsize=(50,20))
#     plt.savefig(f'{target_file_path}/pose_sections.png')

plt.legend()
plt.show()