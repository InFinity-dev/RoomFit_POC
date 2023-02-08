from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.signal import find_peaks
import os

# list 매개변수로 입력받은 리스트의 목록을 출력하고 인덱스를 입력받아 해당 인덱스의 원소를 문자열로 반환하는 함수.
# msg 에는 선택시 띄울 메세지를 입력
def list_selection(list, msg):
    for index, elem in enumerate(list):
        print(f'[{index}] 입력시 [{elem}] 선택')
    sel = int(input(f'\n{msg}'))
    return str(list[sel])

def minmax_norm(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    return (arr - arr_min) // (arr_max - arr_min)


def stand_norm(arr):
    arr_mean = np.nanmean(arr)
    arr_std = np.nanstd(arr)
    return (arr - arr_mean) // arr_std


def posingFilter(arr, meanV, fps):
    posing_in = []
    posing_out = []
    for ft in range(1, len(arr)):
        if (arr[ft - 1] > meanV) and (arr[ft] <= meanV):
            posing_in.append(ft)
        elif (arr[ft - 1] < meanV) and (arr[ft] >= meanV):
            posing_out.append(ft)

    posing = []
    if posing_in[0] < posing_out[0]:
        for p_idx in range(len(posing_in)):
            if posing_out[p_idx] - posing_in[p_idx] > fps * 2:
                posing.append([posing_in[p_idx], posing_out[p_idx]])

    return


# 결과 폴더 내 존재하는 폴더 목록 가져오기
# 결과 폴더 경로 세팅
extracted_folder_path = './extracted'

# 결과 폴더 경로내 존재하는 하위 폴더 목록 가져오기
extracted_folders = os.listdir(extracted_folder_path)
extracted_folders.sort()
print(f'분석한 비디오 Landmark 데이터 목록 : {extracted_folders}\n')

# 분석할 데이터 폴더 입력받아 경로 세팅
# folder_name = input('>>> diff 분석을 수행할 데이터 폴더 이름을 입력하세요 : ')
folder_name = list_selection(extracted_folders,'>>> diff 분석을 수행할 데이터 폴더 이름을 입력하세요 : ')
target_file_path = os.path.join(extracted_folder_path, folder_name)
print(f'\n{target_file_path} 경로에서 다음의 csv파일을 읽어옵니다.')

# 입력 받은 데이터 폴더에 존재 하는 csv데이터 목록 출력
file_list = os.listdir(target_file_path)
file_list_csv = [file for file in file_list if file.endswith(".csv")]
print(f'{file_list_csv}\n')

# 데이터 프레임에 csv 파일 데이터 불러오기
df_x = pd.read_csv(f'{target_file_path}/data_x.csv')
df_y = pd.read_csv(f'{target_file_path}/data_y.csv')
df_v = pd.read_csv(f'{target_file_path}/data_v.csv')

# pandas 데이터 프레임 numpy로 변환
np_x = df_x.to_numpy()
np_y = df_y.to_numpy()
np_v = df_v.to_numpy()

# Visibility Factor 벡터 연산에 활용 위하여 Concat
np_v_temp = np_v.copy()
np_v_temp2 = np_v_temp[:-1]
np_v_temp1 = np_v_temp[1:]
# np_v 에 앞,뒤 원소 중 Visibility 가 높은 원소로 치환
np_v = np.where(np_v_temp1 > np_v_temp2, np_v_temp2, np_v_temp1)
v_vector = np.concatenate((np_v, np_v), axis=1)

# X factor, Y factor 벡터화 하기 위해 V factor 연속 으로 Concat
np_vector = np.concatenate((np_x, np_y), axis=1)

# 유클리드 벡터 연산
# 변화량 구하기 (1칸 shift 하여 차이 구함)
np_vector2 = np_vector.copy()
sub_temp = np_vector2[1:] - np_vector[:-1]

# 구한 Diff 에 해당 프레임에서의 Visibility Factor 가 0.9 이하인 항목에 대해 0으로 처리
sub_temp = np.where(v_vector < 0.9, 0, sub_temp)

# 벡터화
# 제곱 합 구하기
sum_sq = np.square(sub_temp).sum(axis=1)
# 제곱 합들의 제곱근 값 구하기
sqrt_of_sum_sq = np.sqrt(sum_sq)
# Normalization
result = stand_norm(sqrt_of_sum_sq)
# print(result.shape)

# numpy 연산한 결과 pandas 데이터 프레임 화
result_df = pd.DataFrame(result)

# Smoothing Filter 실험

# Smoothing with 사비츠키-골레이(Savitzky-Golay aka.SAVGOL)
result_filtered_savgol = result_df.apply(savgol_filter, window_length=31, polyorder=2)
# Smoothing with Simple Moving Average Method (SMA)
result_filtered_sma = result_df.rolling(100).mean()
# Smoothing with Cumulative Moving Average Method (CMA)
result_filtered_cma = result_df.expanding().mean()
# Smoothing with Exponential Moving Average Method (EMA)
result_filtered_ema = result_df.ewm(span=100).mean()
# Smoothing with Gaussian (GAU)
result_filtered_gau = result_df.rolling(window=10, win_type='gaussian').mean(std=1)

print(f'필터링 적용 전 평균(mean) 값 : {np.mean(result)}')
print(f'필터링 적용 전 표준 편차(std) 값 : {np.std(result)}\n')

print(f'필터링 적용 후 평균(mean) 값 : {result_filtered_savgol.mean()}')
print(f'필터링 적용 후 표준 편차(std) 값 : {result_filtered_savgol.std()}')

result_df.to_csv(f'{target_file_path}/euclidean_diff.csv', index=False)
result_filtered_savgol.to_csv(f'{target_file_path}/savgol_diff.csv', index=False)
result_filtered_sma.to_csv(f'{target_file_path}/sma_diff.csv', index=False)
result_filtered_cma.to_csv(f'{target_file_path}/cma_diff.csv', index=False)
result_filtered_ema.to_csv(f'{target_file_path}/ema_diff.csv', index=False)
result_filtered_gau.to_csv(f'{target_file_path}/gau_diff.csv', index=False)
print(f'\n>>> {target_file_path} 경로에 diff_vector 데이터가 csv 파일로 저장되었습니다.\n')

# Peak 값 찾기
result_filtered_savgol2 = result_filtered_savgol.T
savgol_seq = result_filtered_savgol2.to_numpy()[0]
peaks, properties = find_peaks(savgol_seq, prominence=(0.5, None))
print(f'max prominences 값 : {properties["prominences"].max()}')
print(f'diff_vector 그래프를 출력합니다.')

plt.figure(figsize=(50,20))
plt.plot(result_filtered_savgol)
plt.plot(peaks, savgol_seq[peaks], "x", label='peak_point')
plt.hlines(result_filtered_savgol.mean(), 0, result.shape[0], color="red", linewidth=1, label='sav_filtered mean')
plt.hlines(result_filtered_savgol.mean() + result_filtered_savgol.std(), 0, result.shape[0], color="green", linewidth=1, label='sav_filtered mean-std')
# plt.hlines(result_filtered_savgol.mean() - result_filtered_savgol.std(), 0, result.shape[0], color="green", linewidth=1)
# plt.plot(result_filtered_sma)
# plt.plot(result_filtered_cma)
# plt.plot(result_filtered_ema)
# plt.plot(result_filtered_gau)
plt.legend()
plt.show()