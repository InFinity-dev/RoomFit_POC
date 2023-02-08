import cv2
import mediapipe as mp
from collections import deque
import os

# list 매개변수로 입력받은 리스트의 목록을 출력하고 인덱스를 입력받아 해당 인덱스의 원소를 문자열로 반환하는 함수.
# msg 에는 선택시 띄울 메세지를 입력
def list_selection(list, msg):
    for index, elem in enumerate(list):
        print(f'[{index}] 입력시 [{elem}] 선택')
    sel = int(input(f'\n{msg}'))
    return str(list[sel])

# 움직이는지 판단 기준
criteria = 0.55 # 이전 프레임과 차이
criteria_frame = 20 # 판단을 위한 이전 프레임
criteria_count = 8 # criteria_frame 에서 criteria_count개 보다 많은 'O' 요구 

# 행동 분리 기준
x_count = 5 # 중간에 x 값이 몇개까지 있어도 되는가
scean_count = 120 # 안움직이는 프레임이 몇개 지속되야 행동인가

# 비디오 스트림 사용시
# 타겟 폴더 경로 설정
folder_path = './target_video'
file_list = os.listdir(folder_path)
file_list_mp4 = [file for file in file_list if file.endswith(".mp4")]

# 타겟 폴더 내 mp4 파일 리스트 오름차순 정렬 후 출력
file_list_mp4.sort()
print(f'처리할 영상 목록 : {file_list_mp4}\n')

# 분석할 데이터 폴더 입력받아 경로 세팅
# target_mp4 = input('>> 분석을 수행할 영상 이름을 입력하세요 (확장자 포함 *.mp4): ')
target_mp4 = list_selection(file_list_mp4, '>> 분석을 수행할 영상을 선택하세요. (확장자 *.mp4) : ')

video_name = os.path.join(folder_path, target_mp4)
# print(video_name)

def decide(x, y, z):
    gap_x = round(before_x.popleft() / x, 4)
    gap_y = round(before_y.popleft() / y, 4)

    if(gap_x >= 1) : gap_x = round((gap_x - 1) * 100, 4)
    else : gap_x = round((1 - gap_x) * 100, 4)
    if(gap_y >= 1) : gap_y = round((gap_y - 1) * 100, 4)
    else : gap_y = round((1 - gap_y) * 100, 4)

    gap_z = round(z, 4)

# 이전 프레임보다 (criteria)% 차이가 난다면 움직이는 것으로 판정
    if gap_x > criteria - gap_z :
        # print(now_frame, ': x')
        return False
    elif gap_y > criteria - gap_z :
        # print(now_frame, ': y')
        return False
    # elif gap_z > 20 : 
    #     print('z') 
    #     return False
    else : return True

mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(video_name)

now_frame = 0
first_flag = True

before_x = deque()
before_y = deque()

# 움직이는지 판단하는 것에 이전 (criteria_frame) 프레임을 이용
result_s = deque()
for _ in range(criteria_frame) : 
    result_s.append(1)

result_list = []

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose :
    while cap.isOpened() :
        success, image = cap.read()
        if not success :
            break

        image.flags.writeable = False
        image = cv2.flip(image, 1)
        results = pose.process(image)

        no_move = True
        if results.pose_landmarks :
            for i in range(33) :
                before_x.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x)
                before_y.append(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y)
                if first_flag :
                    if i == 32 : first_flag = False
                    continue
                if no_move :
                    if results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].visibility > 0.7 :
                        no_move = decide(results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].y, results.pose_landmarks.landmark[mp_pose.PoseLandmark(i).value].z)
                    else :
                        before_x.popleft()
                        before_y.popleft()
                else :
                    before_x.popleft()
                    before_y.popleft()
        
        if no_move :
            result_s.popleft()
            result_s.append(1)
        else :
            result_s.popleft()
            result_s.append(0)
        
        if sum(result_s) > criteria_count : # 최종 판단
            result = 'O'
        else :
            result = 'X'

        result_list.append(result)

        cv2.putText(image, str(now_frame), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.putText(image, str(result), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(3) & 0xFF == 27:
            break
        now_frame = now_frame + 1
 
cap.release()

before = result_list[0]
scean = []
scean_result = []
start = 0
for i in range(1, len(result_list)) :
    if(result_list[i]) == 'O' :
        if before == 'X' :
            start = i
    else :
        if before == 'O' :
            end = i-1
            scean.append([start, end])
    before = result_list[i]

if result_list[-1] == 'O' and before == 'O' :
    end = len(result_list)-1
    scean.append([start, end])

for j in range(len(scean)-1) :
    if(scean[j+1][0] - scean[j][1]) < x_count :
        scean[j+1][0] = scean[j][0]
        scean[j][0] = -1

for k in range(len(scean)) :
    if scean[k][0] != -1 and scean[k][1] - scean[k][0] > scean_count :
        scean_result.append(scean[k])

for l in range(len(scean_result)) :
    print(f'행동{l}: {scean_result[l][0]}~{scean_result[l][1]}')