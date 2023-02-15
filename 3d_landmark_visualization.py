import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import os

# 연결할 랜드마크 좌표끼리 그룹화
LANDMARK_GROUPS = [
    [8, 6, 5, 4, 0, 1, 2, 3, 7],   # 눈
    [10, 9],                       # 입
    [11, 13, 15, 17, 19, 15, 21],  # 오른쪽 팔
    [11, 23, 25, 27, 29, 31, 27],  # 우측 신체
    [12, 14, 16, 18, 20, 16, 22],  # 왼쪽 팔
    [12, 24, 26, 28, 30, 32, 28],  # 왼쪽 신체
    [11, 12],                      # 어깨
    [23, 24],                      # 손목
]

def plot_world_landmarks(ax, landmarks, landmark_groups=LANDMARK_GROUPS):

    # 추출된 랜드마크가 없을때 예외처리
    if landmarks is None:
        return

    ax.cla()

    # z축은 반전
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(1, -1)

    for group in landmark_groups:
        plotX, plotY, plotZ = [], [], []

        plotX = [landmarks.landmark[i].x for i in group]
        plotY = [landmarks.landmark[i].y for i in group]
        plotZ = [landmarks.landmark[i].z for i in group]

        ax.plot(plotX, plotZ, plotY)

    plt.pause(.001)
    return

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

# plot 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# mediapipe 변수 설정
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

with mp_pose.Pose(
        min_tracking_confidence=0.5,
        min_detection_confidence=0.5,
        model_complexity=1,
        smooth_landmarks=True,
) as pose:
    while cap.isOpened():

        ret, img = cap.read()

        if not ret:
            print("영상 스트림이 종료되었습니다.")
            break

        # 랜드마크 추출
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 랜드마크로 부터 3D plot 그리기
        plot_world_landmarks(ax, results.pose_world_landmarks)

        # draw image
        cv2.imshow("MediaPipePose", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
