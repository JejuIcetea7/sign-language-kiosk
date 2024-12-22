import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

# 유클리드 거리 계산 함수
def calculate_euclidean_distance(point1, point2):
    """ 두 점 사이의 유클리드 거리를 계산하는 함수 """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
selected_keypoints = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

# 비디오를 처리하고 키포인트 간 거리와 두 손 중심 간 거리를 추출
def process_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(total_frames, int(3 * frame_rate))  # 3초에 해당하는 최대 프레임 수

    frame_count = 0
    data = []  # 거리 정보를 저장할 리스트

    # 비디오가 열려있는 동안 반복하여 처리
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:  # 영상 3초 넘기면 컷
            break

        frame_count += 1

        if frame_count % (frame_rate // 2) == 0:  # 0.5초 간격
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            height, width, _ = frame.shape
            left_hand_distances = [None] * 55  # 왼손 거리 기본값을 None으로 설정
            right_hand_distances = [None] * 55  # 오른손 거리 기본값을 None으로 설정
            distance_between_hands = None  # 양손 중심 거리 기본값

            if results.multi_hand_landmarks:
                hand_centers = []

                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # 선택된 키포인트만 픽셀 좌표로 변환
                    landmarks = [(hand_landmarks.landmark[i].x * width, hand_landmarks.landmark[i].y * height) for i in selected_keypoints]
                    
                    # 선택된 키포인트 간의 모든 쌍에 대해 거리 계산 (총 55개)
                    distances = [
                        calculate_euclidean_distance(landmarks[i], landmarks[j])
                        for i in range(len(landmarks))
                        for j in range(i + 1, len(landmarks))
                    ]
                    
                    # 왼손과 오른손을 구분하여 거리 정보를 저장
                    if idx == 0:
                        left_hand_distances = distances  # 왼손 거리 정보 저장
                    elif idx == 1:
                        right_hand_distances = distances  # 오른손 거리 정보 저장

                    # 손의 중심 좌표 계산
                    hand_center_x = sum([lm[0] for lm in landmarks]) / len(landmarks)
                    hand_center_y = sum([lm[1] for lm in landmarks]) / len(landmarks)
                    hand_centers.append((hand_center_x, hand_center_y))

                # 두 손이 모두 인식된 경우에만 중심 간 거리 계산
                if len(hand_centers) == 2:
                    distance_between_hands = calculate_euclidean_distance(hand_centers[0], hand_centers[1])

            # 왼손 거리, 오른손 거리, 두 손 중심 거리, 라벨 추가
            row_data = left_hand_distances + right_hand_distances + [distance_between_hands, label]
            data.append(row_data)

    cap.release()
    return data

def process_all_videos(root_directory, output_csv):
    """ 모든 폴더의 동영상을 처리하고 CSV 파일로 저장하는 함수 """
    all_data = []

    # 모든 폴더 탐색
    for label in os.listdir(root_directory):
        folder_path = os.path.join(root_directory, label)
        if os.path.isdir(folder_path):
            # 폴더 내 모든 비디오 파일 탐색
            for video_file in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video_file)
                if video_file.lower().endswith(('.mp4', '.avi', '.mov')):  # 비디오 파일 확장자 체크
                    print(f"Processing {video_path}...")
                    video_data = process_video(video_path, label)
                    all_data.extend(video_data)

    # 데이터프레임 생성 및 CSV 파일로 저장
    if all_data:
        # 컬럼 이름 설정: 왼손 거리 55개 + 오른손 거리 55개 + 두 손 중심 거리 1개 + 라벨 1개
        columns = (
            [f"Left_Distance_{i}" for i in range(55)] +
            [f"Right_Distance_{i}" for i in range(55)] +
            ["Distance_Between_Hands", "Label"]
        )
        df = pd.DataFrame(all_data, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")
    else:
        print("No data to save.")

# 루트 디렉토리 및 출력 CSV 파일 설정
root_directory = "sign_language_DT"
output_csv = "final.csv"
process_all_videos(root_directory, output_csv)
