import torch
import cv2
import numpy as np
import mediapipe as mp
import time
import pandas as pd
import torch.nn as nn


"""
이 코드가 하는 일
1. 웹캠 열기
2. 동작의 시작과 끝을 지정 (키보드 S키를 이용해서 눌림과 동시에 시작하여 3초간 입력으로 간주)
3. 인식된 약 3~4초간의 지정된 손가락 키포인트간 거리와 양 손 사이의 거리를 실시간 저장
3-2. 저장된 데이터를 학습시킬때의 데이터 가공 모양으로 전처리해서 모델에 예측을 시켜야함 -> 전처리 함수 만들기
4. 저장된 데이터를 미리 로드한 모델을 이용해서 예측하고 예측 결과를 웹캠 화면 우측 상단에 표시
"""


hidden_size = 64
num_layers = 2

LABELS = [
    "add", "bulgogi", "chicken", "cider", "coke", "four", "hamburger", 
    "help", "here", "no", "one", "payment", "please", "potato", "set", 
    "squid", "three", "togo", "tomato", "two", "without", "yes"
]

#모델 구조
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # LSTM에 입력
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)
        c_0 = torch.zeros(num_layers, x.size(0), hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))  # LSTM의 출력
        out = out[:, -1, :]  # 마지막 time step의 출력 사용
        out = self.fc(out)
        return out



# 유클리드 거리 계산 함수
def calculate_euclidean_distance(point1, point2):
    """ 두 점 사이의 유클리드 거리를 계산하는 함수 """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)



"""
웹캠을 열린 상태에서 S키를 누르면 함수 호출.
S키가 눌리면 입력은 3초간 받도록 하고 3초간의 영상에서 0.5초마다 프레임 추출
총 6개의 프레임을 프레임처리 함수에 넣어서 데이터 추가(하나의 행)
6회 반복해서 6개의행, 111개열의 하나의 샘플을 완성
완성된 샘플을 예측모델에 넣고 예측결과 표시
"""


#3초간 영상 수집 S키 눌리면 호출. 0.5초당 한 번 process_frame함수 호출
def process_webcam(cap, hands, selected_keypoints):
    """
    웹캠에서 3초 동안 데이터를 수집하며, 실시간 화면을 계속 갱신.
    """
    print("데이터 수집 시작...")
    sample = []  # 데이터를 저장할 리스트
    start_time = time.time()

    while len(sample) < 6:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # 웹캠에서 프레임 읽기 및 화면 갱신
        ret, frame = cap.read()
        if not ret:
            print("웹캠 프레임을 읽을 수 없습니다.")
            return None

        # 실시간으로 화면 갱신
        cv2.putText(
            frame, f"Collecting... {len(sample)}/6", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA
        )
        cv2.imshow("Webcam", frame)
    
        # 1ms 동안 키 입력 대기
        key = cv2.waitKey(1) & 0xFF
        if elapsed_time >= len(sample) * 0.5:
            # 0.5초마다 프레임 처리
            row_data = process_frame(frame, hands, selected_keypoints)  # 프레임 처리
            sample.append(row_data)
            print(f"프레임 {len(sample)} 처리 완료.")

    if len(sample) == 6:
        sample = np.array(sample, dtype=np.float32)  # NumPy 배열로 변환
        print("수집 완료. 최종 샘플 모양:", sample.shape)
        return sample
    else:
        print("데이터 수집 실패.")
        return None




#프레임 처리(프레임 하나를 처리함 하나의 영상당 6개의 프레임을 처리해야함)
def process_frame(frame,hands,selected_keypoints):
    """
    하나의 프레임에서 손의 키포인트 간 거리와 두 손 중심 간 거리를 계산합니다.
    """
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)#미디어파이프 사용을 위해 RGB변환

    height, width, _ = frame.shape
    left_hand_distances = [-1] * 55
    right_hand_distances = [-1] * 55
    distance_between_hands = -1 #모두 None으로 초기화


    if results.multi_hand_landmarks: #손이 감지된 경우
        hand_centers = []

        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # 선택된 키포인트를 픽셀 좌표로 변환
            landmarks = [
                (hand_landmarks.landmark[i].x * width, hand_landmarks.landmark[i].y * height)
                for i in selected_keypoints
            ]
            
            # 선택된 키포인트 간의 유클리드 거리 계산
            distances = [
                calculate_euclidean_distance(landmarks[i], landmarks[j])
                for i in range(len(landmarks))
                for j in range(i + 1, len(landmarks))
            ]

            # 왼손과 오른손을 구분하여 거리 저장
            if idx == 0:
                left_hand_distances = distances
            elif idx == 1:
                right_hand_distances = distances

            # 손의 중심 좌표 계산
            hand_center_x = sum([lm[0] for lm in landmarks]) / len(landmarks)
            hand_center_y = sum([lm[1] for lm in landmarks]) / len(landmarks)
            hand_centers.append((hand_center_x, hand_center_y))

        # 두 손이 모두 감지된 경우 중심 간 거리 계산
        if len(hand_centers) == 2:
            distance_between_hands = calculate_euclidean_distance(hand_centers[0], hand_centers[1])

    row_data = left_hand_distances + right_hand_distances + [distance_between_hands]
    
    # 결과 반환
    return row_data


#샘플의 라벨 예측
def predict_sample(sample,model):

    sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가
    model.eval()  # 모델 평가 모드
    
    with torch.no_grad():
        prediction = model(sample_tensor)
        predicted_label_index = torch.argmax(prediction, dim=1).item()
        predicted_label = LABELS[predicted_label_index]  # 매핑된 라벨 이름 반환
    return predicted_label

"""
메인함수에서 웹캠을 열고 데이터수집과 프레임처리  진행
"""

def main():
    # MediaPipe Hands 초기화
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    selected_keypoints = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] #한 손 11개 키포인트


    # 하이퍼파라미터 정의
    input_size = 111  # 손 키포인트와 거리 데이터의 특징 수
    hidden_size = 64  # LSTM의 은닉 상태 크기
    num_layers = 2    # LSTM 레이어 수
    num_classes = 22  # 예측할 클래스 수 (수어 라벨 수)

    # 모델 로드
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=num_classes)
    model.load_state_dict(torch.load('state_88model.pth'))


    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠이 열렸습니다. S키를 눌러 데이터 수집을 시작하세요.")

    prediction_result = ""  # 예측 결과를 저장할 변수

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽을 수 없습니다.")
                break

            # 예측 결과를 프레임에 표시
            cv2.putText(
                frame, prediction_result, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.imshow("Webcam", frame)

            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):  # S키 입력 시 데이터 수집 및 예측
                sample = process_webcam(cap,hands,selected_keypoints)
                if sample is not None:
                    predicted_label = predict_sample(sample, model)
                    prediction_result = f"Predicted: {predicted_label}"
            elif key == 27:  # ESC 키 입력 시 종료
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()