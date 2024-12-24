# sign-language-kiosk
> LSTM, Mediapipe를 이용한 수어번역 모델<br><br>
> ![불고기](https://github.com/user-attachments/assets/4a76571a-91cd-433b-a8ab-17fe4578523c)


<br><br>

## 프로젝트 소개
> 햄버거(키오스크) 주문 상황을 가정하여 22개 단어를 선정하고,<br>
> 22개 단어에 대한 수어 영상을 직접 촬영하여 데이터셋을 제작했습니다.<br>
> 직접 구축한 데이터셋을 LSTM모델에 학습 시키고 웹캠을 통해 시연했습니다.<br>
> 해당 머신러닝 프로젝트는 <취약계층을 위한 키오스크 개선> 팀 프로젝트의 일부분 입니다.

<br><br>

## 흐름도
![수어인식흐름도](https://github.com/user-attachments/assets/c0196e27-c819-42bb-a2f8-f4f92a723b47)

### 학습 데이터 가공 과정
> 1. mediapipe hand landmark detection 모델을 이용해서 검출 가능한 21개 keypoint 중에서<br>
11개를 선택(0, 2, 4, 6, 8, 10, 12, 14, 16, 18 , 20번 keypoint)
> 2. 중복 없이 두 점을 선택하는 조합의 방법으로 거리 데이터 생성
> 3. 왼손 (55개) + 오른손 (55개) + 양손 간 거리 (1개) 총 111개의 열
> 4. 촬영한 동영상이 모두 약 3초 -> 0.5초마다 데이터를 추출하여 총 6개의 행
> 5. 6개의 행과 111개의 열을 하나의 샘플(동영상)으로 가지는 csv파일 생성
> 6. 생성된 csv파일을 LSTM모델에 학습

<br><br>

## 데이터셋
![image](https://github.com/user-attachments/assets/6ae0ce8c-2d24-43a5-9d99-25a59013f043)

> - 22개 클래스, 각 클래스당 약 60개 데이터 수집
> - 약 1300개의 동영상을 팀원 4인이 직접 촬영
> - 다양한 각도와 거리에서 촬영

<br><br>

## 학습결과
![학습결과](https://github.com/user-attachments/assets/8535ec0b-0392-458d-9802-1133f0164b55)

<br><br>

## 팀원소개

> 상명대학교 소프트웨어학과(소프트웨어프로젝트PBL 취약계층을 위한 키오스크 개선 팀)<br>
> 팀장 : 임지안<br>
> 팀원 : 김다애, 유다연, 정수민<br>








