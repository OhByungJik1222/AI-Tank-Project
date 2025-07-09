# YOLO–LiDAR 융합 전차 자동 조준 및 주행 시스템
~~여기 논문이나 피피티 파일 첨부~~

## ✨ 최종 시연 영상

[![Video Label](http://img.youtube.com/vi/p360diqBGcQ/maxresdefault.jpg)](https://youtu.be/p360diqBGcQ)

## 📂 루트 디렉토리 및 주요 파일 구조
```
📁 루트 폴더
├── main/  # 메인 애플리케이션 코드
├── maps/  # 시연 영상용 맵
├── 자율주행 파트/ # 주행 기능 코드
└── 포탑제어 파트/ # 사격 기능 코드

📁 main 폴더
├── models/ # 객체 인식/사격 각도 예측 모델
├── static/ # UI용 이미지
├── templates/ # UI용 HTML 파일
├── app.py # 메인 실행 코드
├── drive.py  # 주행 기능 함수 코드
└── fire.py # 사격 기능 함수 코드
```

## 🛠️ 시스템 구현 환경
 - 프로그래밍 언어 : Python 3.10
 - 서버 구조 : Flask REST API
 - 탐지 모델 : YOLOv8
 - 위치 센서 : LiDAR + RGB 카메라 (시뮬레이터 제공)

## 📖 알고리즘 및 모델
|기능|알고리즘 및 모델|
|---|---|
|자율 주행|A* 알고리즘|
|객체 인식|YOLOv8|
|포탑 제어|XGBoost + DNN|

## 🧱 아키텍쳐

![KakaoTalk_20250703_164305623](https://github.com/user-attachments/assets/2a783122-6182-4be7-bbaa-210d96c175b5)


## 📌 실행 전 유의사항
정확한 사격을 위해 시뮬레이터 해상도 조정 (1920 x 1080)

![image](https://github.com/user-attachments/assets/dfa199bf-9df6-4a8c-b1f5-7ebca06f9661)

속성 → 대상 경로 뒤 아래 해상도 값 입력
```
-screen-width 1920 -screen-height 1080
```


