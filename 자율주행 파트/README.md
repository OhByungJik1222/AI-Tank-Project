# ✨ 주행 시연 영상

[![Video Label](http://img.youtube.com/vi/bz-GHivsUG8/maxresdefault.jpg)](https://youtu.be/bz-GHivsUG8)

# 🧠 A* 알고리즘 적용

### 📌 1. 휴리스틱 함수 선정

시작 지점, 장애물 위치 고정 후, 각 휴리스틱 함수마다 100회씩 수행 후 평가 (평균)

|Heuristic 함수|시간 (s)|거리 (m)|충돌 횟수|
|---|---|---|---|
|Diagonal|27.57 ✅|260.69 ✅|1 ✅|
|Manhatten|32.27|311.06|63|
|Euclidean|27.98|260.59|15|

### 📌 2. Buffer 추가

최단 경로를 구하는 A* 알고리즘 특성 상, 장애물과 근접하게 경로 생성 → ***충돌 발생***<br>
따라서 장애물 주변에 Buffer를 추가하여 기존 장애물보다 더 크게 인식하도록 만듦 → ***충돌 해결***

![image](https://github.com/user-attachments/assets/2148c16a-5f41-4123-8a76-c79fd7e8311a)

# 🧠 LiDAR 활용

### 📌 1. LiDAR 데이터 클러스터링

LiDAR 센서 특성 상 객체가 있는지만 확인되고 그 객체가 어떤 객체인지 알 수 없음.<br>
따라서 LiDAR 범위내에 인식되는 객체(장애물, 지형 등)들을 각각 하나의 객체로 구분 필요

![image](https://github.com/user-attachments/assets/e73c3373-93fb-47b5-ab78-ac8314a61ca5)

### 📌 2. 장애물, 지형 판별

A* 알고리즘 적용을 위해 구분된 객체에서 실제 장애물만 판별

![image](https://github.com/user-attachments/assets/9fa829f8-03cd-4540-ab43-730f315d1946)


# 🧠 알고리즘 작동 방식

![주행 pptmp4](https://github.com/user-attachments/assets/d5f985e6-dbe2-45a4-bef2-3da574696c39)
