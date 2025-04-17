# 🧥 데이터 크리에이터 캠프 2024 예선 - 패션 이미지 선호 여부 예측
데이터 크리에이터 캠프_2024 예선 | 패션 이미지 선호 여부 예측

본 프로젝트는 패션 이미지에 대한 사용자의 선호 여부를 예측하는 것을 목표로 하며, AI가 사람의 패션 취향을 학습하고 이해할 수 있는지 탐색합니다.  
이미지 분류, 통계 기반 분석, 협업 필터링 기반 추천 시스템 등을 통해 문제 해결을 시도했습니다.

---

## 📌 프로젝트 개요

- **주제:** 패션 이미지 선호 여부 예측
- **팀명:** Taverse
- **기간:** 2024년 데이터 크리에이터 캠프 예선
- **목표:** AI는 사람의 마음(패션 취향)을 이해할 수 있을까?

---

## 🗂 폴더 구조 및 데이터셋

- `training_image/`, `validation_image/`  
- `training_label/`, `validation_label/`  
- `no_bg_training_image/`, `no_bg_validation_image/` : 배경 제거된 이미지  
- `final/` : 미션별 코드  
- `process/` : 실험 기록용 폴더

**총 데이터 개수**  
- Train 이미지: 4070장  
- Validation 이미지: 951장  
- 라벨 파일(.json): 약 25만개

---

## 🧩 Mission 구성

### 1️⃣ 이미지 분류 (Mission 1)

#### 🔍 문제
이미지로부터 성별과 스타일을 분류

#### 🛠 진행 방법
- 파일명 기반 `성별-스타일` 정보 추출
- 배경 제거 (rembg 사용)
- ResNet-18 직접 구현 (pretrained X)
- 다양한 전처리/증강 조합 테스트

#### 🧪 최종 모델
- Custom ResNet-18
- Accuracy (Train): 99.66%
- Accuracy (Val): 63.30%

---

### 2️⃣ 선호 여부 예측 - 통계 기반 (Mission 2)

#### 🔍 문제
성별, 스타일 선호 통계를 통해 패션 취향 예측

#### 🛠 진행 방법
- 파일명에서 성별 & 스타일 정보 추출
- JSON 라벨링 파일 처리하여 유효 응답자 추출
- 100명의 응답자 선호 정보표 생성
- 응답자별 스타일 선호도 매핑

---

### 3️⃣ 선호 여부 예측 - 추천 시스템 기반 (Mission 3)

#### 🔍 문제
이미지 유사도, 선호 패턴 유사성을 기반으로 스타일 추천

#### 🛠 진행 방법

- **User-based Collaborative Filtering**
  - 응답자 간 선호 벡터 유사도 계산 (Cosine)
  - 유사한 응답자의 선호 스타일 추천

- **Item-based Collaborative Filtering**
  - 스타일 간 유사도 분석
  - 기존에 선호한 스타일과 유사한 스타일 추천

- **Feature-based Similarity**
  - ResNet-18 중간 feature vector 추출
  - 이미지 간 유사도 기반 선호 여부 판단 (임계값 0.8)

---

## 📈 결과 요약

| 항목              | 값                     |
|------------------|------------------------|
| Train Accuracy   | 99.66%                 |
| Validation Accuracy | 63.30%              |
| 추천 시스템 정확도 | 선호/비선호 유사도 기반 판단 |

---

## 🧠 주요 기술 스택

- Python, PyTorch, Pandas, Numpy
- OpenCV, rembg
- ResNet-18
- Cosine Similarity, Collaborative Filtering

---

## 📎 참고 링크

- [ResNet 구조 참고 구현](https://github.com/shoji9x9/Fashion-MNIST-By-ResNet/)
- [rembg 라이브러리](https://github.com/danielgatis/rembg)

