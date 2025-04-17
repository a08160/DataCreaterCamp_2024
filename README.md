
# 👗 데이터크리에이터 캠프 2024: AI를 활용한 패션 스타일 분류 및 선호 예측

> **"인공지능은 사람의 마음을 이해할 수 있을까?"**

**목적** - 딥러닝 기반의 이미지 분류와 협업 필터링 기반 추천 시스템을 결합하여, 사용자의 패션 취향을 학습하고 예측

---

## 📌 프로젝트 개요

### 🎯 목표
1. **ResNet-18 기반의 패션 스타일 이미지 분류**
2. **응답자 선호 데이터를 이용한 스타일 선호 예측**
3. **이미지 유사도를 통한 협업 필터링 추천 시스템 구현**

- <strong>팀명: Taverse</strong>
- <strong>기간: 2024.09.23~10.31</strong>
---

## 📁 데이터 구조

```
/final
-- README.md
|-- Taverse.pdf
|-- code
|   |-- mission 1.ipynb
|   |-- mission 1.pdf
|   |-- mission 2.ipynb
|   |-- mission 2.pdf
|   |-- mission 3-2.ipynb
|   `-- mission 3-2.pdf
|-- data
|   |-- 3-2 prediction_results.csv
|   |-- survey_style_preferences.csv
```

- 총 이미지 수: `training: 4070`, `validation: 951`
- 라벨 수: `training: 211,345`, `validation: 36,382`
- 일부 JSON 오류 파일 제외 처리

---

## 🔧 사용 기술 & 라이브러리

- **Framework**: PyTorch
- **Model**: ResNet-18 (custom 구현)
- **Image Processing**: rembg, torchvision.transforms
- **추천 시스템**: Collaborative Filtering (User & Item 기반)
- **기타**: pandas, numpy, tqdm, matplotlib

---

## 🧪 실험 (Trial) 요약

### ✅ Best Model – Trial 13
- 모델: `model3` (ResidualBlock + ReLU 추가)
- Optimizer: Adam (lr=0.001), weight_decay=1e-4
- 성능:
  - Train Accuracy: **99.66%**
  - Validation Accuracy: **63.30%**
  - 과적합 완화 + 성능 안정화

### ⚙️ 주요 실험 요약

| Trial | 모델 구조 | 전처리 | 특이사항 | 성능 요약 |
|-------|------------|--------|------------|-------------|
| 1     | model1 (pretrained=False) | Flip, Rotation | Object Detection X | Acc < 10% |
| 2     | model1 | rembg 적용 | 성능 미미 | - |
| 5~6   | model1 | Flip + ColorJitter | StepLR 적용 | Val Acc ~12% |
| 9~10  | model2 (직접 구현) | 다양한 증강 | Dropout, 규제 | 성능 낮음 |
| **13**| model3 (ReLU 추가) | 최적 증강 조합 | **BEST** | **Val Acc 63.30%** |
| 17    | model3 + 2nd ReLU | 학습률 감소 | Val Acc 60.04% | 안정화 추이 |

---

## 🧠 모델 구조 (model3)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        ...
        self.relu = nn.ReLU()

    def forward(self, x):
        ...
        return self.relu(out + shortcut)

class ResNet18(nn.Module):
    def __init__(self, num_classes=30):
        ...
        self.layer1 = self.make_layer(...)
        ...
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
```

---

## 🖼️ 전처리 및 증강

```python
transform_train = transforms.Compose([
    transforms.Resize((200, 266)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(...),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((200, 266)),
    transforms.ToTensor(),
    transforms.Normalize(...)
])
```

- `rembg`를 통해 배경 제거 → 흰색 배경 대체
- validation에는 증강 미적용

---

## 🧑‍🤝‍🧑 협업 필터링 시스템

### ✔ User-based Filtering
- 응답자 간 스타일 선호 유사도 계산 (코사인 유사도)
- 가장 유사한 응답자 기반으로 스타일 추천

### ✔ Item-based Filtering
- 스타일 벡터 간 유사도 비교
- 선호한 스타일과 유사한 스타일 추천

### ✔ 이미지 유사도 기반 예측
- ResNet-18 중간 레이어 feature vector 추출
- 유사도 ≥ 0.8 → 선호로 간주
- 선호 유사도 평균 vs 비선호 평균 비교로 최종 판단

---

## 📊 성능 그래프

```
Train Accuracy: 99.66%
Validation Accuracy: 63.30%
Train Loss: 0.02
Validation Loss: 2.04
```

---

## 🔮 개선 방향

- Checkpoint 저장(`torch.save`)으로 장시간 학습 가능성 확보
- Vision Transformer 등 차세대 모델 도입 가능성 모색
- Active Learning 기반 사용자 피드백 추천 시스템 개선

---

## 📜 라이선스

MIT License
