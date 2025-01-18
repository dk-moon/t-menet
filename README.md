# t-분포에서의 MeNet (Mixed Effects Neural Network)

## 📋 프로젝트 개요

`t-분포에서의 MeNet` 프로젝트는 **혼합 효과 신경망(Mixed Effects Neural Network)**을 활용하여 t-분포 데이터를 분석하고, 고정 효과(Fixed Effects)와 랜덤 효과(Random Effects)를 기반으로 한 예측 모델을 구현하는 것을 목표로 합니다.

---

## 📂 데이터 설명

### 데이터 파일: `t_distribution_data.csv`

| Column Name | Description               |
|-------------|---------------------------|
| `id`        | 클러스터 또는 그룹 ID      |
| `x`         | 입력 변수 (독립 변수)      |
| `y`         | 출력 변수 (종속 변수)      |

- 데이터는 t-분포를 따르는 값으로 구성되어 있으며, `id`를 기반으로 클러스터가 나뉘어져 있습니다.

---

## 🧪 모델 설명

### MeNet (Mixed Effects Neural Network)
MeNet은 고정 효과와 랜덤 효과를 결합하여 클러스터 간 차이를 효과적으로 모델링할 수 있는 신경망입니다.

#### 주요 구성 요소:
1. **고정 효과(Fixed Effects)**:
   - 입력 데이터(`x`)를 기반으로 신경망을 통해 추정됩니다.
   - 신경망 아키텍처:
     - **FC1**: 입력층 → 첫 번째 은닉층
     - **FC2**: 첫 번째 은닉층 → 두 번째 은닉층
     - **FC3**: 두 번째 은닉층 → 출력층

2. **랜덤 효과(Random Effects)**:
   - 클러스터별 특성을 반영하기 위해 고정 효과에 추가적으로 적용됩니다.
   - `b_hat`을 통해 추정되며, 랜덤 효과의 공분산(`D_hat`) 및 잔차 분산(`sig2e_est`)이 학습 과정에서 계산됩니다.

---

## 🚀 모델 학습

### 학습 과정
1. **E-Step**:
   - 랜덤 효과(`b_hat`)를 클러스터별로 업데이트.
   - 클러스터 특성 맵(`Z_i`)과 잔차를 기반으로 랜덤 효과를 계산.

2. **M-Step**:
   - 고정 효과를 업데이트하기 위해 신경망 가중치를 최적화.
   - 손실 함수로 MSE(Mean Squared Error)를 사용.

#### 학습 코드:
```python
model, b_hat, sig2e_est = train_menet(model, train_loader, n_clusters, device, epochs=100)