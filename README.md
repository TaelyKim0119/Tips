# 🎯 이슈예측 모델링 프로젝트

**최고의 데이터 분석가의 다양한 머신러닝, 딥러닝, AutoML 모델 비교 분석**

---

## 📋 프로젝트 개요

이 프로젝트는 이슈 데이터의 미래 값(log_M_norm)을 예측하기 위해 다양한 머신러닝 기법을 적용하고 비교하는 포괄적인 분석입니다.

### 🎯 목표
- `label` 컬럼(다음 시간 단계의 log_M_norm)을 최대한 정확하게 예측
- 시계열 데이터에 적합한 최고의 모델 식별
- 다양한 모델 아키텍처의 성능 벤치마킹

---

## 🔄 PoC(Proof of Concept) 진행 과정

본 프로젝트는 다음과 같은 단계로 진행됩니다:

| 파일 | 제목 | 설명 |
|------|------|------|
| **PoC_1** | 데이터 수집 + L지수 + 감정분석 | 원본 데이터 수집, L지수(L_influence) 계산, 감정분석 수행 |
| **PoC_2** | 데이터 병합 | 다양한 소스의 데이터를 통합하여 단일 데이터셋 구성 |
| **PoC_3** | W지수 | W지수(W_score) 계산 및 특성 엔지니어링 |
| **PoC_4** | P지수 | P지수(P_score) 계산 및 추가 특성 생성 |
| **PoC_4-1** | D지수 | D지수(D_score) 계산 및 검증 |
| **PoC_5** | D지수 | D지수 최적화 및 추가 분석 |
| **PoC_6** | 전체 데이터 전처리 | 모든 특성 정규화, 스케일링, 시퀀스 생성 |
| **PoC_7** | 성능평가 LSTM | 22개 모델 비교 및 LSTM 기반 성능 평가 |

---

## 📊 데이터 정보

- **총 샘플**: 32개 (시계열 데이터)
- **특성 수**: 16개 이상
- **주요 특성**: 
  - D_score, L_influence, W_score (정규화된 버전)
  - final_negative_emotion_score
  - Issue 관련 원-핫 인코딩 특성들
  - Cause, Entity, Event, Impact, Reaction 카운트

---

## 🏗️ 모델 아키텍처

### 1️⃣ **전통 머신러닝 (11개 모델)**
```
✓ Linear Regression        - 기본 선형 회귀
✓ Ridge (L2 정규화)         - 과적합 방지
✓ Lasso (L1 정규화)         - 특성 선택
✓ ElasticNet               - L1+L2 결합
✓ Decision Tree            - 트리 기반 모델
✓ Random Forest            - 앙상블 (100 trees)
✓ Extra Trees              - 극도로 랜덤화된 트리
✓ Gradient Boosting        - 순차적 부스팅
✓ AdaBoost                 - 적응형 부스팅
✓ KNN (k=3)                - 근접이웃
✓ SVR (RBF kernel)         - 서포트 벡터 회귀
```

### 2️⃣ **고급 Gradient Boosting (3개 모델)**
```
✓ XGBoost                  - 극한 그래디언트 부스팅
✓ LightGBM                 - 가벼운 그래디언트 부스팅
✓ CatBoost                 - 카테고리 최적화 부스팅
```

### 3️⃣ **딥러닝 (5개 모델)**
```
✓ LSTM                     - 장단기 메모리
✓ GRU                      - 게이트 순환 유닛
✓ 1D CNN                   - 1D 합성곱 신경망
✓ Bidirectional LSTM       - 양방향 LSTM (시계열 최적화)
✓ MLP                      - 다층 퍼셉트론
```

### 4️⃣ **앙상블 모델 (3개 모델)**
```
✓ Voting Regressor         - 다중 모델 평균
✓ Stacking Regressor       - 메타 러너 기반
✓ Optuna-Optimized GB      - 하이퍼파라미터 최적화 모델
```

---

## 🚀 사용 방법

### 설치
```bash
# 모든 의존성 설치
pip install -r requirements.txt
```

### 실행
```bash
# Jupyter Notebook에서 Poc_7.ipynb 실행
jupyter notebook Poc_7.ipynb
```

또는 Cursor IDE에서 직접 셀을 실행합니다.

---

## 📈 모델 평가 지표

각 모델은 다음 지표로 평가됩니다:

| 지표 | 설명 | 범위 | 더 좋은 값 |
|------|------|------|----------|
| **R² Score** | 결정계수 (설명력) | -∞ ~ 1 | 1에 가까울수록 |
| **RMSE** | 제곱 평균 제곱근 오차 | 0 ~ ∞ | 작을수록 |
| **MAE** | 평균 절대 오차 | 0 ~ ∞ | 작을수록 |
| **MSE** | 평균 제곱 오차 | 0 ~ ∞ | 작을수록 |

---

## 🔍 시계열 분할 전략

```python
TimeSeriesSplit(n_splits=3)을 사용하여 
과거 데이터로 훈련하고 미래 데이터로 테스트
```

**분할 구조:**
```
Fold 1: Train=[0-10], Test=[11-15]
Fold 2: Train=[0-15], Test=[16-22]
Fold 3: Train=[0-22], Test=[23-31] ← 사용
```

---

## 🎓 특성 엔지니어링

### 전처리 단계
1. **정규화**: StandardScaler를 사용한 특성 정규화
2. **시퀀스 생성**: 시계열 모델을 위한 윈도우 크기 3
3. **라벨 생성**: `label = next_timestep_log_M_norm`

### 딥러닝 시퀀스
```python
window_size = 3
X_train_seq.shape = (N, 3, 16)  # (샘플, 시간_스텝, 특성)
```

---

## 💡 AutoML (Optuna) 설정

```python
하이퍼파라미터 탐색 공간:
- n_estimators: [50, 200]
- max_depth: [3, 10]
- min_samples_split: [2, 10]
- min_samples_leaf: [1, 4]
- learning_rate: [0.01, 0.2]

최대 시행: 50 trials
```

---

## 📁 생성 파일

실행 후 생성되는 파일들:

```
output/
├── model_comparison_results.xlsx      # 모든 모델 결과
│   ├── All Models            # 전체 모델 순위
│   ├── Traditional ML        # 전통 ML 비교
│   ├── Boosting              # Boosting 모델
│   └── Deep Learning         # 딥러닝 모델
├── model_performance_comparison.png   # 성능 비교 차트 (4가지 시각화)
├── best_model_analysis.png            # 최고 모델 상세 분석 (4가지 플롯)
└── modeling_report.txt                # 최종 보고서
```

---

## 📊 시각화

### 1. 성능 비교 (model_performance_comparison.png)
- **Top 15 R² 순위**: 각 모델의 R² 스코어
- **Top 15 RMSE 순위**: 각 모델의 RMSE
- **R² vs RMSE**: 산점도로 모델 효율성 표시
- **성능 분포**: 박스플롯으로 지표 분포 확인

### 2. 최고 모델 분석 (best_model_analysis.png)
- **실제값 vs 예측값**: 시계열 비교 플롯
- **잔차 플롯**: 예측 오차 패턴 분석
- **오차 분포**: 절대 오차의 히스토그램
- **Q-Q 플롯**: 잔차 정규성 확인

---

## 🔬 딥러닝 모델 아키텍처

### LSTM 모델
```
Input (window=3, features=16)
    ↓
LSTM(64) + Return Sequences
    ↓
Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dense(16, ReLU)
    ↓
Output(1)
```

### Bidirectional LSTM
```
Input
    ↓
BiLSTM(64) + Return Sequences
    ↓
Dropout(0.2)
    ↓
BiLSTM(32)
    ↓
Dense(16, ReLU) + Dense(1)
```

### 1D CNN
```
Input (window=3, features=16)
    ↓
Conv1D(64, kernel=2) + ReLU
    ↓
Conv1D(32, kernel=2) + ReLU
    ↓
GlobalAveragePooling1D()
    ↓
Dense(16, ReLU) + Dense(1)
```

---

## 🎯 최적 모델 선정 기준

1. **R² Score** (최우선): 0.8 이상 권장
2. **RMSE**: 더 낮을수록 좋음
3. **MAE**: 실제 오차 범위 파악
4. **모델 안정성**: 테스트 데이터 성능 일관성

---

## 📝 예상 결과

일반적으로 이 유형의 시계열 데이터에서:

```
🥇 Top Performer: Stacking Regressor / Bidirectional LSTM
  - R²: 0.7 ~ 0.95 범위
  - RMSE: 0.1 ~ 0.2

🥈 Second Tier: Optuna-Optimized GB / Voting Regressor
  - R²: 0.6 ~ 0.85 범위
  - RMSE: 0.15 ~ 0.25

🥉 Third Tier: Gradient Boosting / Random Forest
  - R²: 0.5 ~ 0.75 범위
  - RMSE: 0.2 ~ 0.3
```

---

## 🔧 커스터마이징

### 모델 추가 (Poc_7.ipynb 수정)
```python
# 새로운 모델 추가 예시
models_traditional['New Model'] = NewModel(params)
```

### 하이퍼파라미터 조정
```python
# Optuna 최적화 공간 확장
'new_param': trial.suggest_float('new_param', min_val, max_val)
```

### 특성 선택 변경
```python
X = df_model.drop(['label', 'unwanted_feature'], axis=1)
```

---

## 📚 참고 자료

- **scikit-learn**: https://scikit-learn.org
- **TensorFlow/Keras**: https://www.tensorflow.org
- **XGBoost**: https://xgboost.readthedocs.io
- **LightGBM**: https://lightgbm.readthedocs.io
- **Optuna**: https://optuna.readthedocs.io

---

## 🐛 문제 해결

### 패키지 설치 오류
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt --no-cache-dir
```

### 메모리 부족 (딥러닝)
```python
# 배치 크기 감소
batch_size = 2  # 4 → 2

# 모델 크기 축소
layers.LSTM(32)  # 64 → 32
```

### TensorFlow 경고
```python
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
```

---

## 📞 연락처 및 지원

질문이나 개선 사항은 이슈로 등록해주세요.

---

## 📄 라이선스

MIT License

---

**작성자**: 딥테크팁스 데이터 분석팀  
**작성일**: 2025년 10월 22일  
**최종 업데이트**: 2025년 10월 22일

