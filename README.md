# 🎯 SK 유심정보 유출 이슈 - 리스크 조기 정보 예측 시스템

**LSTM 기반 소셜 미디어 감정 분석 및 시계열 예측 모델**

---

## 📋 프로젝트 개요

### 📌 데이터셋
- **대상**: SK 유심정보 유중 보안 이슈
- **플랫폼**: YouTube (3월 ~ 9월)
  - 영상 100건
  - 댓글 5,469건

### 🎯 목표
SK 유심정보 유중 관련 부정 여론을 극복하고 이슈의 **강도·결합·파급·주도자**를 추적하여 **단기(6시간 전) 예측**으로 리스크 조기 정보 발생

### 📊 모델 평가 기준
- MSE, RMSE, MAE, MAPE, R²

---

## 🔄 PoC(Proof of Concept) 진행 과정

| 단계 | 파일 | 설명 |
|------|------|------|
| 1️⃣ | **PoC_1** | 데이터 수집 + L지수 + 감정분석 |
| 2️⃣ | **PoC_2** | 데이터 병합 |
| 3️⃣ | **PoC_3** | W지수 계산 |
| 4️⃣ | **PoC_4** | P지수 계산 |
| 4️⃣-1 | **PoC_4-1** | D지수 계산 |
| 5️⃣ | **PoC_5** | D지수 최적화 |
| 6️⃣ | **PoC_6** | 전체 데이터 전처리 |
| 7️⃣ | **PoC_7** | 성능평가 (LSTM 기반) |

---

## 📈 주요 특성 (Features)

| 지표 | 설명 | 역할 |
|------|------|------|
| **L_influence** | L지수 | 이슈의 영향력 |
| **W_score** | W지수 | 관심도 변화 추적 |
| **P_score** | P지수 | 파급력 측정 |
| **D_score** | D지수 | 거리도 및 결합도 |
| **final_negative_emotion_score** | 감정분석 | 부정 감정도 |

---

## 🚀 사용 방법

### 설치
```bash
pip install -r requirements.txt
```

### 실행
```bash
# Jupyter Notebook에서 PoC 파일 순서대로 실행
jupyter notebook Poc_1.ipynb  # 데이터 수집
jupyter notebook Poc_2.ipynb  # 데이터 병합
jupyter notebook Poc_3.ipynb  # W지수
jupyter notebook Poc_4.ipynb  # P지수
jupyter notebook Poc_5.ipynb  # D지수
jupyter notebook Poc_6.ipynb  # 전처리
jupyter notebook Poc_7.ipynb  # LSTM 성능평가
```

---

## 📊 LSTM 모델 아키텍처

```
Input (시간스텝=3, 특성=16)
    ↓
LSTM(64) + Dropout(0.2)
    ↓
LSTM(32)
    ↓
Dense(16, ReLU)
    ↓
Output(1)
```

---

## 📁 출력 파일

```
output/
├── model_performance_comparison.png    # 성능 비교 시각화
├── best_model_analysis.png             # 최고 모델 분석
├── model_comparison_results.xlsx       # 상세 결과
└── modeling_report.txt                 # 최종 보고서
```

---

## 🎓 모델 성능 평가 지표

| 지표 | 설명 | 범위 |
|------|------|------|
| **R²** | 결정계수 | -∞ ~ 1 |
| **RMSE** | 제곱 평균 제곱근 오차 | 0 ~ ∞ |
| **MAE** | 평균 절대 오차 | 0 ~ ∞ |
| **MAPE** | 평균 절대 백분율 오차 | 0% ~ ∞% |
| **MSE** | 평균 제곱 오차 | 0 ~ ∞ |

---

## 📅 PoC 수행 기간

- **시작**: 2025년 10월 15일
- **종료**: 2025년 10월 23일

---

## 📞 문의

이슈나 개선 사항은 GitHub Issues를 통해 등록해주세요.

---

**작성자**: 딥테크팁스 데이터 분석팀  
**최종 업데이트**: 2025년 10월 27일

