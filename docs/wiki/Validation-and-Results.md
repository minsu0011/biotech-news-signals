# 결과를 단계별로 구분하기

## 초기 correctness 진단

V224의 immutable DEV 기반 outer-OOF 전체 1,674행에서 correctness AUC는 0.508730, top20 정확도는 0.540299로 기록됐다. 원 publisher와 ingest 시각 증명 부족 때문에 production discriminator로 채택하지 않았다.

## historical BASE V1

source·split freeze와 model-ready gate를 통과한 뒤 nested outer-OOF 772행을 평가했다. primary 결과는 다음과 같다.

| 지표 | 값 |
| --- | ---: |
| correctness ROC-AUC | 0.458725 |
| Brier | 0.293390 |
| 고확신 coverage | 20.73% |
| 고확신 정확도 | 47.50% |
| HC net | -0.002123 |
| HC net bootstrap lower95 | -0.002970 |
| 상위 5개 사건 제거 HC net | -0.002729 |

15개 실질 후보와 10개 polarity 개입에서 강건한 survivor가 없었다. 공식 direction probability와 예측 방향은 변경하지 않았고 Research Seal·Final Meta를 새 후보 선택에 쓰지 않았다.

두 실험은 표본·데이터·절차가 다르다. 평가 조건이 달라 직접적인 개선치로 비교하지 않았다. 기사 수·embedding coverage가 늘어난 것도 위 성과의 개선을 증명하지 않는다.

## 데이터 성과와 모델 성과

V221 parity 성공, historical 품질 계약 통일, source freeze는 데이터 공학의 결과다. correctness의 강건성 실패와 동시에 성립한다. [테스트 범위](../testing-notes.md)는 실행 확인과 연구 기록의 차이를 따로 남긴다.
