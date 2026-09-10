# 고확신 모델을 채택하지 않은 이유

## 선택 안정성

BASE V1 primary nested OOF는 inner에서 후보를 고르고 outer에서 평가했다. US fold마다 word/char text, all-linear fusion, market context, agreement 등 선택이 바뀌었고 선택 안정성은 0.3333이었다. 한 후보의 좋은 전체 점수만으로 모든 구간에 안정적인 구조라고 보기는 어려웠다.

## 방향 반전 실험

feature block에 alpha -1/0/0.5/1/1.5를 inner에서 고르게 했다. 반전·제거·축소가 실제 선택됐지만 outer 성과와 경제적 강건성이 유지되지 않았다. polarity를 찾았다는 사실을 새 신호의 재현으로 보지 않고 통합을 보류했다.

## 더 높은 정확도의 함정

opportunity dual gate는 61.54% 정확도였지만 13개 사건, 1.68% coverage에 그쳤다. 순수익과 bootstrap 하한도 음수였다. primary보다 높은 정확도라는 이유만으로 채택하지 않았다.

## 다음 병목

BASE V1에서는 시장과 publisher가 강하게 얽혔다. nested US 표본은 Reuters, KR 표본은 Naver에 집중됐고 exact/model-candidate multi-source는 23건뿐이었다. 같은 데이터의 미세 조정보다 독립적인 출처와 prospective 가용 시각 근거가 필요하다고 판단했다.

[결과](Validation-and-Results.md) · [데이터 경계](Data-and-Features.md)
