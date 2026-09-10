# 200여 실험을 계열로 읽기

| 실험 계열 | 대표 질문 | 다음 단계에 남긴 판단 |
| --- | --- | --- |
| 초기 feature·direction | 뉴스 뒤 반응을 어떤 입력으로 설명할까 | 기본 방향·반응 엔진 |
| confidence·direction | 자신 있는 예측이 실제로 맞는가 | 별도 correctness 목표 |
| source recovery | 원문·분봉 결손을 어디서 복구할까 | 결손·권한 상태와 재시도 분리 |
| provider parity | 다른 공급자의 시세로 같은 label이 나오는가 | 고정 event의 시각·가격 비교 |
| causal/PIT | 진입 전에 알 수 있었던 내용인가 | 공개·수집 시각, purge, embargo |
| embedding·text | 내용·표현·novelty가 정답 가능성을 설명하는가 | metadata와 별도 후보·fusion |
| robustness/generalization | 소수 사건·시장·publisher 효과인가 | bootstrap·top 사건 제거·fold 안정성 |
| V221 data epoch | 새 표본이 역할별 요구 수를 채웠는가 | coverage 미달이면 학습·seal 보류 |
| V224 역사/봉인 계열 | 기존 DEV와 별도 평가의 경계를 지킬 수 있는가 | prospective·historical 결과 분리 |
| high-confidence BASE V1 | 넓힌 기사 기반에서 robust P(correct)가 나오는가 | 생존 후보 없음, 공식 통합 보류 |

각 계열은 독립적인 연구 질문이고 모든 번호가 새로운 architecture는 아니다. text·embedding 모델은 표현을 제공하고 reliability는 과거 성공 이력을 fold-safe하게 요약한다. polarity 개입은 기존 feature block의 방향·크기를 바꿔 보는 진단이다.

[실험·연구 코드](../../research)와 루트의 experiment entrypoint를 이 지도에 맞춰 읽는다. 최신 번호를 단일 champion으로 선정하지 않는다.
