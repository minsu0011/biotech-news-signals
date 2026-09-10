# 이벤트와 신뢰도를 연결하는 구조

뉴스·SEC 이벤트와 시장 분봉을 먼저 연결한다. ticker/시장/시각을 기준으로 사건을 정하고, 공개 시각 이후의 진입·반응창에서 방향 label을 만든다. 방향 모델과 correctness 모델은 목표부터 다르다.

direction output은 기존 OOF 예측이다. P(correct)의 목표는 그 예측이 실제로 맞았는지다. 동일 행을 본 방향 모델의 과신을 다시 학습하지 않도록 outer-OOF·purge 경계를 둔다. confidence gate는 점수뿐 아니라 coverage·비용 반영 결과를 본다.

수집 복구와 source parity는 이 앞단의 데이터 공학이다. provider가 바뀌면 event timing, entry price, label 방향이 달라질 수 있으므로 교차 확인 없이 모델 점수를 이어 비교하지 않는다.

[기본 엔진](../../bio_news_30m_v3.py)과 [연구 모듈](../../research)을 따라 데이터·방향·correctness·정책의 네 경계를 구분해서 읽으면 된다.
