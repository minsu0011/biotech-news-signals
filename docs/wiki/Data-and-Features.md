# timestamp·출처·품질의 계약

V224의 기존 DEV에는 정확한 event time과 t+2 entry가 있었지만 별도의 fetched/ingested timestamp는 없었다. 따라서 URL·본문이 t+2에 실제로 가용했다는 독립적인 증명에는 한계가 있었다.

Naver·Google 같은 aggregator host를 원 publisher라고 간주할 수도 없다. US article ID coverage도 희소해 URL hash는 duplicate purge key로만 사용했다. raw URL·기사 ID·label·수익률 자체는 모델 입력이 아니다.

historical BASE V1에서는 `classify_historical_article_quality` 계약을 공통화했다. unknown publisher 정규화, HIGH event match, 정확한 publication time, body 길이 조건을 맞췄다. 지금 crawl한 시각으로 historical first-seen을 채우지 않고 ambiguous mapping을 격리한다.

당시 source freeze에는 article 129,313개와 canonical event 235,566개가 있었지만 strict direction-OOF model-ready는 1,094개, nested 평가 표본은 772개였다. 큰 수집량이 그대로 학습·평가 표본이 되는 것은 아니다.

과거 reliability feature는 fold의 train 이력에서만 계산한다. correctness-history에 필요한 label 성숙 조건, ticker embargo와 duplicate purge는 feature 종류와 독립적인 경계다.
