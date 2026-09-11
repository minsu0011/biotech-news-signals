# 수집 개선과 모델 개선을 나눠 온 과정

V3 이후의 실험은 하나의 긴 모델 번호 목록보다 세 갈래로 읽는 편이 낫다. 첫째는 방향·반응 feature와 모델, 둘째는 원천 복구·provider parity, 셋째는 인과적인 데이터 경계와 confidence다.

V221에서는 Alpaca의 실제 접근과 고정된 DEV sample의 parity를 확인했다. 63개 exact pair의 시각·UP/DOWN 방향은 일치했지만, US_SEC 필요 표본은 78/120으로 42개가 부족했고 Final Meta도 5개가 부족했다. 접근 성공이나 높은 가격 일치만으로 새 epoch를 승인하지 않았다.

V224 계열은 기존 방향을 바꾸지 않고 correct 여부를 추정했다. 방향 점수의 크기만으로 “이 예측은 믿을 만하다”고 정하는 대신, 실제 OOF 정답 여부를 별도 목표로 삼았다. 과거 DEV만 사용한 초기 진단에는 publisher identity·ingest 시각의 제약이 있었다. 이를 해소하려고 historical 뉴스 기반을 넓혔다.

BASE V1에서 worker와 preview의 품질 판정이 달라 같은 기사가 다른 Tier에 들어갔다. body 길이, event match, publisher와 unknown 처리 규칙을 통일한 뒤 outcome-blind source와 split을 동결했다. 이후 DEV label을 결합해 선택 과정에서 결과를 미리 보지 않도록 했다.

model-ready 표본이 생긴 뒤 metadata/text/embedding/reliability/novelty와 방향 반전 실험을 수행했다. 하지만 primary 모델의 선택이 fold마다 흔들리고 고확신 순수익의 하한이 음수였다. 기존 direction과 prospective scorer를 고치지 않고 독립 다중 출처를 추가하는 단계로 남겼다.

높은 확신을 부여한 집단도 BASE V1에서는 정확도 47.50%, 비용 반영 순수익 -0.002123이었다. 이 실패 때문에 confidence 자체의 신뢰성을 별도 문제로 남겼다. [strict-OOF discriminator](../../experiment_v225_high_confidence_news_discriminator.py)는 고정된 방향 예측의 correctness를 추정하며, 시간순 cross-fit과 35분 embargo, 과거 학습 label만으로 계산한 신뢰도 feature를 사용한다. 코드를 구현했다는 사실만으로 독립적인 미래 성능을 입증한 것은 아니다.

또한 [historical seal guard](../../v224_historical_seal_guard.py)는 confidence 모델과 다른 역할이다. 사전 고정한 사건 목록과 예측 해시를 확인하고 순서대로 한 번만 정답에 접근하도록 제한한다. 이 경계를 후보 선택에 이용하거나, 실패한 confidence 모델을 구하기 위해 평가를 반복 개방하지 않는다.
