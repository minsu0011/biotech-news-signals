# Biotech News Signals

바이오 뉴스가 나온 뒤 30분 가격 반응을 설명하고, 방향 예측 중 어떤 사건을 신뢰할 수 있는지 연구한다. 기사 수를 늘리는 일과 좋은 예측 신호를 만드는 일을 따로 다뤘다. 뉴스 시각·원문 출처·가격창이 맞지 않으면 신뢰도 모델도 잘못된 근거를 학습할 수 있기 때문이다.

## 전체 구조와 기술

뉴스·공시 수집 → event/ticker 연결 → 공개 시각·진입 시각 정렬 → 분봉 label → 방향 모델 → 별도 correctness 모델 → 신뢰도·coverage·순수익 평가

- [bio_news_30m_v3.py](bio_news_30m_v3.py): 이벤트와 모델의 기본 흐름
- [research](research): 수집 복구, provider parity, 역사 뉴스와 신뢰도 연구
- 루트의 `experiment_v*.py`: 개별 가설을 시험한 실행 단위

Python, pandas·NumPy, scikit-learn, LightGBM, XGBoost, CatBoost, PyTorch를 사용한다. 텍스트·embedding 모델은 뉴스의 표현을 만들고, 별도 confidence 모델은 기존 방향 예측이 맞을 가능성을 추정한다.

## 데이터 설계

한국·미국 뉴스와 공시를 이벤트에 연결하고, provider의 분봉으로 반응창을 만든다. 공개 시각과 실제 수집 시각은 다른 필드다. 과거 기사를 지금 수집했다고 당시 진입 시점에 원문을 확보할 수 있었다는 증거가 생기지는 않는다.

중복 URL·기사 ID는 purge와 과거 지원 건수를 계산하는 식별자다. raw ID나 미래 수익률을 신뢰도 입력으로 넣지 않는다. source recovery, provider parity, model-ready 표본 수를 각각 관리한다.

## 개발 과정

1. **이벤트·가격 반응의 기본 모델을 만들었다.** 방향과 시장 반응을 예측하면서 feature family와 시간창을 비교했다.
2. **데이터 결손을 모델 문제와 분리했다.** no-bar, symbol alias, 공급자 권한 문제를 수집 복구 계열로 옮겼다. 다운로드 성공을 예측 성능 향상이라고 부르지 않았다.
3. **provider를 교체하기 전에 같은 이벤트로 맞췄다.** V221의 Alpaca parity에서는 가격·시각·방향 일치를 확인했다. 그러나 부족한 역사 이벤트를 다 채우지 못해 새로운 data epoch와 모델 변경을 진행하지 않았다.
4. **방향의 확신과 실제 정답 가능성을 구분했다.** V224는 기존 방향 출력의 outer-OOF 정답 여부를 목표로 별도의 P(correct)를 만들었다. 지나치게 자신 있는 오답을 걸러낼 수 있는지가 질문이었다.
5. **기사 품질 규칙을 공통화했다.** historical BASE V1에서 worker와 preview의 body 길이·event match·publisher 처리 방식이 달랐다. 공통 계약으로 정리한 뒤 source와 split을 고정하고 label을 결합했다.
6. **텍스트와 신뢰도 후보를 넓혔다.** metadata, word/char text, embedding, reliability, novelty, agreement, market context를 비교하고 feature 방향을 반전·제거·축소하는 실험도 했다.
7. **높은 점수만 남기는 방식을 채택하지 않았다.** 후보 선택이 fold마다 흔들렸고 비용과 강건성 기준을 넘지 못했다. 공식 방향 출력은 유지하고 독립적인 다중 출처와 prospective 데이터 확보를 다음 병목으로 남겼다.

## 모델의 역할

방향 모델은 상승·하락을 예측한다. P(correct)는 그 방향을 다시 예측하는 대신 **기존 OOF 예측이 맞았는지**를 배운다. confidence gate는 이 점수로 일부 사건을 선택하되, 정확도만 아니라 coverage와 비용 반영 결과를 함께 요구한다.

publisher/topic reliability는 과거 label만 사용한 신뢰도 추정이며, embedding·text 모델은 내용의 차이를 표현한다. 외부 라이브러리의 모델 구현과 직접 만든 이벤트 계약·평가·provider 연결은 [출처](ATTRIBUTION.md)에서 구분했다.

## 결과: 신뢰도 모델은 왜 보류했나

당시 historical high-confidence BASE V1 기록에서 model-ready 1,094건 중 nested outer-OOF 평가는 772건이었다. 15개 실질 후보와 10개 feature-block 개입을 평가했지만 강건한 생존 후보는 없었다.

- primary correctness ROC-AUC: **0.458725**
- 고확신 coverage: **20.73%**, 정확도: **47.50%**
- 비용 반영 HC net: **-0.002123**, bootstrap lower95: **-0.002970**

별도 opportunity dual gate는 정확도 61.54%였지만 단 13건, coverage 1.68%였고 순수익도 음수였다. 이 숫자만으로 성공했다고 판단하지 않았다.

V224 초기 진단과 historical BASE V1은 평가 조건이 달라 직접적인 개선치로 비교하지 않았다. 시장과 publisher가 얽힌 표본 구성, 독립 다중 출처 부족, 과거 ingest 시각 증명 부족이 남아 있다. 200개가 넘는 실험은 [계열별 문서](docs/wiki/Model-Evolution.md)로 묶었다.

## 시작하기

```powershell
python bio_news_30m_v3.py --help
```

실제 수집·평가에는 provider 권한과 [데이터 준비](data/README.md)가 필요하다. 기사 원문과 공급자 응답은 포함하지 않는다.

## 상세 문서

[연구 안내](docs/wiki/Home.md) · [개발 과정](docs/wiki/Development-Journey.md) · [실험 계열](docs/wiki/Model-Evolution.md) · [수집·시각 병목](docs/wiki/Data-and-Features.md) · [신뢰도 실험 결정](docs/wiki/Experiments-and-Decisions.md) · [결과](docs/wiki/Validation-and-Results.md)
