# 실행 경로와 입력

```powershell
python bio_news_30m_v3.py --help
```

실제 입력은 provider 계정·권한, 이벤트 inventory, 가격창과 동결된 split이다. API key는 환경·로컬 secret 관리 방식으로 설정하고 원문 기사·응답을 저장소에 넣지 않는다. [데이터 안내](../../data/README.md)를 먼저 확인한다.

[research](../../research)의 수집 복구와 high-confidence runner는 서로 다른 입력과 output namespace를 사용한다. 원천 기사 수집이 끝나도 direction-OOF와 label 성숙 조건이 없으면 correctness 학습을 시작할 수 없다.

예전 원천·receipt·수집 정책 파일을 요구하는 연구 runner도 있다. 경로만 바꿔 실행된 결과를 기존 freeze와 같은 실험으로 보지 않는다. 기존 frozen output을 덮어쓰지 말고 새 연구 identity와 입력 근거를 먼저 정한다.
