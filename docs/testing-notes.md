# 테스트와 실행 범위

저장소 루트에서 의존성을 준비한 뒤 다음 범위를 확인할 수 있다.

```powershell
python bio_news_30m_v3.py --help
```

도움말 확인과 과거 뉴스·신뢰도 실험은 다른 범위다. README/Wiki의 historical BASE V1 수치는 원본 FINAL_RESEARCH_REPORT와 비교한 기록 요약이며 새 학습의 결과가 아니다.

뉴스 공개·수집 시각의 차이, 분봉 결손과 symbol alias, 공급자 권한이 남아 있다. 고확신 correctness 모델은 비용과 강건성 기준을 넘지 못했다. 역사 뉴스 실험을 새 시장의 일반화 성과로 확대하지 않는다.

테스트 실행과 전체 원천 수집·학습은 별개다. 연구 결과는 [README](../README.md)의 당시 기록과 입력 조건을 함께 읽는다.
