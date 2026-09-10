"""Reproduce the production V20 DEV policy without creating frozen artifacts."""
from __future__ import annotations

import json

import pandas as pd

import bio_news_30m_v3 as app


def main() -> None:
    cfg = app.Config()
    frame = app.load_search_frame(cfg)
    dev, _, _ = app.split_dev_seal(frame, cfg)
    pieces = []
    for market in ("US", "KR"):
        spec = app.model_specs(market)[0]
        part = app.cv_market_predictions(dev, market, spec, cfg)
        pieces.append(part)
    oof = pd.concat(pieces, ignore_index=True)
    metrics = app.evaluate(oof, cfg.fixed_highconf_threshold, cfg.round_trip_cost)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    oof.to_parquet(app.CACHE / "v20_production_cv.parquet", index=False)


if __name__ == "__main__":
    main()
