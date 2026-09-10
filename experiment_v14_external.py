"""External check of the frozen-shaped V14 estimator on opened V13 sources."""
from __future__ import annotations

import json

import pandas as pd

import bio_news_30m_v3 as app


V14 = {"US": "SEC_V14_DEV", "KR": "NAVER_NEWS_V14_DEV"}
V13 = {
    "US": ("SEC_V13_DEV", "SEC_V13_SEAL"),
    "KR": ("NAVER_NEWS_V13_DEV", "NAVER_NEWS_V13_SEAL"),
}


def report(frame: pd.DataFrame, name: str, cfg: app.Config) -> None:
    if frame.empty or frame.y.nunique() < 2:
        return
    frame = frame.copy()
    frame["eval_weight"] = 1.0
    metrics = app.evaluate(frame, cfg.fixed_highconf_threshold, cfg.round_trip_cost)
    print(json.dumps({
        "set": name, "n": len(frame), "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"], "auc": metrics["auc"],
        "edge": metrics["edge_vs_naive"], "coverage": metrics["highconf_coverage"],
        "highconf_accuracy": metrics["highconf_accuracy"],
        "highconf_net": metrics["strategy_mean_signed_net"],
    }, allow_nan=True))


def main() -> None:
    cfg = app.Config()
    data = app.load_search_frame(cfg)
    data = data[data.y.notna()].copy()
    for market in ("KR", "US"):
        train = data[data.source.eq(V14[market])].copy().reset_index(drop=True)
        model = app.make_model(app.model_specs(market)[0])
        train_features = app.model_frame(train)
        model.fit(train_features, train.y.astype(int))
        oof_probability, _ = model.crossfit_output(train_features)
        internal = train[[
            "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
        ]].copy()
        internal["prob"] = oof_probability
        report(internal, f"{market}_V14_DEV_OOF", cfg)
        predicted = []
        for source in V13[market]:
            test = data[data.source.eq(source)].copy().reset_index(drop=True)
            output = test[[
                "event_id", "event_time_utc", "market", "source", "ticker", "y", "fwd_ret_30m",
            ]].copy()
            output["prob"] = model.predict_proba(app.model_frame(test))[:, 1]
            report(output, f"{market}_{source}", cfg)
            predicted.append(output)
        report(pd.concat(predicted, ignore_index=True), f"{market}_V13_ALL", cfg)


if __name__ == "__main__":
    main()
