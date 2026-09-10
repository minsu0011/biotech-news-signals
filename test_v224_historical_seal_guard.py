import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest
from v224_historical_seal_guard import SealAccessGuard, digest


def setup_members(tmp_path):
    frame = pd.DataFrame({"cert_row_index":range(1,61), "block_id":[1]*20+[2]*20+[3]*20,
                          "canonical_event_id":[f"SEC:1:{i:04}" for i in range(60)],
                          "event_time":pd.date_range("2020-01-01", periods=60, tz="UTC"),
                          "ticker":"TEST", "CIK":"1",
                          "source_family":"US_SEC", "role":"RESEARCH_SEAL"})
    frame.to_csv(tmp_path / "PRECOMMITTED_MEMBERSHIP.csv", index=False)
    (tmp_path / "PRECOMMITTED_MEMBERSHIP_SHA256.txt").write_text(digest(tmp_path / "PRECOMMITTED_MEMBERSHIP.csv"))
    return frame


def prediction(tmp_path, frame, block):
    path=tmp_path / f"BLOCK_{block:02d}_PREDICTIONS_FROZEN.csv"
    frame.loc[frame.block_id.eq(block),["canonical_event_id"]].to_csv(path,index=False)
    proof={"membership_sha":digest(tmp_path/"PRECOMMITTED_MEMBERSHIP.csv"),
           "prediction_sha":digest(path),"deterministic":True}
    (tmp_path/f"BLOCK_{block:02d}_PREDICTION_FREEZE.json").write_text(json.dumps(proof))


def test_no_outcome_before_prediction_and_no_other_block(tmp_path):
    frame=setup_members(tmp_path); guard=SealAccessGuard(tmp_path); calls=[]
    with pytest.raises(RuntimeError,match="DENIED"):
        guard.open_one(frame.canonical_event_id.iloc[0],lambda x:calls.append(x))
    prediction(tmp_path,frame,1); guard.authorize(1)
    with pytest.raises(RuntimeError,match="DENIED"):
        guard.open_one(frame.canonical_event_id.iloc[20],lambda x:calls.append(x))
    assert calls==[]


def test_tamper_and_replay_rejected(tmp_path):
    frame=setup_members(tmp_path); prediction(tmp_path,frame,1); guard=SealAccessGuard(tmp_path); guard.authorize(1)
    event=frame.canonical_event_id.iloc[0]
    guard.open_one(event,lambda x:{"event_id":x})
    with pytest.raises(RuntimeError,match="DENIED"):
        guard.open_one(event,lambda x:None)
    with pytest.raises(RuntimeError,match="ONE_SHOT"):
        SealAccessGuard(tmp_path)
    path=tmp_path/'BLOCK_01_PREDICTIONS_FROZEN.csv'
    with path.open('a') as handle:handle.write('\n')
    with pytest.raises(RuntimeError,match="INTEGRITY"):
        guard.open_one(frame.canonical_event_id.iloc[1],lambda x:None)


def test_exactly_sixty_and_chronological_blocks(tmp_path):
    frame=setup_members(tmp_path); guard=SealAccessGuard(tmp_path); calls=[]
    for block in (1,2,3):
        prediction(tmp_path,frame,block); guard.authorize(block)
        for event in frame.loc[frame.block_id.eq(block),'canonical_event_id']:
            guard.open_one(event,lambda x:calls.append(x))
        guard.finish_block(['y','fwd_ret_30m'])
    assert len(calls)==len(set(calls))==60
    assert guard.counters()['unique_outcome_rows_opened']==60
    with pytest.raises(RuntimeError,match="ORDER"):
        guard.authorize(4)
    with pytest.raises(RuntimeError,match="DENIED"):
        guard.open_one('SEC:1:0061',lambda x:None)


def test_invalid_membership_scope_is_rejected(tmp_path):
    frame=setup_members(tmp_path); frame.loc[0,'role']='FINAL_META'
    frame.to_csv(tmp_path/'PRECOMMITTED_MEMBERSHIP.csv',index=False)
    (tmp_path/'PRECOMMITTED_MEMBERSHIP_SHA256.txt').write_text(digest(tmp_path/'PRECOMMITTED_MEMBERSHIP.csv'))
    with pytest.raises(RuntimeError,match='SCOPE'):
        SealAccessGuard(tmp_path)
