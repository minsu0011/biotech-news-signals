import copy
import json
import numpy as np
import pandas as pd
import pytest
import shutil
import v224_historical_mini_seal_run as run
from test_v224_historical_seal_guard import setup_members


def test_feature_formula_parity_with_single_entry_open(tmp_path,monkeypatch):
    times=pd.date_range('2026-09-01T13:30:00Z',periods=180,freq='min')
    values=100+np.sin(np.arange(180)/9)+np.arange(180)*.01
    bars=pd.DataFrame({'timestamp':times,'open':values,'close':values+.01,'high':values+.1,
                       'low':values-.1,'volume':np.arange(180)+100})
    path=tmp_path/'bars.parquet';bars.to_parquet(path,index=False)
    event=pd.Series({'event_id':'SEC:TEST:1','event_time_utc':'2026-09-01T14:30:00Z','ticker':'TEST',
                     'headline':'Phase 2 endpoint met','body':'Item 8.01 clinical trial',
                     'form':'8-K','event_type':'SEC_8-K','price_provider':'MASSIVE'})
    expected,reason,_=run.causal.causal_feature_snapshot(event,path)
    assert reason=='OK'
    actual,reason,audit=run.strict_feature(event,path,pd.Timestamp('2026-09-01T14:32:00Z'))
    assert reason=='OK' and audit['entry_open_rows_loaded']==1
    expected['price_source']=actual['price_source']
    assert actual==expected
    # Arbitrary changes after entry cannot influence features.
    later=bars.timestamp.gt(pd.Timestamp('2026-09-01T14:32:00Z'))
    bars.loc[later,['open','close','high','low','volume']]=999999
    bars.to_parquet(path,index=False)
    unchanged,_,_=run.strict_feature(event,path,pd.Timestamp('2026-09-01T14:32:00Z'))
    assert unchanged==actual


def test_real_frozen_model_three_blocks_prediction_before_synthetic_outcomes(tmp_path,monkeypatch):
    m=setup_members(tmp_path)
    snapshots=[]
    for row in m.itertuples(index=False):
        snapshots.append({'event_id':row.canonical_event_id,'feature_row':{
            'event_id':row.canonical_event_id,'event_time_utc':str(row.event_time),
            'source_family':'US_SEC','headline':'Item 8.01 results','body':'Phase 2 clinical trial results.',
            'form':'8-K','event_type':'SEC_8-K','regular_session_eligible':True,'pre_ret_1':0.01}})
    (tmp_path/'CAUSAL_FEATURE_SNAPSHOTS.json').write_text(json.dumps({'rows':snapshots}))
    monkeypatch.setattr(run,'OUT',tmp_path)
    monkeypatch.setattr(run,'validate_preopen',lambda membership,by_id:None)
    opened=[]; reports=[]
    def synthetic(record):
        identity=record['event_id']; index=m.canonical_event_id.tolist().index(identity)
        block=index//20+1
        assert (tmp_path/f'BLOCK_{block:02d}_PREDICTIONS_FROZEN.csv').is_file()
        assert (tmp_path/f'BLOCK_{block:02d}_PREDICTION_FREEZE.json').is_file()
        if block>1:assert (tmp_path/f'CERT_WORKING_STATE_AFTER_BLOCK_{block-1:02d}.json').is_file()
        opened.append(identity)
        return {'event_id':identity,'y':index%2,'fwd_ret_30m':.01 if index%2 else -.01}
    monkeypatch.setattr(run,'outcome_for',synthetic)
    monkeypatch.setattr(run,'final_report',lambda scored,blocks,guard,runtime,reason=None:reports.append(
        {'rows':len(scored),'blocks':len(blocks),'counters':guard.counters(),'reason':reason}))
    run.evaluate()
    assert len(opened)==len(set(opened))==60
    assert len(reports)==1 and reports[0]['rows']==60 and reports[0]['reason'] is None
    assert reports[0]['counters']['FINAL_META_OUTCOME_ROWS_READ']==0
    state=json.loads((tmp_path/'CERT_WORKING_STATE_AFTER_BLOCK_03.json').read_text())
    assert state['machine_state']['confirmation_rows_completed']==60


@pytest.mark.parametrize('correct',[True,False])
def test_synthetic_final_report_fixed_gates_and_all_outputs(tmp_path,monkeypatch,correct):
    m=setup_members(tmp_path)
    source=run.OUT
    for name in ['PRE_CERT_INTEGRITY.json','METADATA_PREFLIGHT.json']:
        shutil.copyfile(source/name,tmp_path/name)
    hashes={'v224_historical_seal_guard.py':run.digest(run.ROOT/'v224_historical_seal_guard.py')}
    (tmp_path/'PRE_OPEN_EXECUTION_CODE_FREEZE.json').write_text(json.dumps(hashes))
    monkeypatch.setattr(run,'OUT',tmp_path)
    guard=run.SealAccessGuard(tmp_path);guard.opened=set(m.canonical_event_id);guard.completed=3
    y=np.arange(60)%2;pred=y.astype(bool) if correct else ~y.astype(bool)
    scores=pd.DataFrame({'event_id':m.canonical_event_id,'event_time_utc':m.event_time,'y':y,
                         'prediction':pred,'probability':np.where(pred,.8,.2),
                         'raw_probability':np.where(pred,.8,.2),'high_conf':True,
                         'confidence_signal':.9,'fwd_ret_30m':np.where(y,.01,-.01)})
    runtime=run.locked.load_frozen_runtime(run.ROOT)
    policy=runtime.scorer.derive_policy(runtime.initial_state)
    blocks=[{'block':b,'first_date':str(scores.event_time_utc.iloc[(b-1)*20]),
             'last_date':str(scores.event_time_utc.iloc[b*20-1]),'policy':policy,
             'metrics':run.locked.metrics_for(scores.iloc[(b-1)*20:b*20],runtime.scorer.COST)} for b in (1,2,3)]
    run.final_report(scores,blocks,guard,runtime)
    cert=json.loads((tmp_path/'V224_HISTORICAL_MINI_SEAL_CERTIFICATE.json').read_text())
    assert cert['overall_status'].endswith('_PASS' if correct else '_FAIL')
    assert cert['opened_rows']==60 and cert['blocks']==3
    assert cert['report_sha']==run.digest(tmp_path/'FINAL_REPORT.md')
    assert (tmp_path/'FINAL_REPORT.md').read_text().startswith('# OPENED SEAL REGION')
    assert 'TEN REQUIRED QUESTIONS' in (tmp_path/'FINAL_REPORT.md').read_text()
    assert 'NO RETUNING WAS PERFORMED.' in (tmp_path/'CONSOLE_SUMMARY.txt').read_text()


def test_preopen_rejects_future_block_state_before_prices(tmp_path,monkeypatch):
    m=setup_members(tmp_path).astype(str)
    shutil.copyfile(run.ROOT/run.locked.FROZEN_REL/run.locked.FROZEN_STATE_NAME,tmp_path/'CERT_WORKING_STATE_INITIAL.json')
    (tmp_path/'CONTAMINATION_CLEARANCE.json').write_text(json.dumps({'status':'PASS','identity_or_accession_overlap':0}))
    monkeypatch.setattr(run,'OUT',tmp_path)
    by_id={r.canonical_event_id:{'actual_exit_metadata':'2030-01-01T00:00:00Z',
            'feature_row':{'actual_entry_time_utc':r.event_time}} for r in m.itertuples(index=False)}
    with pytest.raises(RuntimeError,match='BLOCK_STATE_USES_UNAVAILABLE_OUTCOME'):
        run.validate_preopen(m,by_id)
    assert not (tmp_path/'OUTCOME_ACCESS_JOURNAL.jsonl').exists()
