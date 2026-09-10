"""Verify already-opened certificate evidence; never access source price outcomes."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import v224_prospective_locked_evaluation as locked
from v224_historical_seal_guard import digest

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'cert_research/V224_HISTORICAL_MINI_SEAL'


def read_json(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    cert=read_json('V224_HISTORICAL_MINI_SEAL_CERTIFICATE.json')
    membership=pd.read_csv(OUT/'PRECOMMITTED_MEMBERSHIP.csv',dtype=str)
    assert cert['membership_sha']==digest(OUT/'PRECOMMITTED_MEMBERSHIP.csv')
    assert cert['report_sha']==digest(OUT/'FINAL_REPORT.md')
    assert cert['opened_rows']==60 and cert['blocks']==3
    frozen=read_json('PRE_OPEN_EXECUTION_CODE_FREEZE.json')
    assert all(digest(ROOT/name)==sha for name,sha in frozen.items())
    runtime=locked.load_frozen_runtime(ROOT)
    state=read_json('CERT_WORKING_STATE_INITIAL.json')
    ledger=pd.read_csv(OUT/'SEAL_OPEN_LEDGER.csv')
    journal=[json.loads(line) for line in (OUT/'OUTCOME_ACCESS_JOURNAL.jsonl').read_text().splitlines()]
    assert len(journal)==len({r['event_id'] for r in journal})==60
    assert {r['event_id'] for r in journal}==set(membership.canonical_event_id)
    assert ledger.block_id.tolist()==[1,2,3] and ledger.row_count.tolist()==[20,20,20]
    all_rows=[];checks=[]
    for block in (1,2,3):
        proof=read_json(f'BLOCK_{block:02d}_PREDICTION_FREEZE.json')
        predpath=OUT/f'BLOCK_{block:02d}_PREDICTIONS_FROZEN.csv'
        assert proof['prediction_sha']==digest(predpath)
        assert proof['feature_sha']==digest(OUT/f'BLOCK_{block:02d}_FEATURES_FROZEN.csv')
        assert proof['state_fingerprint']==runtime.scorer.state_fingerprint(state)
        assert proof['policy']==runtime.scorer.derive_policy(state)
        prediction=pd.read_csv(predpath,float_precision='round_trip')
        # These files contain only previously authorized, already-opened 20-row evidence.
        outcomes=pd.read_csv(OUT/f'BLOCK_{block:02d}_OPENED_OUTCOMES.csv',float_precision='round_trip')
        expected=set(membership.loc[membership.block_id.eq(str(block)),'canonical_event_id'])
        assert len(prediction)==len(outcomes)==20 and set(prediction.event_id)==set(outcomes.event_id)==expected
        intents=[r for r in journal if r['block_id']==block]
        assert len(intents)==20 and all(pd.Timestamp(r['opened_at'])>=pd.Timestamp(proof['created_at']) for r in intents)
        assert all(r['prediction_sha']==proof['prediction_sha'] for r in intents)
        assert ledger.loc[ledger.block_id.eq(block),'prediction_sha'].iloc[0]==proof['prediction_sha']
        merged=prediction.merge(outcomes,on='event_id',validate='one_to_one')
        assert locked.metrics_for(merged,runtime.scorer.COST)==read_json(f'BLOCK_{block:02d}_METRICS.json')
        state=runtime.scorer.complete_block(state,prediction,outcomes)
        stored=read_json(f'CERT_WORKING_STATE_AFTER_BLOCK_{block:02d}.json')
        assert runtime.scorer.state_fingerprint(state)==runtime.scorer.state_fingerprint(stored)
        all_rows.append(merged)
        checks.append({'block':block,'prediction_and_feature_hash':True,'prediction_precedes_open':True,
                       'frozen_state_transition_exact':True,'stored_metrics_reproduced':True})
    recomputed=locked.metrics_for(pd.concat(all_rows,ignore_index=True),runtime.scorer.COST,include_bootstrap=True)
    reported=read_json('OVERALL_METRICS.json')
    assert all(reported[k]==value for k,value in recomputed.items())
    locked.write_immutable(OUT/'POST_CERT_READBACK_AUDIT.json',locked.pretty_json_bytes({
        'passed':True,'blocks':checks,'unique_opened_ids':60,'source_price_outcomes_read_by_finalizer':0,
        'additional_seal_rows_opened':0,'all_metrics_and_bootstrap_exactly_reproduced':True,
        'original_frozen_artifacts_and_execution_code_unchanged':True,
        'method':'Stored authorized certificate evidence only; no model prediction rerun and no source price reads.'}))
    overlay=ROOT/'state/V224_HISTORICAL_MINI_SEAL_ACCESS_STATE.json'
    locked.write_immutable(overlay,locked.pretty_json_bytes({
        'status':'OPENED_HISTORICAL_CERT_EVIDENCE','certification_status':cert['overall_status'],
        'source_family':'US_SEC','original_role':'RESEARCH_SEAL_POOL','opened_rows':60,
        'opened_event_ids':membership.canonical_event_id.tolist(),'membership_sha':cert['membership_sha'],
        'ledger':str((OUT/'SEAL_OPEN_LEDGER.csv').relative_to(ROOT)),
        'reusable_as_independent_seal':False,'additional_opening_authorized':False,
        'original_role_registry_unchanged':True,
        'instruction':'Subtract these IDs from unopened reserve in every future inventory. Do not retune on or recertify this 60-row seal.'}))
    for name in frozen:
        locked.write_immutable(OUT/'code'/Path(name).name,(ROOT/name).read_bytes())
    locked.write_immutable(OUT/'code'/Path(__file__).name,Path(__file__).read_bytes())
    paths=set(p for p in OUT.rglob('*') if p.is_file() and p.name not in {'SHA256_MANIFEST.csv','SHA256_MANIFEST_SHA256.txt'})
    paths.add(overlay)
    paths.update(ROOT/name for name in frozen)
    paths.add(Path(__file__))
    for name in read_json('PRE_CERT_INTEGRITY.json')['files']:
        paths.add(ROOT/locked.FROZEN_REL/name)
    for row in read_json('EXACT_ACCESSION_TEXT_AUDIT_RECOVERY.json')['rows']:
        paths.update(ROOT/source['path'] for source in row['raw_sources'])
    manifest=pd.DataFrame([{'root_relative_path':str(path.relative_to(ROOT)).replace('\\','/'),
                           'bytes':path.stat().st_size,'sha256':digest(path)} for path in sorted(paths)])
    manifest_sha=locked.write_immutable(OUT/'SHA256_MANIFEST.csv',manifest.to_csv(index=False,lineterminator='\n').encode())
    locked.write_immutable(OUT/'SHA256_MANIFEST_SHA256.txt',(manifest_sha+'\n').encode())
    print(json.dumps({'post_cert_readback':'PASS','manifest_files':len(manifest),
                      'manifest_sha256':manifest_sha,'status':cert['overall_status'],
                      'additional_outcomes_opened':0}),flush=True)


if __name__=='__main__':
    main()
