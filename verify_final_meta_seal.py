"""Independent verifier for a committed FINAL_META certificate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT=Path(__file__).resolve().parent


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(4*1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def atomic_json(value:Any,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f".tmp.{os.getpid()}")
    with temporary.open("w",encoding="utf-8",newline="\n") as handle:
        json.dump(value,handle,ensure_ascii=False,indent=2);handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def main()->int:
    parser=argparse.ArgumentParser();parser.add_argument("--certificate",required=True)
    parser.add_argument("--write-marker",action="store_true");args=parser.parse_args()
    certificate=Path(args.certificate).resolve();report=json.loads(certificate.read_text(encoding="utf-8"))
    checks=report.get("checks",[]);source_sha=str(report.get("source_sha256",""))
    expected_event_hash=hashlib.sha256("\n".join(sorted(str(value) for value in
        report.get("event_ids",[]))).encode()).hexdigest()
    intent=ROOT/"state"/"seals"/source_sha/"OPEN_INTENT.json"
    result_path=ROOT/"state"/"seals"/source_sha/"RESULT.json"
    location_valid=(certificate.name=="CERTIFICATE.json" and certificate.parent.name==source_sha and
                    certificate.parent.parent.name=="FINAL_META" and certificate.is_relative_to(ROOT/"cert"))
    result=json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    result_valid=(result.get("status")=="FINAL_META_SEAL_PASS" and
                  result.get("certificate_sha256")==sha256(certificate))
    valid=(report.get("stage")=="final_meta" and
        report.get("status")=="FINAL_META_SEAL_PASS" and report.get("contract_integrity") is True and
        report.get("no_leakage") is True and report.get("passed")==12 and report.get("total")==12 and
        len(checks)==12 and all(item.get("pass") is True for item in checks) and
        report.get("seal_reuse_forbidden") is True and report.get("opened_once") is True and
        len(source_sha)==64 and location_valid and intent.is_file() and result_valid and
        report.get("event_ids_sha256")==expected_event_hash and len(report.get("event_ids",[]))>=500)
    if not valid:print(json.dumps({"verified":False,"certificate":str(certificate)},indent=2));return 2
    if args.write_marker:
        try:relative=str(certificate.relative_to(ROOT))
        except ValueError:relative=str(certificate)
        atomic_json({"status":"FINAL_PASS","certificate":relative,
            "certificate_sha256":sha256(certificate)},ROOT/"state"/"FINAL_PASS")
    print(json.dumps({"verified":True,"certificate_sha256":sha256(certificate),
                      "marker_written":args.write_marker},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
