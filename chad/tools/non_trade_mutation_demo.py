import json, hashlib, sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/ubuntu/chad_finale")
DB = ROOT / "runtime/exec_state_paper.sqlite3"
OUTCFG = ROOT / "runtime/non_trade_mutation_demo.json"

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def claim_idempotency(conn: sqlite3.Connection, key: str, scope: str, payload_hash: str, meta: dict) -> bool:
    try:
        conn.execute(
            "insert into idempotency_events(key, scope, first_seen_utc, payload_hash, meta_json) values(?,?,?,?,?)",
            (key, scope, utc_now(), payload_hash, json.dumps(meta, separators=(",", ":"), sort_keys=True)),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def main() -> int:
    mutation = {
        "type": "config_change",
        "path": str(OUTCFG),
        "set": {"demo_flag": True, "demo_value": 123},
    }
    payload = json.dumps(mutation, separators=(",", ":"), sort_keys=True)
    payload_hash = sha256(payload)
    idem_key = f"config_change:{payload_hash}"

    conn = sqlite3.connect(DB)
    ok = claim_idempotency(conn, idem_key, "config_change", payload_hash, {"mutation": mutation})

    if not ok:
        print(json.dumps({"ok": True, "applied": False, "reason": "duplicate_suppressed", "idempotency_key": idem_key}))
        return 0

    OUTCFG.write_text(json.dumps(mutation["set"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "applied": True, "idempotency_key": idem_key, "out": str(OUTCFG)}))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
