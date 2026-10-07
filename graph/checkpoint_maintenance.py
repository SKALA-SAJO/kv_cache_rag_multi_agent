"""체크포인트 DB 정리 — 정상 종료한 run의 중간 스냅샷만 지운다 (README "지속성 비용" ③).

배경: SqliteSaver는 superstep마다 변경분이 아니라 State 전체를 새 행으로 저장한다. Supervisor 1스텝이
체크포인트 약 2개라 한 번 실행에 수 MB가 쌓이고(스텝 상한까지 가면 12~14MB), 실행마다 누적된다.

정리 대상과 범위:
  - 대상 : 정상 종료한 run(thread)의 마지막 체크포인트를 제외한 이전 체크포인트와 그 writes.
    마지막 체크포인트에 최종 State 전체(근거·보고서·verdict)가 들어 있어 페이로드 손실은 없다.
    인용문을 줄이는 방식(Truncation)과 달리 "원본 보존"이 유지된다.
  - 비대상 : Ctrl+C·예외로 끊긴 run, 실패한 run. `--resume`에 필요한 중간 체크포인트를 보존한다.
  - 한계 : 정리는 종료 후에 하므로 실행 중 최대 용량은 줄지 않는다(그건 재작업·스텝 예산이 상한).
    정리한 run은 `--resume`으로 이어갈 수 없다(이미 끝난 run이라 필요 없음).
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any


def is_run_finished(graph: Any, config: dict) -> bool:
    """정상 종료 판정: 더 실행할 노드가 없고(next == ()) 최종 보고서가 State에 있다.

    체크포인터 연결이 열려 있는 동안 호출해야 한다(상태를 체크포인트에서 읽는다).
    """
    snapshot = graph.get_state(config)
    return not snapshot.next and bool(snapshot.values.get("final_report"))


def _db_bytes(db_path: Path) -> int:
    """DB 본체 + WAL 파일 크기 합(WAL에 아직 반영 안 된 쓰기까지 포함)."""
    return sum(os.path.getsize(p) for p in (db_path, Path(f"{db_path}-wal")) if os.path.exists(p))


def prune_checkpoints(db_path: str | Path, thread_id: str) -> tuple[int, int]:
    """thread_id의 마지막 체크포인트만 남기고 이전 체크포인트·writes를 지운 뒤 DB를 축소한다.

    마지막 = checkpoint_id 최대값(LangGraph가 최신을 고르는 기준과 같다). 다른 thread와 존재하지 않는
    thread는 건드리지 않는다. (정리 전 바이트, 정리 후 바이트)를 돌려준다.
    체크포인터 연결을 닫은 뒤에 호출해야 VACUUM이 잠금에 막히지 않는다.
    """
    db_path = Path(db_path)
    before = _db_bytes(db_path)
    # VACUUM과 wal_checkpoint는 트랜잭션 밖에서만 실행되므로 autocommit(isolation_level=None)으로 연다.
    conn = sqlite3.connect(str(db_path), isolation_level=None)
    try:
        latest = conn.execute(
            "SELECT MAX(checkpoint_id) FROM checkpoints WHERE thread_id = ?", (thread_id,)
        ).fetchone()[0]
        if latest is None:
            return before, before
        conn.execute("BEGIN")
        conn.execute("DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id <> ?", (thread_id, latest))
        conn.execute("DELETE FROM writes WHERE thread_id = ? AND checkpoint_id <> ?", (thread_id, latest))
        conn.execute("COMMIT")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("VACUUM")
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")  # VACUUM 결과가 WAL에 남아 파일이 커 보이지 않게 비운다
    finally:
        conn.close()
    return before, _db_bytes(db_path)
