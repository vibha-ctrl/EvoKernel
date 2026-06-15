"""
Candidate Store — SQLite-backed persistence for the search state.

Stores every candidate (pass/fail/benchmark/profile) across all generations
so the generator can learn from the full history and the report generator
can reconstruct the search tree.
"""

import json
import sqlite3
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class Candidate:
    code: str
    kernel_type: str
    generation: int
    parent_id: Optional[str] = None

    # Set at creation
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Verification
    verify_passed: Optional[bool] = None
    verify_error_type: Optional[str] = None
    verify_error_msg: Optional[str] = None
    verify_max_error: Optional[float] = None

    # Benchmark
    latency_us: Optional[float] = None
    latency_p99_us: Optional[float] = None
    throughput_gb_s: Optional[float] = None
    bandwidth_utilization_pct: Optional[float] = None

    # Profile
    num_warps: Optional[int] = None
    num_stages: Optional[int] = None
    shared_mem_bytes: Optional[int] = None
    register_count: Optional[int] = None
    theoretical_occupancy_pct: Optional[float] = None
    dram_utilization_pct: Optional[float] = None
    l1_hit_rate_pct: Optional[float] = None
    stall_memory_dependency_pct: Optional[float] = None
    stall_long_scoreboard_pct: Optional[float] = None
    sm_active_cycles_pct: Optional[float] = None

    @property
    def label(self) -> str:
        return f"gen{self.generation}_{self.id}"

    @property
    def is_verified(self) -> bool:
        return self.verify_passed is True

    @property
    def is_benchmarked(self) -> bool:
        return self.latency_us is not None

    @property
    def is_profiled(self) -> bool:
        return self.num_warps is not None


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    kernel_type TEXT NOT NULL,
    generation INTEGER NOT NULL,
    parent_id TEXT,
    code TEXT NOT NULL,
    created_at TEXT,

    verify_passed INTEGER,
    verify_error_type TEXT,
    verify_error_msg TEXT,
    verify_max_error REAL,

    latency_us REAL,
    latency_p99_us REAL,
    throughput_gb_s REAL,
    bandwidth_utilization_pct REAL,

    num_warps INTEGER,
    num_stages INTEGER,
    shared_mem_bytes INTEGER,
    register_count INTEGER,
    theoretical_occupancy_pct REAL,
    dram_utilization_pct REAL,
    l1_hit_rate_pct REAL,
    stall_memory_dependency_pct REAL,
    stall_long_scoreboard_pct REAL,
    sm_active_cycles_pct REAL
)
"""


class CandidateStore:
    def __init__(self, db_path: str = "evokernel.db"):
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(CREATE_TABLE)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save(self, c: Candidate):
        """Insert or replace a candidate record."""
        d = asdict(c)
        # SQLite doesn't have bool — convert
        d["verify_passed"] = int(c.verify_passed) if c.verify_passed is not None else None

        cols = ", ".join(d.keys())
        placeholders = ", ".join(["?"] * len(d))
        sql = f"INSERT OR REPLACE INTO candidates ({cols}) VALUES ({placeholders})"
        with self._conn() as conn:
            conn.execute(sql, list(d.values()))

    def update_verify(self, candidate_id: str, passed: bool, error_type: str | None,
                      error_msg: str | None, max_error: float | None):
        with self._conn() as conn:
            conn.execute(
                """UPDATE candidates SET verify_passed=?, verify_error_type=?,
                   verify_error_msg=?, verify_max_error=? WHERE id=?""",
                (int(passed), error_type, error_msg, max_error, candidate_id),
            )

    def update_benchmark(self, candidate_id: str, latency_us: float, latency_p99_us: float,
                         throughput_gb_s: float, bandwidth_utilization_pct: float):
        with self._conn() as conn:
            conn.execute(
                """UPDATE candidates SET latency_us=?, latency_p99_us=?,
                   throughput_gb_s=?, bandwidth_utilization_pct=? WHERE id=?""",
                (latency_us, latency_p99_us, throughput_gb_s, bandwidth_utilization_pct,
                 candidate_id),
            )

    def update_profile(self, candidate_id: str, profile: dict):
        fields = [
            "num_warps", "num_stages", "shared_mem_bytes", "register_count",
            "theoretical_occupancy_pct", "dram_utilization_pct",
            "l1_hit_rate_pct", "stall_memory_dependency_pct",
            "stall_long_scoreboard_pct", "sm_active_cycles_pct",
        ]
        updates = {k: profile.get(k) for k in fields if k in profile}
        if not updates:
            return
        set_clause = ", ".join(f"{k}=?" for k in updates)
        with self._conn() as conn:
            conn.execute(
                f"UPDATE candidates SET {set_clause} WHERE id=?",
                [*updates.values(), candidate_id],
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get(self, candidate_id: str) -> Candidate | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM candidates WHERE id=?",
                               (candidate_id,)).fetchone()
        return _row_to_candidate(row) if row else None

    def get_generation(self, generation: int, kernel_type: str) -> list[Candidate]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidates WHERE generation=? AND kernel_type=? ORDER BY latency_us ASC",
                (generation, kernel_type),
            ).fetchall()
        return [_row_to_candidate(r) for r in rows]

    def get_best(self, kernel_type: str, n: int = 1) -> list[Candidate]:
        """Return the n fastest verified+benchmarked candidates."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM candidates
                   WHERE kernel_type=? AND verify_passed=1 AND latency_us IS NOT NULL
                   ORDER BY latency_us ASC LIMIT ?""",
                (kernel_type, n),
            ).fetchall()
        return [_row_to_candidate(r) for r in rows]

    def get_failed(self, kernel_type: str, generation: int | None = None) -> list[Candidate]:
        with self._conn() as conn:
            if generation is None:
                rows = conn.execute(
                    """SELECT * FROM candidates
                       WHERE kernel_type=? AND verify_passed=0
                       ORDER BY created_at ASC""",
                    (kernel_type,),
                ).fetchall()
                return [_row_to_candidate(r) for r in rows]
            rows = conn.execute(
                """SELECT * FROM candidates
                   WHERE kernel_type=? AND generation=? AND verify_passed=0
                   ORDER BY created_at ASC""",
                (kernel_type, generation),
            ).fetchall()
        return [_row_to_candidate(r) for r in rows]

    def get_all_tried_configs(self, kernel_type: str) -> list[dict]:
        """Return list of config dicts for all benchmarked candidates."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT num_warps, num_stages, shared_mem_bytes, latency_us
                   FROM candidates WHERE kernel_type=? AND latency_us IS NOT NULL""",
                (kernel_type,),
            ).fetchall()
        return [
            {
                "num_warps": r["num_warps"],
                "num_stages": r["num_stages"],
                "latency_us": r["latency_us"],
            }
            for r in rows
        ]

    def generation_summary(self, kernel_type: str) -> list[dict]:
        """Per-generation best latency for progress tracking."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT generation, MIN(latency_us) as best_latency_us, COUNT(*) as total,
                          SUM(CASE WHEN verify_passed=1 THEN 1 ELSE 0 END) as passed
                   FROM candidates WHERE kernel_type=?
                   GROUP BY generation ORDER BY generation ASC""",
                (kernel_type,),
            ).fetchall()
        return [dict(r) for r in rows]


def _row_to_candidate(row: sqlite3.Row) -> Candidate:
    d = dict(row)
    d["verify_passed"] = bool(d["verify_passed"]) if d["verify_passed"] is not None else None
    return Candidate(**d)
