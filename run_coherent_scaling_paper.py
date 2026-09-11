#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coherent-scaling simulation.

Required file in the same directory
-----------------------------------
    logical_csb.py


A "repeat" means one COMPLETE scaling sweep across every epsilon in
PAPER_SCHEDULE.  The outer loop is repeat_id, then epsilon.  Therefore, once
repeat 1 finishes, the analysis notebook can already draw the full scaling
curve while repeat 2 is still running.

Increase N_REPEATS later and rerun this file.  Existing epsilon points for
earlier repeats are skipped.

All persistent numerical data are kept in one SQLite database.
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import hashlib
import json
import math
import sqlite3
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False

import logical_csb as base


# =============================================================================
# Experiment settings
# =============================================================================

# Complete scaling sweeps and parallelism.
N_REPEATS = 10
N_JOBS = -1
BASE_SEED = 1234
DATABASE_FILE = Path("coherent_phase_gate_scaling.sqlite")

# Target logical cycle and FTEC settings.
TARGET_GATE = "RZ90"
MIN_EC_ROUNDS = 2
MAX_EC_ROUNDS = 4
DECODER_CACHE = Path("steane_flag_decoder_stabilizer_max4.pkl")

# Coherent physical Z-rotation model.
TWO_QUBIT_ANGLE_FACTOR = 2.0
IDLE_ANGLE_FACTOR = 1.0

# Logical preparation, measurement, and QEC settings.
CIRCUIT_OPTIONS = {
    "prep_gates_noisy": True,
    "prep_qec_after_gate": True,
    "measurement_basis_gates_noisy": True,
    "qec_after_each_logical_gate": True,
    "fallback_on_unverified": True,
}

# CSB sampling settings.
CSB_STEPS = list(range(13))
CSB_SHOTS = 1000
CSB_CHUNK_SHOTS = 100

# (physical coherent angle epsilon, CSB repetition factor r)
PAPER_SCHEDULE = [
    (0.00300, 1001),
    (0.00424, 253),
    (0.00600, 65),
    (0.00849, 17),
    (0.01200, 5),
    (0.01700, 1),
]



_REQUIRED_API = [
    "get_or_build_decoder",
    "get_worker_decoder",
    "get_csb_branches",
    "simulate_coherent_csb_point",
    "make_homogeneous_coherent_noise",
    "fit_single_gate_spectrum",
]

_missing = [name for name in _REQUIRED_API if not hasattr(base, name)]
if _missing:
    raise AttributeError(
        "The loaded logical_csb.py is missing required coherent-scaling API: "
        + ", ".join(_missing)
    )


def get_decoder(verbose: bool = True):
    return base.get_or_build_decoder(
        cache_file=DECODER_CACHE,
        min_rounds=MIN_EC_ROUNDS,
        max_rounds=MAX_EC_ROUNDS,
        verbose=bool(verbose),
    )


def make_paper_coherent_noise(epsilon: float):
    return base.make_homogeneous_coherent_noise(
        strength=float(epsilon),
        two_qubit_factor=TWO_QUBIT_ANGLE_FACTOR,
        idle_factor=IDLE_ANGLE_FACTOR,
    )


def _chunk_sizes(total: int, chunk_shots: int) -> List[int]:
    total = int(total)
    chunk_shots = max(1, int(chunk_shots))
    sizes = []
    while total > 0:
        n = min(total, chunk_shots)
        sizes.append(n)
        total -= n
    return sizes


def _combine_chunks(sub: pd.DataFrame) -> Dict[str, Any]:
    sub = sub.sort_values("chunk_id")
    shots = sub["shots"].to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    stderrs = sub["stderr"].to_numpy(dtype=float)

    total = int(np.sum(shots))
    mean = float(np.sum(shots * means) / max(total, 1))

    if total > 1:
        variances = (stderrs * np.sqrt(np.maximum(shots, 1.0))) ** 2
        ss_within = float(np.sum(np.maximum(shots - 1, 0) * variances))
        ss_between = float(np.sum(shots * (means - mean) ** 2))
        sample_var = max((ss_within + ss_between) / (total - 1), 0.0)
        stderr = math.sqrt(sample_var / total)
    else:
        stderr = 0.0

    first = sub.iloc[0]
    return {
        "target_gate": TARGET_GATE,
        "branch": first["branch"],
        "branch_kind": first["branch_kind"],
        "initial": first["initial"],
        "measure": first["measure"],
        "measure_sign": int(first["measure_sign"]),
        "step": int(first["step"]),
        "rep": int(first["rep"]),
        "depth": int(first["depth"]),
        "mean": mean,
        "stderr": float(stderr),
        "shots": total,
        "requested_total_shots": int(first["requested_total_shots"]),
        "unverified_rate": float(
            np.sum(shots * sub["unverified_rate"].to_numpy(dtype=float))
            / max(total, 1)
        ),
        "n_chunks": int(len(sub)),
    }


def run_phase_gate_csb(
    decoder,
    noise,
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run one complete RZ(pi/2) CSB data point for one epsilon and one repeat.

    Returns both the aggregated CSB signals and every chunk-level result so the
    complete simulation data can be stored.
    """
    steps = [int(step) for step in steps]
    rep = int(rep)
    shots = int(shots)
    chunk_shots = max(1, int(chunk_shots))

    rng = np.random.default_rng(int(seed))
    tasks = []
    point_id = 0

    for branch_name, branch in base.get_csb_branches(TARGET_GATE).items():
        branch_shots = (
            max(1, shots // 2)
            if branch["kind"].startswith("equatorial")
            else shots
        )

        for step in steps:
            depth = rep * step
            for chunk_id, n_chunk in enumerate(
                _chunk_sizes(branch_shots, chunk_shots)
            ):
                tasks.append(
                    {
                        "point_id": point_id,
                        "chunk_id": chunk_id,
                        "target_gate": TARGET_GATE,
                        "branch": branch_name,
                        "branch_kind": branch["kind"],
                        "initial": branch["initial"],
                        "measure": branch["measure"],
                        "measure_sign": int(branch.get("measure_sign", +1)),
                        "step": step,
                        "rep": rep,
                        "depth": depth,
                        "shots": int(n_chunk),
                        "requested_total_shots": shots,
                        "seed": int(rng.integers(2**32 - 1)),
                    }
                )
            point_id += 1

    cache_path = str(DECODER_CACHE)
    if not DECODER_CACHE.exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    def run_task(task):
        worker_decoder = base.get_worker_decoder(
            cache_file=cache_path,
            min_rounds=MIN_EC_ROUNDS,
            max_rounds=MAX_EC_ROUNDS,
        )
        mean, stderr, unverified_rate = base.simulate_coherent_csb_point(
            worker_decoder,
            noise,
            CIRCUIT_OPTIONS,
            TARGET_GATE,
            task["branch"],
            task["depth"],
            task["shots"],
            task["seed"],
        )
        return {
            **task,
            "mean": float(mean),
            "stderr": float(stderr),
            "unverified_rate": float(unverified_rate),
        }

    if verbose:
        print("Coherent logical RZ(pi/2) CSB")
        print("  steps       =", steps)
        print("  rep         =", rep)
        print("  true depths =", [rep * step for step in steps])
        print("  shots       =", shots)
        print("  chunk shots =", chunk_shots)
        print("  n_jobs      =", n_jobs)
        print("  noise       =", noise)

    if JOBLIB_AVAILABLE and int(n_jobs) != 1 and len(tasks) > 1:
        chunk_rows = Parallel(
            n_jobs=int(n_jobs),
            backend="loky",
            batch_size=1,
        )(
            delayed(run_task)(task)
            for task in tasks
        )
    else:
        chunk_rows = [run_task(task) for task in tasks]

    chunk_df = (
        pd.DataFrame(chunk_rows)
        .sort_values(["branch", "step", "chunk_id"])
        .reset_index(drop=True)
    )

    signal_rows = [
        _combine_chunks(sub)
        for _, sub in chunk_df.groupby("point_id", sort=True)
    ]

    signal_df = (
        pd.DataFrame(signal_rows)
        .sort_values(["branch", "step"])
        .reset_index(drop=True)
    )

    return signal_df, chunk_df


def fit_phase_gate_spectrum(signal_df: pd.DataFrame) -> Dict[str, float]:
    result = base.fit_single_gate_spectrum(signal_df, TARGET_GATE)
    return {
        key: float(value) if isinstance(value, (int, float, np.number)) else value
        for key, value in result.items()
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect_database() -> sqlite3.Connection:
    DATABASE_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_FILE, timeout=60.0)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = FULL")
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_sessions (
            session_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            requested_total_repeats INTEGER NOT NULL,
            n_jobs INTEGER NOT NULL,
            elapsed_seconds REAL,
            completed_points_this_session INTEGER NOT NULL DEFAULT 0,
            error_message TEXT
        );

        CREATE TABLE IF NOT EXISTS point_results (
            epsilon REAL NOT NULL,
            rep INTEGER NOT NULL,
            repeat_id INTEGER NOT NULL,
            repeat_seed INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            started_at TEXT NOT NULL,
            completed_at TEXT NOT NULL,
            elapsed_seconds REAL NOT NULL,
            g_offdiag REAL NOT NULL,
            delta_theta REAL NOT NULL,
            g_axis REAL NOT NULL,
            process_fidelity REAL,
            process_infidelity REAL,
            average_gate_fidelity REAL,
            average_gate_infidelity REAL,
            PRIMARY KEY (epsilon, rep, repeat_id),
            FOREIGN KEY (session_id) REFERENCES run_sessions(session_id)
        );

        CREATE TABLE IF NOT EXISTS csb_signals (
            epsilon REAL NOT NULL,
            rep INTEGER NOT NULL,
            repeat_id INTEGER NOT NULL,
            branch TEXT NOT NULL,
            branch_kind TEXT NOT NULL,
            initial_state TEXT NOT NULL,
            measure TEXT NOT NULL,
            measure_sign INTEGER NOT NULL,
            step INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            mean REAL NOT NULL,
            stderr REAL NOT NULL,
            shots INTEGER NOT NULL,
            requested_total_shots INTEGER NOT NULL,
            unverified_rate REAL NOT NULL,
            n_chunks INTEGER NOT NULL,
            PRIMARY KEY (epsilon, rep, repeat_id, branch, step),
            FOREIGN KEY (epsilon, rep, repeat_id)
                REFERENCES point_results(epsilon, rep, repeat_id)
        );

        CREATE TABLE IF NOT EXISTS csb_chunks (
            epsilon REAL NOT NULL,
            rep INTEGER NOT NULL,
            repeat_id INTEGER NOT NULL,
            branch TEXT NOT NULL,
            step INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            depth INTEGER NOT NULL,
            shots INTEGER NOT NULL,
            chunk_seed INTEGER NOT NULL,
            mean REAL NOT NULL,
            stderr REAL NOT NULL,
            unverified_rate REAL NOT NULL,
            PRIMARY KEY (
                epsilon, rep, repeat_id, branch, step, chunk_id
            ),
            FOREIGN KEY (epsilon, rep, repeat_id)
                REFERENCES point_results(epsilon, rep, repeat_id)
        );
        """
    )
    conn.commit()


def settings_json() -> str:
    data = {
        "target_gate": TARGET_GATE,
        "min_ec_rounds": MIN_EC_ROUNDS,
        "max_ec_rounds": MAX_EC_ROUNDS,
        "decoder_cache": str(DECODER_CACHE),
        "circuit_options": CIRCUIT_OPTIONS,
        "csb_steps": CSB_STEPS,
        "csb_shots": CSB_SHOTS,
        "csb_chunk_shots": CSB_CHUNK_SHOTS,
        "schedule": [
            {"epsilon": epsilon, "rep": rep}
            for epsilon, rep in PAPER_SCHEDULE
        ],
        "epsilon_1q_factor": 1.0,
        "epsilon_2q_factor": TWO_QUBIT_ANGLE_FACTOR,
        "epsilon_idle_factor": IDLE_ANGLE_FACTOR,
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def source_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes_json() -> str:
    # The runner depends only on logical_csb.py.  Experiment settings are
    # tracked separately by settings_json().
    data = {
        "logical_csb.py": source_hash(Path(base.__file__).resolve()),
    }
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def check_metadata(conn: sqlite3.Connection) -> None:
    current = {
        "paper_settings": settings_json(),
        "source_hashes": source_hashes_json(),
    }
    existing = dict(conn.execute("SELECT key, value FROM metadata").fetchall())

    if not existing:
        with conn:
            conn.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                list(current.items()),
            )
        return

    for key, value in current.items():
        if existing.get(key) != value:
            raise RuntimeError(
                f"Database metadata mismatch for {key}. "
                "Do not mix results from different settings or source versions "
                "in the same database."
            )


def point_exists(
    conn: sqlite3.Connection,
    epsilon: float,
    rep: int,
    repeat_id: int,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM point_results
        WHERE epsilon = ? AND rep = ? AND repeat_id = ?
        """,
        (float(epsilon), int(rep), int(repeat_id)),
    ).fetchone()
    return row is not None


def deterministic_seed(epsilon: float, rep: int, repeat_id: int) -> int:
    point_tag = int.from_bytes(
        hashlib.blake2b(
            f"{float(epsilon):.17g}|{int(rep)}".encode("utf-8"),
            digest_size=4,
        ).digest(),
        byteorder="little",
        signed=False,
    )
    ss = np.random.SeedSequence(
        [BASE_SEED, point_tag, int(repeat_id), 0xC05B]
    )
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def start_session(conn: sqlite3.Connection) -> tuple[str, float]:
    session_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + hashlib.blake2b(
            str(time.time_ns()).encode(), digest_size=3
        ).hexdigest()
    )
    with conn:
        conn.execute(
            """
            INSERT INTO run_sessions(
                session_id, started_at, status,
                requested_total_repeats, n_jobs
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, utc_now(), "running", int(N_REPEATS), int(N_JOBS)),
        )
    return session_id, time.perf_counter()


def finish_session(
    conn: sqlite3.Connection,
    session_id: str,
    start_perf: float,
    status: str,
    completed_points: int,
    error_message: str | None = None,
) -> None:
    elapsed = time.perf_counter() - start_perf
    with conn:
        conn.execute(
            """
            UPDATE run_sessions
            SET finished_at = ?,
                status = ?,
                elapsed_seconds = ?,
                completed_points_this_session = ?,
                error_message = ?
            WHERE session_id = ?
            """,
            (
                utc_now(),
                status,
                float(elapsed),
                int(completed_points),
                error_message,
                session_id,
            ),
        )


def save_point(
    conn: sqlite3.Connection,
    *,
    epsilon: float,
    rep: int,
    repeat_id: int,
    seed: int,
    session_id: str,
    started_at: str,
    elapsed_seconds: float,
    signal_df: pd.DataFrame,
    chunk_df: pd.DataFrame,
    fit: dict,
) -> None:
    # One transaction: notebook readers see either the complete point or none.
    with conn:
        conn.execute(
            """
            INSERT INTO point_results(
                epsilon, rep, repeat_id, repeat_seed, session_id,
                started_at, completed_at, elapsed_seconds,
                g_offdiag, delta_theta, g_axis,
                process_fidelity, process_infidelity,
                average_gate_fidelity, average_gate_infidelity
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                float(epsilon),
                int(rep),
                int(repeat_id),
                int(seed),
                session_id,
                started_at,
                utc_now(),
                float(elapsed_seconds),
                float(fit["g_offdiag"]),
                float(fit["delta_theta"]),
                float(fit["g_axis"]),
                float(fit.get("process_fidelity", np.nan)),
                float(fit.get("process_infidelity", np.nan)),
                float(fit.get("average_gate_fidelity", np.nan)),
                float(fit.get("average_gate_infidelity", np.nan)),
            ),
        )

        conn.executemany(
            """
            INSERT INTO csb_signals(
                epsilon, rep, repeat_id,
                branch, branch_kind, initial_state, measure, measure_sign,
                step, depth, mean, stderr, shots, requested_total_shots,
                unverified_rate, n_chunks
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    float(epsilon),
                    int(rep),
                    int(repeat_id),
                    str(row.branch),
                    str(row.branch_kind),
                    str(row.initial),
                    str(row.measure),
                    int(row.measure_sign),
                    int(row.step),
                    int(row.depth),
                    float(row.mean),
                    float(row.stderr),
                    int(row.shots),
                    int(row.requested_total_shots),
                    float(row.unverified_rate),
                    int(row.n_chunks),
                )
                for row in signal_df.itertuples(index=False)
            ],
        )

        conn.executemany(
            """
            INSERT INTO csb_chunks(
                epsilon, rep, repeat_id,
                branch, step, chunk_id, depth, shots, chunk_seed,
                mean, stderr, unverified_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    float(epsilon),
                    int(rep),
                    int(repeat_id),
                    str(row.branch),
                    int(row.step),
                    int(row.chunk_id),
                    int(row.depth),
                    int(row.shots),
                    int(row.seed),
                    float(row.mean),
                    float(row.stderr),
                    float(row.unverified_rate),
                )
                for row in chunk_df.itertuples(index=False)
            ],
        )


def progress_table(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            repeat_id,
            COUNT(*) AS completed_epsilon_points,
            SUM(elapsed_seconds) AS accumulated_point_time_seconds
        FROM point_results
        GROUP BY repeat_id
        ORDER BY repeat_id
        """,
        conn,
    )


def main() -> None:
    if N_REPEATS < 1:
        raise ValueError("N_REPEATS must be >= 1")

    conn = connect_database()
    ensure_schema(conn)
    check_metadata(conn)

    decoder = get_decoder(verbose=True)
    session_id, session_start = start_session(conn)
    completed_this_session = 0

    print("Database:", DATABASE_FILE.resolve())
    print("Target total complete scaling repeats:", N_REPEATS)
    print("Paper schedule:", PAPER_SCHEDULE)

    try:
        # repeat_id is OUTER loop: repeat 1 becomes a complete scaling curve
        # before repeat 2 starts.
        for repeat_id in range(N_REPEATS):
            print("\n" + "#" * 80)
            print(f"SCALING REPEAT {repeat_id + 1}/{N_REPEATS}")
            print("#" * 80)

            for epsilon, rep in PAPER_SCHEDULE:
                if point_exists(conn, epsilon, rep, repeat_id):
                    print(
                        f"SKIP epsilon={epsilon:.5f}, repeat={repeat_id + 1}: "
                        "already stored"
                    )
                    continue

                seed = deterministic_seed(epsilon, rep, repeat_id)
                noise = make_paper_coherent_noise(epsilon)

                print("\n" + "=" * 80)
                print(
                    f"epsilon={epsilon:.5f} | rep={rep} | "
                    f"scaling repeat={repeat_id + 1}/{N_REPEATS} | seed={seed}"
                )
                print("=" * 80)

                point_started_at = utc_now()
                point_start = time.perf_counter()

                signal_df, chunk_df = run_phase_gate_csb(
                    decoder=decoder,
                    noise=noise,
                    steps=CSB_STEPS,
                    rep=rep,
                    shots=CSB_SHOTS,
                    seed=seed,
                    n_jobs=N_JOBS,
                    chunk_shots=CSB_CHUNK_SHOTS,
                    verbose=True,
                )

                fit = fit_phase_gate_spectrum(signal_df)
                point_elapsed = time.perf_counter() - point_start

                save_point(
                    conn,
                    epsilon=epsilon,
                    rep=rep,
                    repeat_id=repeat_id,
                    seed=seed,
                    session_id=session_id,
                    started_at=point_started_at,
                    elapsed_seconds=point_elapsed,
                    signal_df=signal_df,
                    chunk_df=chunk_df,
                    fit=fit,
                )

                completed_this_session += 1
                print(
                    "Saved complete epsilon point | "
                    f"wall time = {point_elapsed / 3600:.3f} h "
                    f"({point_elapsed:.1f} s)"
                )

            # A full repeat is complete only after every epsilon is present.
            count = conn.execute(
                "SELECT COUNT(*) FROM point_results WHERE repeat_id = ?",
                (int(repeat_id),),
            ).fetchone()[0]
            if int(count) == len(PAPER_SCHEDULE):
                repeat_seconds = conn.execute(
                    """
                    SELECT SUM(elapsed_seconds)
                    FROM point_results
                    WHERE repeat_id = ?
                    """,
                    (int(repeat_id),),
                ).fetchone()[0]
                print(
                    f"\nSCALING REPEAT {repeat_id + 1} COMPLETE | "
                    f"sum of epsilon-point wall times = "
                    f"{float(repeat_seconds) / 3600:.3f} h"
                )

        finish_session(
            conn,
            session_id,
            session_start,
            "complete",
            completed_this_session,
        )

    except KeyboardInterrupt:
        finish_session(
            conn,
            session_id,
            session_start,
            "interrupted",
            completed_this_session,
            "KeyboardInterrupt",
        )
        print(
            "\nInterrupted. Every epsilon point already printed as saved is "
            "committed and can be analyzed immediately."
        )
        raise

    except Exception:
        message = traceback.format_exc()
        finish_session(
            conn,
            session_id,
            session_start,
            "failed",
            completed_this_session,
            message,
        )
        raise

    finally:
        print("\nCurrent progress:")
        try:
            print(progress_table(conn).to_string(index=False))
            session = conn.execute(
                """
                SELECT elapsed_seconds
                FROM run_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            if session and session[0] is not None:
                print(
                    f"\nThis script invocation wall time: "
                    f"{float(session[0]) / 3600:.3f} h "
                    f"({float(session[0]):.1f} s)"
                )
        finally:
            conn.close()


if __name__ == "__main__":
    main()
