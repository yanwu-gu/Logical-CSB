#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified logical channel-spectrum benchmarking simulation toolkit.

Merged from the uploaded single-logical-qubit, logical-CX, logical-T, universal
logical-PEC, and coherent state-vector modules. This file contains simulation
and analysis implementations only. All experiment/run settings are supplied by
the calling notebook, and numerical results remain in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import itertools
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import least_squares, linprog
    SCIPY_AVAILABLE = True
except Exception:
    least_squares = None
    linprog = None
    SCIPY_AVAILABLE = False

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except Exception:
    Parallel = None
    delayed = None
    JOBLIB_AVAILABLE = False


# =============================================================================
# Basic utilities
# =============================================================================

def bits_to_int(bits: List[int] | np.ndarray) -> int:
    out = 0
    for i, b in enumerate(bits):
        if int(b) & 1:
            out |= 1 << i
    return int(out)


def pauli_weight(x: np.ndarray, z: np.ndarray) -> int:
    return int(np.count_nonzero((x % 2) | (z % 2)))


def sample_one_qubit_pauli(rng: np.random.Generator, p: float) -> str:
    if rng.random() >= p:
        return "I"
    return rng.choice(np.array(list("XYZ")))


def sample_two_qubit_pauli(rng: np.random.Generator, p: float) -> Tuple[str, str]:
    if rng.random() >= p:
        return "I", "I"
    all_pairs = [(a, b) for a in "IXYZ" for b in "IXYZ" if not (a == "I" and b == "I")]
    return all_pairs[int(rng.integers(len(all_pairs)))]


def apply_pauli_to_frame(x: np.ndarray, z: np.ndarray, q: int, p: str) -> None:
    if p in ("X", "Y"):
        x[q] ^= 1
    if p in ("Z", "Y"):
        z[q] ^= 1



def compose_logical_pauli_into_recovery(rx: np.ndarray, rz: np.ndarray, pauli: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compose a sampled logical Pauli with a physical Pauli-frame recovery.

    This does not insert a physical gate.  It merely changes the Pauli frame.
    For the Steane code we use the all-ones representatives of logical X/Z.
    """
    out_x = rx.copy()
    out_z = rz.copy()
    logical = np.ones(7, dtype=np.uint8)

    if pauli == "I":
        return out_x, out_z
    if pauli in ("X", "Y"):
        out_x ^= logical
    if pauli in ("Z", "Y"):
        out_z ^= logical
    if pauli not in ("I", "X", "Y", "Z"):
        raise ValueError(f"Unknown logical Pauli {pauli}")
    return out_x, out_z

# =============================================================================
# Steane code and verified flagged FTEC decoder
# =============================================================================

class SteaneCode:
    """Steane [[7,1,3]] CSS code with Hamming parity-check matrix."""

    def __init__(self):
        self.H = np.array(
            [
                [0, 0, 0, 1, 1, 1, 1],
                [0, 1, 1, 0, 0, 1, 1],
                [1, 0, 1, 0, 1, 0, 1],
            ],
            dtype=np.uint8,
        )
        self.supports = [[int(q) for q in np.flatnonzero(self.H[i])] for i in range(3)]
        self.col_int = [bits_to_int(self.H[:, q]) for q in range(7)]
        self.qubit_from_syndrome = {s: q for q, s in enumerate(self.col_int)}

        self.rowspace: List[np.ndarray] = []
        for mask in range(8):
            v = np.zeros(7, dtype=np.uint8)
            for r in range(3):
                if (mask >> r) & 1:
                    v ^= self.H[r]
            self.rowspace.append(v)

    def syndrome(self, error_bits: np.ndarray) -> int:
        return bits_to_int((self.H @ (error_bits.astype(np.uint8) % 2)) % 2)

    def ordinary_decode(self, syndrome: int) -> List[int]:
        syndrome = int(syndrome)
        if syndrome == 0:
            return []
        return [self.qubit_from_syndrome[syndrome]]

    def min_weight_mod_stabilizer(self, x: np.ndarray, z: np.ndarray) -> int:
        best = 99
        x = x.astype(np.uint8) % 2
        z = z.astype(np.uint8) % 2
        for sx in self.rowspace:
            for sz in self.rowspace:
                best = min(best, pauli_weight(x ^ sx, z ^ sz))
        return int(best)


@dataclass(frozen=True)
class Op:
    name: str
    args: Tuple[Any, ...] = ()
    meta: Tuple[Any, ...] = ()


@dataclass
class FaultExample:
    key: Any
    x: np.ndarray
    z: np.ndarray
    tag: str
    source: str


class AdaptiveFlagCircuitModel:
    """
    One syndrome ancilla + one flag ancilla adaptive flagged Steane EC template.

    X-check: reset syndrome |+>, flag |0>, flag-CNOTs, data CNOTs, measure syndrome X and flag Z.
    Z-check: reset syndrome |0>, flag |+>, flag-CNOTs, data CNOTs, measure syndrome Z and flag X.
    """

    def __init__(self, min_rounds: int, max_rounds: int):
        self.code = SteaneCode()
        self.min_rounds = int(min_rounds)
        self.max_rounds = int(max_rounds)
        self.s = 7
        self.f = 8
        self.ops: List[Op] = []
        self._build_ops()

    def _add(self, name: str, *args: Any, meta: Tuple[Any, ...] = ()) -> None:
        self.ops.append(Op(name=name, args=tuple(args), meta=tuple(meta)))

    def _build_x_check(self, r: int, check_id: int) -> None:
        support = self.code.supports[check_id]
        self._add("DATA_IDLE", meta=(r, "X_CHECK", check_id, "before"))
        self._add("RESET_X", self.s, meta=(r, "X_CHECK", check_id, "reset_s"))
        self._add("RESET_Z", self.f, meta=(r, "X_CHECK", check_id, "reset_f"))
        self._add("CX", self.s, self.f, meta=(r, "X_CHECK", check_id, "flag_1"))
        for k, q in enumerate(support):
            self._add("CX", self.s, q, meta=(r, "X_CHECK", check_id, f"data_{k}"))
        self._add("CX", self.s, self.f, meta=(r, "X_CHECK", check_id, "flag_2"))
        self._add("MEAS_X", self.s, meta=(r, "X_CHECK", check_id, "syndrome"))
        self._add("MEAS_Z", self.f, meta=(r, "X_CHECK", check_id, "flag"))

    def _build_z_check(self, r: int, check_id: int) -> None:
        support = self.code.supports[check_id]
        self._add("DATA_IDLE", meta=(r, "Z_CHECK", check_id, "before"))
        self._add("RESET_Z", self.s, meta=(r, "Z_CHECK", check_id, "reset_s"))
        self._add("RESET_X", self.f, meta=(r, "Z_CHECK", check_id, "reset_f"))
        self._add("CX", self.f, self.s, meta=(r, "Z_CHECK", check_id, "flag_1"))
        for k, q in enumerate(support):
            self._add("CX", q, self.s, meta=(r, "Z_CHECK", check_id, f"data_{k}"))
        self._add("CX", self.f, self.s, meta=(r, "Z_CHECK", check_id, "flag_2"))
        self._add("MEAS_Z", self.s, meta=(r, "Z_CHECK", check_id, "syndrome"))
        self._add("MEAS_X", self.f, meta=(r, "Z_CHECK", check_id, "flag"))

    def _build_ops(self) -> None:
        for r in range(self.max_rounds):
            for i in range(3):
                self._build_x_check(r, i)
            for i in range(3):
                self._build_z_check(r, i)

    @staticmethod
    def cx_update(x: np.ndarray, z: np.ndarray, control: int, target: int) -> None:
        x[target] ^= x[control]
        z[control] ^= z[target]

    def terminal_reliable(self, x_syn, z_syn, x_flags, z_flags, completed_rounds: int) -> bool:
        if completed_rounds < self.min_rounds or completed_rounds < 2:
            return False
        last = completed_rounds - 1
        prev = completed_rounds - 2
        if x_flags[last] != 0 or z_flags[last] != 0:
            return False
        return x_syn[last] == x_syn[prev] and z_syn[last] == z_syn[prev]

    def run_deterministic_fault(self, initial_x=None, initial_z=None, fault=None):
        """Used only for decoder construction/verification by single-fault enumeration."""
        x = np.zeros(9, dtype=np.uint8)
        z = np.zeros(9, dtype=np.uint8)
        if initial_x is not None:
            x[:7] ^= initial_x.astype(np.uint8)
        if initial_z is not None:
            z[:7] ^= initial_z.astype(np.uint8)

        x_syn = [0 for _ in range(self.max_rounds)]
        z_syn = [0 for _ in range(self.max_rounds)]
        x_flags = [0 for _ in range(self.max_rounds)]
        z_flags = [0 for _ in range(self.max_rounds)]
        fault_executed = fault is None
        completed = 0

        def maybe_insert_fault(idx: int) -> None:
            nonlocal fault_executed
            if fault is None or fault[0] != idx:
                return
            fault_executed = True
            if fault[1] == "P":
                spec = fault[2]
                if isinstance(spec[0], tuple):
                    for q, p in spec:
                        if p != "I":
                            apply_pauli_to_frame(x, z, int(q), str(p))
                else:
                    q, p = spec
                    apply_pauli_to_frame(x, z, int(q), str(p))

        for idx, op in enumerate(self.ops):
            if op.name in ("RESET_X", "RESET_Z"):
                q = int(op.args[0]); x[q] = 0; z[q] = 0
            elif op.name == "CX":
                c, t = map(int, op.args); self.cx_update(x, z, c, t)
            elif op.name == "MEAS_Z":
                q = int(op.args[0]); outcome = int(x[q])
                if fault is not None and fault[0] == idx and fault[1] == "M":
                    outcome ^= 1; fault_executed = True
                r, basis, check, kind = op.meta
                if basis == "Z_CHECK" and kind == "syndrome": z_syn[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag": z_flags[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag": x_flags[int(r)] ^= outcome << int(check)
                x[q] = 0; z[q] = 0
            elif op.name == "MEAS_X":
                q = int(op.args[0]); outcome = int(z[q])
                if fault is not None and fault[0] == idx and fault[1] == "M":
                    outcome ^= 1; fault_executed = True
                r, basis, check, kind = op.meta
                if basis == "X_CHECK" and kind == "syndrome": x_syn[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag": x_flags[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag": z_flags[int(r)] ^= outcome << int(check)
                x[q] = 0; z[q] = 0
            elif op.name == "DATA_IDLE":
                pass
            else:
                raise ValueError(op.name)

            maybe_insert_fault(idx)

            if op.name == "MEAS_X" and op.meta[1] == "Z_CHECK" and op.meta[2] == 2 and op.meta[3] == "flag":
                completed = int(op.meta[0]) + 1
                if self.terminal_reliable(x_syn, z_syn, x_flags, z_flags, completed):
                    if fault is not None and fault[0] > idx:
                        fault_executed = False
                    break

        if completed == 0:
            completed = self.max_rounds
        key = (tuple(x_syn[:completed]), tuple(z_syn[:completed]), tuple(x_flags[:completed]), tuple(z_flags[:completed]))
        return key, x[:7].copy(), z[:7].copy(), fault_executed, completed

    def fault_locations(self):
        locs = []
        for idx, op in enumerate(self.ops):
            if op.name in ("RESET_X", "RESET_Z"):
                q = int(op.args[0])
                for p in "XYZ": locs.append((idx, "P", (q, p), op))
            elif op.name == "CX":
                c, t = map(int, op.args)
                for p1 in "IXYZ":
                    for p2 in "IXYZ":
                        if p1 == "I" and p2 == "I": continue
                        locs.append((idx, "P", ((c, p1), (t, p2)), op))
            elif op.name in ("MEAS_X", "MEAS_Z"):
                locs.append((idx, "M", None, op))
            elif op.name == "DATA_IDLE":
                for q in range(7):
                    for p in "XYZ": locs.append((idx, "P", (q, p), op))
        return locs


class VerifiedSteaneFlagDecoder:
    def __init__(self, min_rounds: int, max_rounds: int, build: bool, verbose: bool):
        self.code = SteaneCode()
        self.model = AdaptiveFlagCircuitModel(min_rounds=min_rounds, max_rounds=max_rounds)
        self.min_rounds = int(min_rounds)
        self.max_rounds = int(max_rounds)
        self.verbose = bool(verbose)
        self.examples: List[FaultExample] = []
        self.bins: Dict[Any, List[FaultExample]] = {}
        self.table: Dict[Any, Tuple[np.ndarray, np.ndarray]] = {}
        if build:
            self.build_and_verify()

    def add_example(self, key, x, z, tag, source):
        ex = FaultExample(key, x.astype(np.uint8).copy(), z.astype(np.uint8).copy(), tag, source)
        self.examples.append(ex)
        self.bins.setdefault(key, []).append(ex)

    def collect_examples(self):
        self.examples.clear(); self.bins.clear()
        key, x, z, _, _ = self.model.run_deterministic_fault()
        self.add_example(key, x, z, "clean", "no_fault")
        for q in range(7):
            for p in "XYZ":
                ix = np.zeros(7, dtype=np.uint8); iz = np.zeros(7, dtype=np.uint8)
                if p in ("X", "Y"): ix[q] = 1
                if p in ("Z", "Y"): iz[q] = 1
                key, x, z, _, _ = self.model.run_deterministic_fault(initial_x=ix, initial_z=iz)
                self.add_example(key, x, z, "initial", f"initial_{p}{q}")
        for idx, kind, spec, op in self.model.fault_locations():
            key, x, z, executed, rounds = self.model.run_deterministic_fault(fault=(idx, kind, spec))
            if not executed:
                continue
            self.add_example(key, x, z, "fault", f"op{idx}_{kind}_{spec}_{op.name}_{op.meta}_rounds{rounds}")

    def candidate_recoveries(self, examples):
        out = []; seen = set()
        def add(rx, rz):
            k = (tuple(map(int, rx)), tuple(map(int, rz)))
            if k not in seen:
                seen.add(k); out.append((rx.astype(np.uint8).copy(), rz.astype(np.uint8).copy()))
        zero = np.zeros(7, dtype=np.uint8); add(zero, zero)
        for q in range(7):
            rx = np.zeros(7, dtype=np.uint8); rx[q] = 1; add(rx, zero)
            rz = np.zeros(7, dtype=np.uint8); rz[q] = 1; add(zero, rz)
            rx = np.zeros(7, dtype=np.uint8); rz = np.zeros(7, dtype=np.uint8); rx[q] = rz[q] = 1; add(rx, rz)
        for ex in examples:
            add(ex.x, ex.z)
            rx = np.zeros(7, dtype=np.uint8); rz = np.zeros(7, dtype=np.uint8)
            for q in self.code.ordinary_decode(self.code.syndrome(ex.x)): rx[q] = 1
            for q in self.code.ordinary_decode(self.code.syndrome(ex.z)): rz[q] = 1
            add(rx, rz)
        for support in self.code.supports:
            for k in range(len(support)+1):
                v = np.zeros(7, dtype=np.uint8)
                for q in support[k:]: v[q] = 1
                add(v, zero); add(zero, v); add(v, v)
        return out

    def candidate_is_valid(self, rx, rz, examples):
        for ex in examples:
            min_w = self.code.min_weight_mod_stabilizer(ex.x ^ rx, ex.z ^ rz)
            if ex.tag in ("clean", "initial"):
                if min_w != 0: return False
            else:
                if min_w > 1: return False
        return True

    @staticmethod
    def recovery_score(rx, rz):
        return 10 * pauli_weight(rx, rz) + int(np.sum(rx) + np.sum(rz))

    def build_table(self):
        unresolved = []
        for key, examples in self.bins.items():
            best = None; best_score = 10**9
            for rx, rz in self.candidate_recoveries(examples):
                if self.candidate_is_valid(rx, rz, examples):
                    score = self.recovery_score(rx, rz)
                    if score < best_score:
                        best = (rx.copy(), rz.copy()); best_score = score
            if best is None:
                unresolved.append((key, examples))
            else:
                self.table[key] = best
        return unresolved

    def verify_table(self):
        bad = []
        for ex in self.examples:
            if ex.key not in self.table:
                bad.append(ex); continue
            rx, rz = self.table[ex.key]
            min_w = self.code.min_weight_mod_stabilizer(ex.x ^ rx, ex.z ^ rz)
            if ex.tag in ("clean", "initial"):
                if min_w != 0: bad.append(ex)
            else:
                if min_w > 1: bad.append(ex)
        return bad

    def build_and_verify(self):
        self.collect_examples()
        unresolved = self.build_table()
        if unresolved:
            raise RuntimeError(f"Decoder construction failed: {len(unresolved)} unresolved bins")
        bad = self.verify_table()
        if bad:
            raise RuntimeError(f"Decoder verification failed: {len(bad)} bad examples")
        if self.verbose:
            print("Verified flagged Steane EC decoder.")
            print("  examples checked =", len(self.examples))
            print("  decoder bins     =", len(self.table))

    def save(self, filename):
        payload = {"min_rounds": self.min_rounds, "max_rounds": self.max_rounds, "table": self.table,
                   "num_examples": len(self.examples), "num_bins": len(self.table)}
        with open(filename, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filename, verbose: bool):
        with open(filename, "rb") as f:
            payload = pickle.load(f)
        obj = cls(payload["min_rounds"], payload["max_rounds"], build=False, verbose=verbose)
        obj.table = payload["table"]
        if verbose:
            print("Loaded verified flagged Steane EC decoder.")
            print("  decoder bins     =", len(obj.table))
        return obj

    def decode(self, key):
        return self.table[key]


def get_or_build_decoder(
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> VerifiedSteaneFlagDecoder:
    """Load or construct the verified flagged-Steane decoder."""
    path = Path(cache_file)
    if path.exists():
        return VerifiedSteaneFlagDecoder.load(path, verbose=verbose)
    dec = VerifiedSteaneFlagDecoder(
        min_rounds=min_rounds,
        max_rounds=max_rounds,
        build=True,
        verbose=verbose,
    )
    dec.save(path)
    return dec


# Per-worker decoder cache. The cache path and EC-round settings are supplied
# by the notebook and are part of the cache key.
_WORKER_DECODERS: Dict[Tuple[str, int, int], VerifiedSteaneFlagDecoder] = {}


def get_worker_decoder(
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
) -> VerifiedSteaneFlagDecoder:
    key = (str(cache_file), int(min_rounds), int(max_rounds))
    if key not in _WORKER_DECODERS:
        _WORKER_DECODERS[key] = get_or_build_decoder(
            cache_file=cache_file,
            min_rounds=min_rounds,
            max_rounds=max_rounds,
            verbose=False,
        )
    return _WORKER_DECODERS[key]


# =============================================================================
# Stochastic Pauli stabilizer simulator
# =============================================================================

@dataclass(frozen=True)
class NoiseParams:
    p_1q: float
    p_2q: float
    p_meas: float
    p_reset: float
    p_idle: float
    p_final_meas: float


def make_homogeneous_noise(
    strength: float,
    two_qubit_factor: float,
    final_meas_factor: float,
) -> NoiseParams:
    """Construct the circuit-level stochastic Pauli noise model."""
    strength = float(strength)
    return NoiseParams(
        p_1q=strength,
        p_2q=float(two_qubit_factor) * strength,
        p_meas=strength,
        p_reset=strength,
        p_idle=strength,
        p_final_meas=float(final_meas_factor) * strength,
    )


def validate_circuit_options(options: Mapping[str, bool]) -> Dict[str, bool]:
    """Validate circuit-boundary choices supplied by the calling notebook."""
    required = (
        "prep_gates_noisy",
        "prep_qec_after_gate",
        "measurement_basis_gates_noisy",
        "qec_after_each_logical_gate",
        "fallback_on_unverified",
    )
    missing = [name for name in required if name not in options]
    if missing:
        raise ValueError(f"Missing circuit options: {missing}")
    return {name: bool(options[name]) for name in required}



class StabilizerSteaneFTSimulator:
    """Pauli-frame trajectory simulator for one Steane logical block."""

    def __init__(
        self,
        decoder: VerifiedSteaneFlagDecoder,
        noise: NoiseParams,
        options: Mapping[str, bool],
        seed: Optional[int] = None,
    ):
        self.code = SteaneCode()
        self.decoder = decoder
        self.model = decoder.model
        self.noise = noise
        self.options = validate_circuit_options(options)
        self.rng = np.random.default_rng(seed)
        self.x = np.zeros(7, dtype=np.uint8)
        self.z = np.zeros(7, dtype=np.uint8)
        self.unverified_count = 0
        self.ec_cycles = 0
        self.ec_rounds = 0

    def _apply_h_data(self):
        self.x, self.z = self.z.copy(), self.x.copy()

    def _apply_s_data(self):
        self.z ^= self.x

    def _apply_rx90_data(self):
        self.x ^= self.z

    def _apply_ry90_data(self):
        self.x, self.z = self.z.copy(), self.x.copy()

    def _apply_1q_gate_noise_all_data(self):
        for q in range(7):
            p = sample_one_qubit_pauli(self.rng, self.noise.p_1q)
            if p != "I":
                apply_pauli_to_frame(self.x, self.z, q, p)

    def _apply_1q_gate_noise_to_temp_frame(self, tx: np.ndarray, tz: np.ndarray):
        for q in range(7):
            p = sample_one_qubit_pauli(self.rng, self.noise.p_1q)
            if p != "I":
                apply_pauli_to_frame(tx, tz, q, p)

    def apply_logical_gate(
        self,
        gate: str,
        noisy: bool,
        apply_qec: bool,
        post_ec_logical_pauli: str = "I",
    ) -> None:
        if gate == "RZ90":
            self._apply_s_data()
        elif gate == "RX90":
            self._apply_rx90_data()
        elif gate == "RY90":
            self._apply_ry90_data()
        elif gate == "H":
            self._apply_h_data()
        elif gate in ("X", "Z", "Sdg"):
            if gate == "Sdg":
                self._apply_s_data()
        else:
            raise ValueError(f"Unknown logical gate {gate}")

        if noisy:
            self._apply_1q_gate_noise_all_data()

        if apply_qec:
            self.run_ft_ec(extra_logical_pauli=post_ec_logical_pauli)
        else:
            self.apply_logical_pauli(post_ec_logical_pauli)

    def prepare_initial(self, label: str) -> None:
        """Prepare a CSB input from an ideal encoded |0_L> reference state."""
        self.x[:] = 0
        self.z[:] = 0
        self.unverified_count = 0
        self.ec_cycles = 0
        self.ec_rounds = 0

        prep_gates = {
            "0": [],
            "1": ["X"],
            "+": ["H"],
            "-": ["H", "Z"],
            "+i": ["H", "RZ90"],
            "-i": ["H", "Sdg"],
        }[label]
        for gate in prep_gates:
            self.apply_logical_gate(
                gate,
                noisy=self.options["prep_gates_noisy"],
                apply_qec=self.options["prep_qec_after_gate"],
            )

    def run_ft_ec(self, extra_logical_pauli: str = "I") -> Dict[str, Any]:
        x = np.zeros(9, dtype=np.uint8)
        z = np.zeros(9, dtype=np.uint8)
        x[:7] = self.x
        z[:7] = self.z
        x_syn = [0 for _ in range(self.model.max_rounds)]
        z_syn = [0 for _ in range(self.model.max_rounds)]
        x_flags = [0 for _ in range(self.model.max_rounds)]
        z_flags = [0 for _ in range(self.model.max_rounds)]
        completed = 0

        for op in self.model.ops:
            if op.name == "DATA_IDLE":
                for q in range(7):
                    p = sample_one_qubit_pauli(self.rng, self.noise.p_idle)
                    if p != "I":
                        apply_pauli_to_frame(x, z, q, p)
            elif op.name in ("RESET_X", "RESET_Z"):
                q = int(op.args[0])
                x[q] = 0
                z[q] = 0
                p = sample_one_qubit_pauli(self.rng, self.noise.p_reset)
                if p != "I":
                    apply_pauli_to_frame(x, z, q, p)
            elif op.name == "CX":
                c, t = map(int, op.args)
                AdaptiveFlagCircuitModel.cx_update(x, z, c, t)
                p1, p2 = sample_two_qubit_pauli(self.rng, self.noise.p_2q)
                if p1 != "I":
                    apply_pauli_to_frame(x, z, c, p1)
                if p2 != "I":
                    apply_pauli_to_frame(x, z, t, p2)
            elif op.name == "MEAS_Z":
                q = int(op.args[0])
                outcome = int(x[q])
                if self.rng.random() < self.noise.p_meas:
                    outcome ^= 1
                r, basis, check, kind = op.meta
                if basis == "Z_CHECK" and kind == "syndrome":
                    z_syn[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag":
                    z_flags[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag":
                    x_flags[int(r)] ^= outcome << int(check)
                x[q] = 0
                z[q] = 0
            elif op.name == "MEAS_X":
                q = int(op.args[0])
                outcome = int(z[q])
                if self.rng.random() < self.noise.p_meas:
                    outcome ^= 1
                r, basis, check, kind = op.meta
                if basis == "X_CHECK" and kind == "syndrome":
                    x_syn[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag":
                    x_flags[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag":
                    z_flags[int(r)] ^= outcome << int(check)
                x[q] = 0
                z[q] = 0
            else:
                raise ValueError(op.name)

            if (
                op.name == "MEAS_X"
                and op.meta[1] == "Z_CHECK"
                and op.meta[2] == 2
                and op.meta[3] == "flag"
            ):
                completed = int(op.meta[0]) + 1
                if self.model.terminal_reliable(x_syn, z_syn, x_flags, z_flags, completed):
                    break

        if completed == 0:
            completed = self.model.max_rounds

        key = (
            tuple(x_syn[:completed]),
            tuple(z_syn[:completed]),
            tuple(x_flags[:completed]),
            tuple(z_flags[:completed]),
        )
        unverified = False
        try:
            rx, rz = self.decoder.decode(key)
        except KeyError:
            if not self.options["fallback_on_unverified"]:
                raise
            unverified = True
            rx = np.zeros(7, dtype=np.uint8)
            rz = np.zeros(7, dtype=np.uint8)
            for q in self.code.ordinary_decode(z_syn[completed - 1]):
                rx[q] = 1
            for q in self.code.ordinary_decode(x_syn[completed - 1]):
                rz[q] = 1

        rx_total, rz_total = compose_logical_pauli_into_recovery(rx, rz, extra_logical_pauli)
        x[:7] ^= rx_total
        z[:7] ^= rz_total
        self.x = x[:7].copy()
        self.z = z[:7].copy()
        self.ec_cycles += 1
        self.ec_rounds += completed
        self.unverified_count += int(unverified)
        return {"rounds": completed, "unverified": unverified}

    def apply_logical_pauli(self, pauli: str) -> None:
        logical = np.ones(7, dtype=np.uint8)
        if pauli == "I":
            return
        if pauli in ("X", "Y"):
            self.x ^= logical
        if pauli in ("Z", "Y"):
            self.z ^= logical
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(pauli)

    def logical_measurement_flip(self, pauli: str) -> int:
        tx = self.x.copy()
        tz = self.z.copy()
        if pauli == "X":
            tx, tz = tz.copy(), tx.copy()
            if self.options["measurement_basis_gates_noisy"]:
                self._apply_1q_gate_noise_to_temp_frame(tx, tz)
        elif pauli == "Y":
            tz ^= tx
            if self.options["measurement_basis_gates_noisy"]:
                self._apply_1q_gate_noise_to_temp_frame(tx, tz)
            tx, tz = tz.copy(), tx.copy()
            if self.options["measurement_basis_gates_noisy"]:
                self._apply_1q_gate_noise_to_temp_frame(tx, tz)
        elif pauli != "Z":
            raise ValueError(pauli)

        bits = tx.copy()
        for q in range(7):
            if self.rng.random() < self.noise.p_final_meas:
                bits[q] ^= 1
        syndrome = self.code.syndrome(bits)
        if syndrome != 0:
            bits[self.code.qubit_from_syndrome[syndrome]] ^= 1
        return int(np.sum(bits) % 2)

    def measure_logical_pauli(self, pauli: str, ideal_bloch: np.ndarray) -> int:
        idx = {"X": 0, "Y": 1, "Z": 2}[pauli]
        mean = float(ideal_bloch[idx])
        p_plus = 0.5 * (1.0 + mean)
        ideal_outcome = +1 if self.rng.random() < p_plus else -1
        flip = self.logical_measurement_flip(pauli)
        return ideal_outcome * ((-1) ** flip)


# =============================================================================
# Logical Bloch-vector updates and CSB branches
# =============================================================================

def initial_bloch(label: str) -> np.ndarray:
    table = {
        "0": np.array([0, 0, 1], dtype=int),
        "1": np.array([0, 0, -1], dtype=int),
        "+": np.array([1, 0, 0], dtype=int),
        "-": np.array([-1, 0, 0], dtype=int),
        "+i": np.array([0, 1, 0], dtype=int),
        "-i": np.array([0, -1, 0], dtype=int),
    }
    return table[label].copy()


def update_bloch(b: np.ndarray, gate: str) -> np.ndarray:
    x, y, z = map(int, b)
    if gate == "RZ90":
        return np.array([-y, x, z], dtype=int)
    if gate == "RX90":
        return np.array([x, -z, y], dtype=int)
    if gate == "RY90":
        return np.array([z, y, -x], dtype=int)
    if gate == "H":
        return np.array([z, -y, x], dtype=int)
    if gate == "X":
        return np.array([x, -y, -z], dtype=int)
    if gate == "Z":
        return np.array([-x, -y, z], dtype=int)
    if gate == "Sdg":
        return np.array([y, -x, z], dtype=int)
    raise ValueError(gate)


def get_csb_branches(target_gate: str):
    """
    CSB branches for enhanced single-qubit quarter-turn spectroscopy.

    Off-diagonal eigen-operator spectrum:
      We use two quadratures with the same default CSB initial state
      (|phi_+>+|phi_->)/sqrt(2).  The total requested CSB shots are split
      equally between eq_real and eq_imag.

      RZ90: prepare |+>, measure X and Y
      RX90: prepare |0>, measure Z and -Y
      RY90: prepare |0>, measure Z and X

      The `measure_sign` field is multiplied into the measured expectation so
      that the complex signal C_L = real_L + i imag_L has ideal phase i^L.

    Axis / projector-degenerate subspace spectrum:
      A single axis eigenstate is enough.  We fit the decay of
          <A>_L = f_A^L
      using one branch only.
    """
    if target_gate == "RZ90":
        return {
            "eq_real": {"initial": "+", "measure": "X", "measure_sign": +1, "kind": "equatorial_real"},
            "eq_imag": {"initial": "+", "measure": "Y", "measure_sign": +1, "kind": "equatorial_imag"},
            "axis": {"initial": "0", "measure": "Z", "measure_sign": +1, "kind": "axis"},
        }
    if target_gate == "RX90":
        return {
            # RX eigenstates are |+>, |->; no-relative-phase CSB state is |0>.
            "eq_real": {"initial": "0", "measure": "Z", "measure_sign": +1, "kind": "equatorial_real"},
            "eq_imag": {"initial": "0", "measure": "Y", "measure_sign": -1, "kind": "equatorial_imag"},
            "axis": {"initial": "+", "measure": "X", "measure_sign": +1, "kind": "axis"},
        }
    if target_gate == "RY90":
        return {
            # RY eigenstates are |+i>, |-i>; no-relative-phase CSB state is |0>.
            "eq_real": {"initial": "0", "measure": "Z", "measure_sign": +1, "kind": "equatorial_real"},
            "eq_imag": {"initial": "0", "measure": "X", "measure_sign": +1, "kind": "equatorial_imag"},
            "axis": {"initial": "+i", "measure": "Y", "measure_sign": +1, "kind": "axis"},
        }
    raise ValueError(target_gate)


def _simulate_csb_point_reference(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    target_gate: str,
    branch_name: str,
    depth: int,
    shots: int,
    seed: int,
):
    branch = get_csb_branches(target_gate)[branch_name]
    rng = np.random.default_rng(seed)
    outcomes = []
    unverified = 0
    ec_cycles = 0
    for _ in range(int(shots)):
        sim = StabilizerSteaneFTSimulator(
            decoder, noise, options, seed=int(rng.integers(2**32 - 1))
        )
        sim.prepare_initial(branch["initial"])
        b = initial_bloch(branch["initial"])
        for _g in range(int(depth)):
            sim.apply_logical_gate(
                target_gate,
                noisy=True,
                apply_qec=options["qec_after_each_logical_gate"],
            )
            b = update_bloch(b, target_gate)
        outcomes.append(
            branch.get("measure_sign", +1)
            * sim.measure_logical_pauli(branch["measure"], b)
        )
        unverified += sim.unverified_count
        ec_cycles += sim.ec_cycles
    mean = float(np.mean(outcomes)) if shots > 0 else np.nan
    stderr = (
        float(np.std(outcomes, ddof=1) / math.sqrt(shots)) if shots > 1 else 0.0
    )
    return mean, stderr, unverified / max(ec_cycles, 1)


def make_csb_depth_table(steps: Sequence[int], rep: int) -> pd.DataFrame:
    return pd.DataFrame({
        "step": [int(s) for s in steps],
        "rep": [int(rep) for _ in steps],
        "true_depth_L": [int(rep) * int(s) for s in steps],
    })


def _run_csb_spectrum_reference(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    target_gates: Sequence[str],
    verbose: bool,
) -> pd.DataFrame:
    rep = int(rep)
    steps = [int(s) for s in steps]
    rows = []
    rng = np.random.default_rng(seed)
    for target_gate in target_gates:
        for branch_name, branch_spec in get_csb_branches(target_gate).items():
            branch_shots = (
                max(1, int(shots) // 2)
                if branch_spec["kind"].startswith("equatorial")
                else int(shots)
            )
            for step in steps:
                depth = rep * step
                mean, se, unv = _simulate_csb_point_reference(
                    decoder,
                    noise,
                    options,
                    target_gate,
                    branch_name,
                    depth,
                    branch_shots,
                    int(rng.integers(2**32 - 1)),
                )
                rows.append({
                    "target_gate": target_gate,
                    "branch": branch_name,
                    "branch_kind": branch_spec["kind"],
                    "initial": branch_spec["initial"],
                    "measure": branch_spec["measure"],
                    "measure_sign": int(branch_spec.get("measure_sign", +1)),
                    "step": int(step),
                    "rep": rep,
                    "depth": int(depth),
                    "mean": mean,
                    "stderr": se,
                    "shots": branch_shots,
                    "requested_total_shots": int(shots),
                    "unverified_rate": unv,
                })
                if verbose:
                    print(
                        f"CSB {target_gate:4s} {branch_name:8s} "
                        f"step {step:3d} rep {rep:3d} depth {depth:5d} "
                        f"shots {branch_shots:6d} mean {mean:+.5f} stderr {se:.5f}"
                    )
    return pd.DataFrame(rows)


def estimate_exponential_from_signal(depths, signal, use_even_only=False):
    """
    Estimate the per-cycle spectral factor g from signal(L) ~ g^L.

    `depths` must be the true number of repeated logical cycles L, not merely
    the plotted step.  `run_csb_spectrum` already stores true L in the `depth`
    column even when a repetition factor rep is used.
    """
    depths = np.asarray(depths, dtype=float)
    signal = np.asarray(signal, dtype=float)
    if use_even_only:
        cos_vals = np.cos(depths * np.pi / 2)
        mask = np.isfinite(signal) & (depths > 0) & (np.abs(cos_vals) > 0.5) & (signal * cos_vals > 0)
        y = signal[mask] / cos_vals[mask]
        x = depths[mask]
    else:
        mask = np.isfinite(signal) & (depths > 0) & (signal > 0)
        x = depths[mask]
        y = signal[mask]
    if len(x) < 2:
        return np.nan
    slope, intercept = np.polyfit(x, np.log(np.maximum(y, 1e-300)), 1)
    return float(np.exp(slope))


def simplex_project(v: np.ndarray) -> np.ndarray:
    """Euclidean projection of a real vector onto the probability simplex."""
    v = np.asarray(v, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho_candidates = u * np.arange(1, n + 1) > (cssv - 1)
    if not np.any(rho_candidates):
        return np.ones(n) / n
    rho = np.nonzero(rho_candidates)[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1)
    return np.maximum(v - theta, 0.0)



def merge_enhanced_quadratures(csb_df: pd.DataFrame, gate: str) -> pd.DataFrame:
    """
    Merge eq_real and eq_imag rows into a complex off-diagonal CSB signal.

    Returns rows with:
      complex_signal = real_mean + i imag_mean
      magnitude      = |complex_signal|
      magnitude_stderr propagated from the two quadrature standard errors.
    """
    sub = csb_df[csb_df.target_gate == gate]
    real = sub[sub.branch == "eq_real"].sort_values("depth")
    imag = sub[sub.branch == "eq_imag"].sort_values("depth")

    cols = ["step", "rep", "depth"]
    m = real[cols + ["mean", "stderr", "shots"]].merge(
        imag[cols + ["mean", "stderr", "shots"]],
        on=cols,
        suffixes=("_real", "_imag"),
        how="inner",
    )

    xr = m["mean_real"].to_numpy(dtype=float)
    yi = m["mean_imag"].to_numpy(dtype=float)
    sr = m["stderr_real"].to_numpy(dtype=float)
    si = m["stderr_imag"].to_numpy(dtype=float)

    mag = np.sqrt(xr * xr + yi * yi)
    mag_safe = np.maximum(mag, 1e-15)
    mag_stderr = np.sqrt((xr / mag_safe) ** 2 * sr ** 2 + (yi / mag_safe) ** 2 * si ** 2)

    m["complex_signal"] = xr + 1j * yi
    m["magnitude"] = mag
    m["magnitude_stderr"] = mag_stderr

    # Demodulated real component should be approximately +g^L for a pure
    # Pauli-noise channel and correct quadrature convention.
    L = m["depth"].to_numpy(dtype=float)
    omega = np.pi / 2
    demod = (xr + 1j * yi) * np.exp(-1j * omega * L)
    m["demod_real"] = np.real(demod)
    m["demod_imag"] = np.imag(demod)
    return m


def estimate_exponential_from_complex_quadratures(depths, real_signal, imag_signal):
    """
    Estimate g from enhanced CSB complex signal C_L = real_L + i imag_L.

    We use |C_L| ~ g^L.  This avoids losing half the depths at phase nodes and
    avoids aliasing the +/-i eigenvalues by using an even repetition block.
    """
    depths = np.asarray(depths, dtype=float)
    real_signal = np.asarray(real_signal, dtype=float)
    imag_signal = np.asarray(imag_signal, dtype=float)
    mag = np.sqrt(real_signal ** 2 + imag_signal ** 2)

    mask = np.isfinite(mag) & (depths > 0) & (mag > 0)
    x = depths[mask]
    y = mag[mask]
    if len(x) < 2:
        return np.nan
    slope, intercept = np.polyfit(x, np.log(np.maximum(y, 1e-300)), 1)
    return float(np.exp(slope))



def fit_complex_csb_eigenvalue(qdf: pd.DataFrame, omega0: float = np.pi / 2) -> Dict[str, Any]:
    """
    Fit the enhanced CSB complex off-diagonal signal

        C_L = B * lambda^L,
        lambda = g * exp(i * (omega0 + delta)).

    Parameters
    ----------
    qdf:
        Output of `merge_enhanced_quadratures`.
    omega0:
        Ideal eigenphase per logical cycle.  For R_A(pi/2), omega0=pi/2
        under the quadrature conventions used here.

    Returns
    -------
    A dict containing the complex eigenvalue, magnitude g, phase theta,
    coherent phase error delta, and local-fit stderr estimates when available.

    The Pauli-noise learning equations use the magnitude g.  The process
    fidelity uses the coherent angle through cos(delta).
    """
    L = qdf["depth"].to_numpy(dtype=float)
    x = qdf["mean_real"].to_numpy(dtype=float)
    y = qdf["mean_imag"].to_numpy(dtype=float)
    sx = qdf["stderr_real"].to_numpy(dtype=float)
    sy = qdf["stderr_imag"].to_numpy(dtype=float)

    sx = np.maximum(np.where(np.isfinite(sx), sx, 1.0), 1e-12)
    sy = np.maximum(np.where(np.isfinite(sy), sy, 1.0), 1e-12)

    z = x + 1j * y
    mask = (
        np.isfinite(L)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(sx)
        & np.isfinite(sy)
    )

    L = L[mask]
    x = x[mask]
    y = y[mask]
    sx = sx[mask]
    sy = sy[mask]
    z = z[mask]

    # If scipy is unavailable, fall back to magnitude-only decay and zero angle error.
    if (not SCIPY_AVAILABLE) or len(L) < 3:
        g = estimate_exponential_from_complex_quadratures(L, x, y)
        theta = float(omega0)
        delta = 0.0
        lam = g * np.exp(1j * theta)
        return {
            "lambda": lam,
            "lambda_real": float(np.real(lam)),
            "lambda_imag": float(np.imag(lam)),
            "g": float(g),
            "g_stderr": np.nan,
            "theta": theta,
            "delta_theta": delta,
            "delta_theta_stderr": np.nan,
            "B": np.nan + 1j * np.nan,
            "success": False,
            "cost": np.nan,
        }

    # Initial B from L=0 if available; otherwise first observed point demodulated.
    if np.any(L == 0):
        B0 = z[L == 0][0]
    else:
        B0 = z[0] * np.exp(-1j * omega0 * L[0])

    # Initial log_g from magnitude fit.
    mag = np.abs(z)
    mag_mask = (L > 0) & np.isfinite(mag) & (mag > 1e-10)
    if np.sum(mag_mask) >= 2:
        log_g0, _ = np.polyfit(L[mag_mask], np.log(np.maximum(mag[mag_mask], 1e-300)), 1)
    else:
        log_g0 = -1e-3

    # Initial delta from demodulated phase slope.
    z_demod = z * np.exp(-1j * omega0 * L)
    phase = np.unwrap(np.angle(z_demod))
    phase_mask = (L > 0) & (np.abs(z_demod) > 1e-10) & np.isfinite(phase)
    if np.sum(phase_mask) >= 2:
        delta0, phi0 = np.polyfit(L[phase_mask], phase[phase_mask], 1)
    else:
        delta0 = 0.0

    def residual(params):
        B_re, B_im, log_g, delta = params
        B = B_re + 1j * B_im
        lam = np.exp(log_g) * np.exp(1j * (omega0 + delta))
        pred = B * (lam ** L)
        return np.concatenate([
            (pred.real - x) / sx,
            (pred.imag - y) / sy,
        ])

    p0 = np.array([np.real(B0), np.imag(B0), log_g0, delta0], dtype=float)
    lower = np.array([-2.0, -2.0, -1.0, -np.pi], dtype=float)
    upper = np.array([+2.0, +2.0, +0.05, +np.pi], dtype=float)
    p0 = np.minimum(np.maximum(p0, lower + 1e-9), upper - 1e-9)

    fit = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        max_nfev=50000,
    )

    B_re, B_im, log_g, delta = fit.x
    g = float(np.exp(log_g))
    theta = float(omega0 + delta)
    lam = g * np.exp(1j * theta)

    # Local covariance approximation from the weighted residual Jacobian.
    try:
        J = fit.jac
        dof = max(1, len(fit.fun) - len(fit.x))
        reduced_chi2 = float(np.sum(fit.fun ** 2) / dof)
        cov = np.linalg.pinv(J.T @ J) * reduced_chi2
        log_g_stderr = float(np.sqrt(max(cov[2, 2], 0.0)))
        delta_stderr = float(np.sqrt(max(cov[3, 3], 0.0)))
        g_stderr = float(g * log_g_stderr)
    except Exception:
        cov = None
        log_g_stderr = np.nan
        delta_stderr = np.nan
        g_stderr = np.nan

    return {
        "lambda": lam,
        "lambda_real": float(np.real(lam)),
        "lambda_imag": float(np.imag(lam)),
        "g": float(g),
        "g_stderr": g_stderr,
        "theta": theta,
        "delta_theta": float(delta),
        "delta_theta_stderr": delta_stderr,
        "B": B_re + 1j * B_im,
        "success": bool(fit.success),
        "cost": float(fit.cost),
        "log_g": float(log_g),
        "log_g_stderr": log_g_stderr,
        "cov": cov,
    }


def process_fidelity_from_complex_offdiag(g_offdiag: float, delta_theta: float, g_axis: float) -> Dict[str, float]:
    """
    Single-qubit process/average fidelity from the three spectral modes.

    Relative to the ideal R_A(pi/2) channel, the two off-diagonal modes
    contribute 2*g*cos(delta), while the axis mode contributes g_axis:

        F_process = (1 + g_axis + 2*g*cos(delta)) / 4.

    The average gate fidelity is (2 F_process + 1)/3.
    """
    Fp = float((1.0 + g_axis + 2.0 * g_offdiag * math.cos(delta_theta)) / 4.0)
    Favg = float((2.0 * Fp + 1.0) / 3.0)
    return {
        "process_fidelity": Fp,
        "process_infidelity": 1.0 - Fp,
        "average_gate_fidelity": Favg,
        "average_gate_infidelity": 1.0 - Favg,
    }


def estimate_axis_decay_with_offset(depths, signal, stderr=None, r_grid_size: int = 2001):
    """
    Estimate f from an axis / projector-degenerate signal with a possible
    fixed-point component.

    Model:
        y_L = c + a f^L

    This covers both common readout conventions:
      - signed Pauli expectation: c≈0, a≈1, y_L≈f^L
      - projector/success probability: c≈1/2, a≈1/2, y_L≈1/2+f^L/2

    For each candidate f, c and a are obtained by weighted linear least squares.
    We return the f that minimizes the weighted residual.
    """
    depths = np.asarray(depths, dtype=float)
    y = np.asarray(signal, dtype=float)

    mask = np.isfinite(depths) & np.isfinite(y) & (depths >= 0)
    depths = depths[mask]
    y = y[mask]

    if stderr is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.asarray(stderr, dtype=float)[mask]
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

    if len(depths) < 3:
        # Fall back to the old no-offset estimate if not enough points.
        return estimate_exponential_from_signal(depths, y, use_even_only=False)

    # Most Pauli fidelities should be in [0,1]. We allow a tiny overshoot above 1
    # to absorb finite-sample fluctuations without forcing a boundary estimate.
    r_values = np.linspace(0.0, 1.02, int(r_grid_size))

    best_r = np.nan
    best_sse = np.inf
    best_c = np.nan
    best_a = np.nan

    w = 1.0 / np.maximum(sigma, 1e-12)

    for r in r_values:
        basis = r ** depths
        X = np.column_stack([np.ones_like(depths), basis])
        Xw = X * w[:, None]
        yw = y * w
        try:
            beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        residual = yw - Xw @ beta
        sse = float(np.dot(residual, residual))
        if sse < best_sse:
            best_sse = sse
            best_r = float(r)
            best_c = float(beta[0])
            best_a = float(beta[1])

    return best_r



def fit_axis_decay_signed_continuous(depths, signal, stderr=None) -> Dict[str, Any]:
    """
    Continuous weighted nonlinear fit for a signed logical Pauli axis signal.

    Model:
        y_L = A * f^L = A * exp(log_f * L)

    This is the correct model for the current simulator because
    `measure_logical_pauli` returns +/-1 outcomes and the averaged signal is
    a signed Pauli expectation.  A projector probability should first be
    converted to an expectation via y=2P-1, or fitted with fixed c=1/2.

    Unlike a grid search over candidate f values, this fit treats log_f as a
    continuous parameter.  Bootstrap error bars are therefore not quantized by
    an artificial grid spacing.
    """
    L = np.asarray(depths, dtype=float)
    y = np.asarray(signal, dtype=float)

    mask = np.isfinite(L) & np.isfinite(y) & (L >= 0)
    L = L[mask]
    y = y[mask]

    if stderr is None:
        sigma = np.ones_like(y)
    else:
        sigma = np.asarray(stderr, dtype=float)[mask]
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)
    sigma = np.maximum(sigma, 1e-12)

    if len(L) < 2:
        return {
            "f": np.nan,
            "f_stderr": np.nan,
            "A": np.nan,
            "A_stderr": np.nan,
            "log_f": np.nan,
            "log_f_stderr": np.nan,
            "success": False,
            "cost": np.nan,
        }

    mag = np.abs(y)
    fit_mask = (L > 0) & (mag > 1e-12)
    if np.sum(fit_mask) >= 2:
        slope, intercept = np.polyfit(L[fit_mask], np.log(mag[fit_mask]), 1)
        log_f0 = float(slope)
        A0 = float(np.sign(np.mean(y)) * np.exp(intercept))
    else:
        log_f0 = -1e-3
        A0 = float(y[0] if len(y) else 1.0)

    # Allow a small f>1 fluctuation for finite-shot Monte Carlo noise.
    lower = np.array([-2.0, -1.0], dtype=float)     # A, log_f
    upper = np.array([+2.0, +0.05], dtype=float)

    p0 = np.array([A0, log_f0], dtype=float)
    p0 = np.minimum(np.maximum(p0, lower + 1e-9), upper - 1e-9)

    def residual(params):
        A, log_f = params
        pred = A * np.exp(log_f * L)
        return (pred - y) / sigma

    if SCIPY_AVAILABLE:
        fit = least_squares(
            residual,
            p0,
            bounds=(lower, upper),
            max_nfev=50000,
        )
        A, log_f = fit.x
        success = bool(fit.success)
        cost = float(fit.cost)

        try:
            J = fit.jac
            dof = max(1, len(fit.fun) - len(fit.x))
            reduced_chi2 = float(np.sum(fit.fun ** 2) / dof)
            cov = np.linalg.pinv(J.T @ J) * reduced_chi2
            A_stderr = float(np.sqrt(max(cov[0, 0], 0.0)))
            log_f_stderr = float(np.sqrt(max(cov[1, 1], 0.0)))
        except Exception:
            A_stderr = np.nan
            log_f_stderr = np.nan
    else:
        if np.sum(fit_mask) < 2:
            return {
                "f": np.nan,
                "f_stderr": np.nan,
                "A": np.nan,
                "A_stderr": np.nan,
                "log_f": np.nan,
                "log_f_stderr": np.nan,
                "success": False,
                "cost": np.nan,
            }
        log_f, log_abs_A = np.polyfit(L[fit_mask], np.log(mag[fit_mask]), 1)
        A = float(np.sign(np.mean(y)) * np.exp(log_abs_A))
        success = False
        cost = np.nan
        A_stderr = np.nan
        log_f_stderr = np.nan

    f = float(np.exp(log_f))
    f_stderr = float(f * log_f_stderr) if np.isfinite(log_f_stderr) else np.nan

    return {
        "f": f,
        "f_stderr": f_stderr,
        "A": float(A),
        "A_stderr": A_stderr,
        "log_f": float(log_f),
        "log_f_stderr": float(log_f_stderr),
        "success": success,
        "cost": cost,
    }



def fit_note_style_spectra(csb_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract note-style CSB spectral factors for each quarter-turn gate.

    Offdiag fit:
      A full complex fit is performed on the enhanced CSB signal:
          C_L = B * lambda^L,
          lambda = g * exp(i*(pi/2 + delta)).
      Pauli learning uses g=|lambda|.  Process fidelity uses both g and
      the angle error delta.

    Axis fit:
      A single axis eigenstate is used. Because the simulator records signed
      Pauli expectations, the axis signal is fitted continuously as A*f_A^L.
      For projector probabilities, convert P to 2P-1 or fix the identity offset.
    """
    rows = []
    axis_label = {"RZ90": "Z", "RX90": "X", "RY90": "Y"}

    for gate in ["RZ90", "RX90", "RY90"]:
        sub = csb_df[csb_df.target_gate == gate]

        q = merge_enhanced_quadratures(csb_df, gate)
        complex_fit = fit_complex_csb_eigenvalue(q, omega0=np.pi / 2)

        g_offdiag = complex_fit["g"]
        theta_offdiag = complex_fit["theta"]
        delta_theta = complex_fit["delta_theta"]

        axis = sub[sub.branch == "axis"].sort_values("depth")
        axis_signal = axis["mean"].to_numpy()
        axis_stderr = axis["stderr"].to_numpy() if "stderr" in axis.columns else None

        # The current simulator records signed Pauli expectations (+/-1 outcomes),
        # not raw projector probabilities.  Therefore no free identity-offset mode
        # should be fitted here.  Use a continuous weighted fit y_L=A*f_A^L.
        axis_fit = fit_axis_decay_signed_continuous(
            axis.depth.to_numpy(),
            axis_signal,
            stderr=axis_stderr,
        )
        g_axis = axis_fit["f"]

        fidelity = process_fidelity_from_complex_offdiag(
            g_offdiag=float(g_offdiag),
            delta_theta=float(delta_theta),
            g_axis=float(g_axis),
        )

        rows.append({
            "target_gate": gate,
            "axis": axis_label[gate],
            "g_offdiag_note": float(g_offdiag),
            "theta_offdiag": float(theta_offdiag),
            "delta_theta": float(delta_theta),
            "lambda_offdiag_real": float(complex_fit["lambda_real"]),
            "lambda_offdiag_imag": float(complex_fit["lambda_imag"]),
            "complex_fit_success": bool(complex_fit["success"]),
            "complex_fit_cost": float(complex_fit["cost"]),
            "g_axis": float(g_axis),
            "g_axis_local_stderr": float(axis_fit["f_stderr"]),
            "axis_amplitude": float(axis_fit["A"]),
            "axis_fit_success": bool(axis_fit["success"]),
            "axis_fit_cost": float(axis_fit["cost"]),
            "process_fidelity": fidelity["process_fidelity"],
            "process_infidelity": fidelity["process_infidelity"],
            "average_gate_fidelity": fidelity["average_gate_fidelity"],
            "average_gate_infidelity": fidelity["average_gate_infidelity"],
            "offdiag_equation": f"w_I - w_{axis_label[gate]}",
            "axis_equation": f"f_{axis_label[gate]}",
        })

    return pd.DataFrame(rows)


def solve_pauli_from_rz_rx_spectra(spectra_df: pd.DataFrame, enforce_simplex: bool = True):
    """
    Identify the single-logical-qubit Pauli distribution from RZ90 and RX90.

    Unknown vector is w = [w_I, w_X, w_Y, w_Z].

    RZ90 gives:
      g_offdiag_Z = w_I - w_Z,
      g_axis_Z    = f_Z = w_I - w_X - w_Y + w_Z.

    RX90 gives:
      g_offdiag_X = w_I - w_X,
      g_axis_X    = f_X = w_I + w_X - w_Y - w_Z.

    Together with normalization, these equations overdetermine w slightly;
    we solve by least squares and optionally project finite-shot estimates to
    the probability simplex.
    """
    d = {row["target_gate"]: row for _, row in spectra_df.iterrows()}
    gZ = float(d["RZ90"]["g_offdiag_note"])
    fZ = float(d["RZ90"]["g_axis"])
    gX = float(d["RX90"]["g_offdiag_note"])
    fX = float(d["RX90"]["g_axis"])

    # rows act on [wI, wX, wY, wZ]
    A = np.array([
        [1,  1,  1,  1],   # normalization
        [1,  0,  0, -1],   # RZ offdiag: wI - wZ
        [1, -1, -1,  1],   # RZ axis: fZ
        [1, -1,  0,  0],   # RX offdiag: wI - wX
        [1,  1, -1, -1],   # RX axis: fX
    ], dtype=float)
    b = np.array([1.0, gZ, fZ, gX, fX], dtype=float)

    w_lstsq, residuals, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
    w = simplex_project(w_lstsq) if enforce_simplex else w_lstsq

    wI, wX, wY, wZ = map(float, w)
    f = pauli_fidelities_from_probs(w)

    learned = {
        "p_I": wI, "p_X": wX, "p_Y": wY, "p_Z": wZ,
        "f_X": f["X"], "f_Y": f["Y"], "f_Z": f["Z"],
        "raw_lstsq_p_I": float(w_lstsq[0]),
        "raw_lstsq_p_X": float(w_lstsq[1]),
        "raw_lstsq_p_Y": float(w_lstsq[2]),
        "raw_lstsq_p_Z": float(w_lstsq[3]),
        "least_squares_residual_norm": float(np.linalg.norm(A @ w - b)),
        "least_squares_rank": int(rank),
    }
    return learned, A, b


def validate_with_ry_spectra(spectra_df: pd.DataFrame, learned: Dict[str, float]) -> Dict[str, float]:
    """
    Use RY90 only as a validation, not for fitting.

    The validation is reported both at the spectral level and as a single
    probability-level check.  It is not depth dependent: depth was already used
    only to fit the RY spectra.
    """
    d = {row["target_gate"]: row for _, row in spectra_df.iterrows()}
    w = np.array([learned["p_I"], learned["p_X"], learned["p_Y"], learned["p_Z"]], dtype=float)

    pred_offdiag_y = w[0] - w[2]                         # w_I - w_Y
    pred_axis_y = learned["f_Y"]                         # w_I - w_X + w_Y - w_Z
    meas_offdiag_y = float(d["RY90"]["g_offdiag_note"])
    meas_axis_y = float(d["RY90"]["g_axis"])

    # Convert the two RY spectral checks into a direct p_Y probability estimate.
    # RY offdiag: g_Y^off = w_I - w_Y -> w_Y = w_I - g_Y^off
    pY_from_ry_offdiag = w[0] - meas_offdiag_y
    # RY axis: f_Y = w_I - w_X + w_Y - w_Z -> w_Y = f_Y - w_I + w_X + w_Z
    pY_from_ry_axis = meas_axis_y - w[0] + w[1] + w[3]
    pY_from_ry_avg = 0.5 * (pY_from_ry_offdiag + pY_from_ry_axis)
    pY_pred = w[2]

    return {
        "RY_offdiag_wI_minus_wY_measured": meas_offdiag_y,
        "RY_offdiag_wI_minus_wY_predicted": pred_offdiag_y,
        "RY_offdiag_abs_error": abs(meas_offdiag_y - pred_offdiag_y),
        "RY_axis_fY_measured": meas_axis_y,
        "RY_axis_fY_predicted": pred_axis_y,
        "RY_axis_abs_error": abs(meas_axis_y - pred_axis_y),

        # Single probability-level validation requested by the user.
        "RY_pY_pred_from_RZ_RX": pY_pred,
        "RY_pY_meas_from_offdiag": pY_from_ry_offdiag,
        "RY_pY_diff_from_offdiag": pY_from_ry_offdiag - pY_pred,
        "RY_pY_meas_from_axis": pY_from_ry_axis,
        "RY_pY_diff_from_axis": pY_from_ry_axis - pY_pred,
        "RY_pY_meas_avg": pY_from_ry_avg,
        "RY_pY_diff_avg": pY_from_ry_avg - pY_pred,
        "RY_pY_abs_diff_avg": abs(pY_from_ry_avg - pY_pred),
    }


def pauli_fidelities_from_probs(w: np.ndarray | List[float]) -> Dict[str, float]:
    wI, wX, wY, wZ = map(float, w)
    return {
        "I": 1.0,
        "X": wI + wX - wY - wZ,
        "Y": wI - wX + wY - wZ,
        "Z": wI - wX - wY + wZ,
    }


def infer_pauli_from_csb(csb_df: pd.DataFrame):
    """
    Full learning pipeline:
      1. Fit CSB spectra attached to eigenoperators/subspaces.
      2. Use only RZ90+RX90 equations to learn w=[wI,wX,wY,wZ].
      3. Use RY90 spectra as a consistency check.
    """
    spectra_df = fit_note_style_spectra(csb_df)
    learned, A, b = solve_pauli_from_rz_rx_spectra(spectra_df, enforce_simplex=True)
    validation = validate_with_ry_spectra(spectra_df, learned)
    row = {**learned, **validation}
    pauli_df = pd.DataFrame([row])
    eta = {"X": learned["f_X"], "Y": learned["f_Y"], "Z": learned["f_Z"]}
    return spectra_df, pauli_df, eta


def ry_probability_validation_table(pauli_df: pd.DataFrame) -> pd.DataFrame:
    """
    Probability-level RY validation table.  The spectral table is preferred for the main text.
    """
    cols = [
        "RY_pY_pred_from_RZ_RX",
        "RY_pY_meas_from_offdiag",
        "RY_pY_diff_from_offdiag",
        "RY_pY_meas_from_axis",
        "RY_pY_diff_from_axis",
        "RY_pY_meas_avg",
        "RY_pY_diff_avg",
        "RY_pY_abs_diff_avg",
    ]
    return pauli_df[cols].copy()


def ry_probability_abs_validation_table(pauli_df: pd.DataFrame) -> pd.DataFrame:
    """
    Probability-level RY validation with absolute errors only.
    """
    row = pauli_df.iloc[0]
    return pd.DataFrame([{
        "pY_pred_from_RZ_RX": row["RY_pY_pred_from_RZ_RX"],
        "pY_meas_from_RY_offdiag": row["RY_pY_meas_from_offdiag"],
        "abs_diff_offdiag": abs(row["RY_pY_meas_from_offdiag"] - row["RY_pY_pred_from_RZ_RX"]),
        "pY_meas_from_RY_axis": row["RY_pY_meas_from_axis"],
        "abs_diff_axis": abs(row["RY_pY_meas_from_axis"] - row["RY_pY_pred_from_RZ_RX"]),
        "pY_meas_avg": row["RY_pY_meas_avg"],
        "abs_diff_avg": row["RY_pY_abs_diff_avg"],
    }])


def ry_spectrum_validation_table(pauli_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preferred compact RY validation table.

    Shows RY measured spectra, RZ+RX predictions, and absolute mismatch.
    If bootstrap standard errors are available, they are included.
    """
    row = pauli_df.iloc[0]

    def get(name):
        return row[name] if name in row.index else np.nan

    return pd.DataFrame([
        {
            "RY_spectrum": "g_offdiag = w_I - w_Y",
            "measured": get("RY_offdiag_wI_minus_wY_measured"),
            "measured_stderr": get("RY_offdiag_wI_minus_wY_measured_stderr"),
            "predicted_from_RZ_RX": get("RY_offdiag_wI_minus_wY_predicted"),
            "predicted_stderr": get("RY_offdiag_wI_minus_wY_predicted_stderr"),
            "abs_difference": get("RY_offdiag_abs_error"),
            "abs_difference_stderr": get("RY_offdiag_abs_error_stderr"),
        },
        {
            "RY_spectrum": "g_axis = f_Y",
            "measured": get("RY_axis_fY_measured"),
            "measured_stderr": get("RY_axis_fY_measured_stderr"),
            "predicted_from_RZ_RX": get("RY_axis_fY_predicted"),
            "predicted_stderr": get("RY_axis_fY_predicted_stderr"),
            "abs_difference": get("RY_axis_abs_error"),
            "abs_difference_stderr": get("RY_axis_abs_error_stderr"),
        },
    ])

















def complex_fit_summary_table(spectra_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compact table for complex offdiag eigenvalue fit and angle-aware fidelity.
    """
    cols = [
        "target_gate", "axis",
        "g_offdiag_note", "g_offdiag_note_stderr",
        "delta_theta", "delta_theta_stderr",
        "g_axis", "g_axis_stderr",
        "process_fidelity", "process_fidelity_stderr",
        "process_infidelity", "process_infidelity_stderr",
        "average_gate_infidelity", "average_gate_infidelity_stderr",
    ]
    return spectra_df[[c for c in cols if c in spectra_df.columns]].copy()



def inverse_pauli_quasiprobabilities_from_probs(w: np.ndarray | List[float]) -> Dict[str, float]:
    """
    Quasiprobability coefficients q_P for the inverse Pauli channel.

    If E(P_a)=f_a P_a for a=X,Y,Z, then E^{-1}(P_a)=P_a/f_a.
    The inverse map can be written as sum_P q_P P(.)P.  The q_P are the
    inverse Walsh-Hadamard transform of [1,1/f_X,1/f_Y,1/f_Z].
    """
    f = pauli_fidelities_from_probs(w)
    eps = 1e-12

    def safe_fidelity(value: float) -> float:
        value = float(value)
        if abs(value) < eps:
            return eps if value >= 0 else -eps
        return value

    fX = safe_fidelity(f["X"])
    fY = safe_fidelity(f["Y"])
    fZ = safe_fidelity(f["Z"])
    inv = {"I": 1.0, "X": 1.0/fX, "Y": 1.0/fY, "Z": 1.0/fZ}
    q = {
        "I": 0.25 * (inv["I"] + inv["X"] + inv["Y"] + inv["Z"]),
        "X": 0.25 * (inv["I"] + inv["X"] - inv["Y"] - inv["Z"]),
        "Y": 0.25 * (inv["I"] - inv["X"] + inv["Y"] - inv["Z"]),
        "Z": 0.25 * (inv["I"] - inv["X"] - inv["Y"] + inv["Z"]),
    }
    return q


def summarize_inverse_channel(pauli_df: pd.DataFrame) -> pd.DataFrame:
    row = pauli_df.iloc[0]
    w = np.array([row["p_I"], row["p_X"], row["p_Y"], row["p_Z"]], dtype=float)
    q = inverse_pauli_quasiprobabilities_from_probs(w)
    gamma = sum(abs(v) for v in q.values())
    return pd.DataFrame([{
        "q_I": q["I"], "q_X": q["X"], "q_Y": q["Y"], "q_Z": q["Z"],
        "gamma_per_cycle": gamma,
        "sampling_overhead_per_cycle": gamma**2,
    }])


# =============================================================================
# RB-like random Clifford circuits and Pauli-channel mitigation
# =============================================================================

PRIMITIVE_GATES = ["RZ90", "RX90", "RY90"]

GATE_MATS = {
    "RZ90": np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=int),
    "RX90": np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=int),
    "RY90": np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=int),
}


def mat_key(M):
    return tuple(M.astype(int).reshape(-1).tolist())


def build_clifford_words():
    I = np.eye(3, dtype=int)
    words = {mat_key(I): []}
    queue = [I]
    while queue:
        M = queue.pop(0)
        w = words[mat_key(M)]
        for g in PRIMITIVE_GATES:
            M2 = GATE_MATS[g] @ M
            k = mat_key(M2)
            if k not in words:
                words[k] = w + [g]
                queue.append(M2)
    return words

CLIFFORD_WORDS = build_clifford_words()


def sequence_matrix(seq):
    M = np.eye(3, dtype=int)
    for g in seq:
        M = GATE_MATS[g] @ M
    return M


def inverse_word_for_sequence(seq):
    M = sequence_matrix(seq)
    Minv = M.T
    return CLIFFORD_WORDS[mat_key(Minv)]


def inverse_conjugate_pauli_label(gate: str, label: str) -> str:
    # Map Pauli label backward across gate: G^dag P G.
    # Use signed permutation matrix.  label vector is transformed by M^T.
    idx = {"X":0, "Y":1, "Z":2}[label]
    v = np.zeros(3, dtype=int); v[idx] = 1
    out = GATE_MATS[gate].T @ v
    j = int(np.flatnonzero(np.abs(out))[0])
    return ["X", "Y", "Z"][j]


def mitigation_factor_for_sequence(seq, eta):
    """
    Pure spectral attenuation prediction for reference only.

    The actual mitigation demo below uses full PEC sampling, not this simple
    division.  This function is kept as a diagnostic.
    """
    current = "Z"
    factor = 1.0
    for gate in reversed(seq):
        factor *= eta[current]
        current = inverse_conjugate_pauli_label(gate, current)
    return float(factor)


def simulate_sequence_expectation(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seq: Sequence[str],
    shots: int,
    seed: int,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    outcomes = np.empty(int(shots), dtype=float)
    for shot in range(int(shots)):
        sim = StabilizerSteaneFTSimulator(
            decoder, noise, options, seed=int(rng.integers(2**32 - 1))
        )
        sim.prepare_initial("0")
        b = initial_bloch("0")
        for gate in seq:
            sim.apply_logical_gate(
                gate,
                noisy=True,
                apply_qec=options["qec_after_each_logical_gate"],
            )
            b = update_bloch(b, gate)
        outcomes[shot] = sim.measure_logical_pauli("Z", b)
    return (
        float(np.mean(outcomes)),
        float(np.std(outcomes, ddof=1) / math.sqrt(shots)) if shots > 1 else 0.0,
    )


def sample_inverse_pauli(rng: np.random.Generator, q: Mapping[str, float]):
    labels = ["I", "X", "Y", "Z"]
    weights = np.array([abs(float(q[p])) for p in labels], dtype=float)
    gamma = float(np.sum(weights))
    probs = weights / gamma
    idx = int(rng.choice(len(labels), p=probs))
    p = labels[idx]
    sign = +1.0 if q[p] >= 0 else -1.0
    return p, sign * gamma


def simulate_sequence_complete_pec(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seq: Sequence[str],
    inverse_q: Mapping[str, float],
    shots: int,
    seed: int,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    weighted = np.empty(int(shots), dtype=float)
    raw = np.empty(int(shots), dtype=float)
    for shot in range(int(shots)):
        sim = StabilizerSteaneFTSimulator(
            decoder, noise, options, seed=int(rng.integers(2**32 - 1))
        )
        sim.prepare_initial("0")
        b = initial_bloch("0")
        weight = 1.0
        for gate in seq:
            correction, q_weight = sample_inverse_pauli(rng, inverse_q)
            sim.apply_logical_gate(
                gate,
                noisy=True,
                apply_qec=options["qec_after_each_logical_gate"],
                post_ec_logical_pauli=correction,
            )
            b = update_bloch(b, gate)
            weight *= q_weight
        outcome = sim.measure_logical_pauli("Z", b)
        raw[shot] = outcome
        weighted[shot] = weight * outcome
    return {
        "pec_expectation": float(np.mean(weighted)),
        "pec_stderr": (
            float(np.std(weighted, ddof=1) / math.sqrt(shots)) if shots > 1 else 0.0
        ),
        "raw_expectation_same_samples": float(np.mean(raw)),
        "mean_abs_weight": float(np.mean(np.abs(weighted))),
        "max_abs_weight": float(np.max(np.abs(weighted))) if len(weighted) else 0.0,
    }


def _chunk_sizes(total: int, chunk: int) -> List[int]:
    total = int(total)
    chunk = max(1, int(chunk))
    out: List[int] = []
    while total > 0:
        n = min(chunk, total)
        out.append(n)
        total -= n
    return out


def estimate_rb_work(
    depths: Sequence[int],
    nseq: int,
    noisy_shots: int,
    pec_shots: int,
) -> Dict[str, Any]:
    depths = list(map(int, depths))
    approx_inverse_len = 3
    total_sequences = int(nseq) * len(depths)
    total_cycle_trajectories = int(nseq) * sum(
        (d + approx_inverse_len) * (int(noisy_shots) + int(pec_shots))
        for d in depths
    )
    return {
        "num_depths": len(depths),
        "num_sequences": total_sequences,
        "shots_per_sequence_total": int(noisy_shots) + int(pec_shots),
        "approx_logical_cycle_shot_updates": total_cycle_trajectories,
    }


def _simulate_noisy_chunk_worker(
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seq: Sequence[str],
    shots: int,
    seed: int,
) -> Dict[str, float]:
    decoder = get_worker_decoder(cache_file, min_rounds, max_rounds)
    mean, stderr = simulate_sequence_expectation(
        decoder, noise, options, seq, int(shots), int(seed)
    )
    n = int(shots)
    if n > 1:
        sample_var = float(stderr) ** 2 * n
        sumsq = sample_var * (n - 1) + n * float(mean) ** 2
    else:
        sumsq = n * float(mean) ** 2
    return {"shots": n, "sum": float(mean) * n, "sumsq": float(sumsq)}


def _simulate_pec_chunk_worker(
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seq: Sequence[str],
    inverse_q: Mapping[str, float],
    shots: int,
    seed: int,
) -> Dict[str, float]:
    decoder = get_worker_decoder(cache_file, min_rounds, max_rounds)
    res = simulate_sequence_complete_pec(
        decoder, noise, options, seq, inverse_q, int(shots), int(seed)
    )
    n = int(shots)
    mean = float(res["pec_expectation"])
    stderr = float(res["pec_stderr"])
    if n > 1:
        sample_var = stderr ** 2 * n
        sumsq = sample_var * (n - 1) + n * mean ** 2
    else:
        sumsq = n * mean ** 2
    return {
        "shots": n,
        "sum": mean * n,
        "sumsq": float(sumsq),
        "mean_abs_weight": float(res["mean_abs_weight"]),
        "max_abs_weight": float(res["max_abs_weight"]),
    }


def run_rb_mitigation_demo(
    decoder,
    eta: Mapping[str, float],
    noise: NoiseParams,
    options: Mapping[str, bool],
    depths: Sequence[int],
    nseq: int,
    noisy_shots: int,
    pec_shots: int,
    seed: int,
    n_jobs: int,
    noisy_chunk_shots: int,
    pec_chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    inverse_q: Optional[Mapping[str, float]],
    verbose: bool,
) -> pd.DataFrame:
    """Validate a CSB-learned logical Pauli inverse on random Clifford words."""
    rng = np.random.default_rng(seed)
    depths = list(map(int, depths))
    nseq = int(nseq)
    noisy_shots = int(noisy_shots)
    pec_shots = int(pec_shots)

    if inverse_q is None:
        fX, fY, fZ = float(eta["X"]), float(eta["Y"]), float(eta["Z"])
        w = np.array([
            (1 + fX + fY + fZ) / 4,
            (1 + fX - fY - fZ) / 4,
            (1 - fX + fY - fZ) / 4,
            (1 - fX - fY + fZ) / 4,
        ])
        inverse_q = inverse_pauli_quasiprobabilities_from_probs(w)

    sequence_records = []
    for depth in depths:
        for seq_id in range(nseq):
            forward = [
                PRIMITIVE_GATES[int(rng.integers(len(PRIMITIVE_GATES)))]
                for _ in range(depth)
            ]
            seq = forward + inverse_word_for_sequence(forward)
            sequence_records.append({
                "depth": depth,
                "seq_id": seq_id,
                "seq": seq,
                "total_primitive_gates": len(seq),
                "attenuation_factor": mitigation_factor_for_sequence(seq, eta),
            })

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    tasks = []
    task_meta = []
    for rec_idx, rec in enumerate(sequence_records):
        for n in _chunk_sizes(noisy_shots, noisy_chunk_shots):
            task_meta.append((rec_idx, "noisy"))
            tasks.append(("noisy", rec["seq"], n, int(rng.integers(2**32 - 1))))
        for n in _chunk_sizes(pec_shots, pec_chunk_shots):
            task_meta.append((rec_idx, "pec"))
            tasks.append(("pec", rec["seq"], n, int(rng.integers(2**32 - 1))))

    if verbose:
        print("Logical RB/PEC")
        print("  depths              =", depths)
        print("  sequences/depth     =", nseq)
        print("  noisy shots/sequence=", noisy_shots)
        print("  PEC shots/sequence  =", pec_shots)
        print("  n_jobs              =", n_jobs)

    def run_task(task):
        mode, seq, nshots, sd = task
        if mode == "noisy":
            return _simulate_noisy_chunk_worker(
                cache_path, min_rounds, max_rounds, noise, options, seq, nshots, sd
            )
        return _simulate_pec_chunk_worker(
            cache_path, min_rounds, max_rounds, noise, options, seq,
            inverse_q, nshots, sd
        )

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_task)(task) for task in tasks
        )
    else:
        results = [run_task(task) for task in tasks]

    seq_aggs = [{
        **rec,
        "noisy_sum": 0.0, "noisy_sumsq": 0.0, "noisy_n": 0,
        "pec_sum": 0.0, "pec_sumsq": 0.0, "pec_n": 0,
    } for rec in sequence_records]
    for (rec_idx, mode), res in zip(task_meta, results):
        agg = seq_aggs[rec_idx]
        n = int(res["shots"])
        agg[f"{mode}_sum"] += float(res["sum"])
        agg[f"{mode}_sumsq"] += float(res["sumsq"])
        agg[f"{mode}_n"] += n

    for agg in seq_aggs:
        for mode in ("noisy", "pec"):
            n = int(agg[f"{mode}_n"])
            s = float(agg[f"{mode}_sum"])
            ss = float(agg[f"{mode}_sumsq"])
            agg[f"{mode}_expectation"] = s / max(n, 1)
            if n > 1:
                var = max((ss - s * s / n) / (n - 1), 0.0)
                agg[f"{mode}_sampling_stderr_within_sequence"] = math.sqrt(var / n)
            else:
                agg[f"{mode}_sampling_stderr_within_sequence"] = 0.0
        agg["postprocessed_spectral_check"] = (
            agg["noisy_expectation"] / agg["attenuation_factor"]
            if abs(agg["attenuation_factor"]) > 1e-12 else np.nan
        )

    seq_df = pd.DataFrame(seq_aggs)
    rows = []
    for depth in depths:
        sub = seq_df[seq_df.depth == depth]
        row = {
            "rb_depth": depth,
            "num_sequences": len(sub),
            "mean_total_primitive_gates": float(sub.total_primitive_gates.mean()),
            "noisy_expectation": float(sub.noisy_expectation.mean()),
            "pec_expectation": float(sub.pec_expectation.mean()),
            "postprocessed_spectral_check": float(np.nanmean(sub.postprocessed_spectral_check)),
        }
        for mode in ("noisy", "pec"):
            vals = sub[f"{mode}_expectation"].to_numpy(dtype=float)
            row[f"{mode}_std_across_sequences"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# Merged Numba-accelerated CSB trajectory kernel
# =============================================================================
#
# This section is merged from logical_csb_stabilizer_numba_fast.py.
# It preserves the readable Python reference simulator above, but overrides
# simulate_csb_point(...) with a Numba-compiled Pauli-frame trajectory kernel
# whenever numba is available.  All high-level fitting, Pauli learning, RB/PEC,
# plotting and table utilities remain unchanged.

_PYTHON_REFERENCE_SIMULATE_CSB_POINT = _simulate_csb_point_reference
_PYTHON_REFERENCE_RUN_CSB_SPECTRUM = _run_csb_spectrum_reference

try:
    import numba as nb
    NUMBA_AVAILABLE = True
except Exception:
    nb = None
    NUMBA_AVAILABLE = False

# Operation type codes.
OP_DATA_IDLE = 0
OP_RESET_X = 1
OP_RESET_Z = 2
OP_CX = 3
OP_MEAS_Z = 4
OP_MEAS_X = 5

BASIS_NONE = 0
BASIS_X_CHECK = 1
BASIS_Z_CHECK = 2

KIND_NONE = 0
KIND_SYNDROME = 1
KIND_FLAG = 2

GATE_RZ90 = 1
GATE_RX90 = 2
GATE_RY90 = 3

MEAS_X = 1
MEAS_Y = 2
MEAS_Z = 3

INIT_0 = 1
INIT_1 = 2
INIT_PLUS = 3
INIT_MINUS = 4
INIT_PLUS_I = 5
INIT_MINUS_I = 6

PREP_H = 1
PREP_X = 2
PREP_Z = 3
PREP_S = 4

# Steane parity-check row masks, q0 is LSB.
H_ROW0 = (1 << 3) | (1 << 4) | (1 << 5) | (1 << 6)
H_ROW1 = (1 << 1) | (1 << 2) | (1 << 5) | (1 << 6)
H_ROW2 = (1 << 0) | (1 << 2) | (1 << 4) | (1 << 6)
LOGICAL_MASK_7 = (1 << 7) - 1
UINT64_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)

_COMPILED_CACHE: Dict[Tuple[int, int, int], Dict[str, Any]] = {}


def _label_to_init_code(label: str) -> int:
    table = {"0": INIT_0, "1": INIT_1, "+": INIT_PLUS, "-": INIT_MINUS, "+i": INIT_PLUS_I, "-i": INIT_MINUS_I}
    return table[label]


def _pauli_to_meas_code(pauli: str) -> int:
    return {"X": MEAS_X, "Y": MEAS_Y, "Z": MEAS_Z}[pauli]


def _gate_to_code(gate: str) -> int:
    return {"RZ90": GATE_RZ90, "RX90": GATE_RX90, "RY90": GATE_RY90}[gate]


def _encode_key_from_lists(x_syn, z_syn, x_flags, z_flags, completed: int) -> int:
    """Encode decoder key into a signed int64-compatible integer."""
    code = int(completed) & 7
    shift = 3
    for arr in (x_syn, z_syn, x_flags, z_flags):
        for r in range(4):
            val = int(arr[r]) if r < len(arr) else 0
            code |= (val & 7) << shift
            shift += 3
    return int(code)


def _mask_from_bits(bits) -> int:
    out = 0
    for i, b in enumerate(bits):
        if int(b) & 1:
            out |= 1 << i
    return int(out)


def _compile_model_and_decoder(decoder):
    model = decoder.model
    nops = len(model.ops)
    op_type = np.zeros(nops, dtype=np.int8)
    q1 = np.full(nops, -1, dtype=np.int8)
    q2 = np.full(nops, -1, dtype=np.int8)
    round_id = np.full(nops, -1, dtype=np.int8)
    basis_id = np.zeros(nops, dtype=np.int8)
    check_id = np.full(nops, -1, dtype=np.int8)
    kind_id = np.zeros(nops, dtype=np.int8)

    for i, op in enumerate(model.ops):
        if op.name == "DATA_IDLE":
            op_type[i] = OP_DATA_IDLE
        elif op.name == "RESET_X":
            op_type[i] = OP_RESET_X; q1[i] = int(op.args[0])
        elif op.name == "RESET_Z":
            op_type[i] = OP_RESET_Z; q1[i] = int(op.args[0])
        elif op.name == "CX":
            op_type[i] = OP_CX; q1[i] = int(op.args[0]); q2[i] = int(op.args[1])
        elif op.name == "MEAS_Z":
            op_type[i] = OP_MEAS_Z; q1[i] = int(op.args[0])
        elif op.name == "MEAS_X":
            op_type[i] = OP_MEAS_X; q1[i] = int(op.args[0])
        else:
            raise ValueError(op.name)

        if len(op.meta) >= 4:
            r, basis, chk, kind = op.meta[:4]
            round_id[i] = int(r)
            basis_id[i] = BASIS_X_CHECK if basis == "X_CHECK" else BASIS_Z_CHECK if basis == "Z_CHECK" else BASIS_NONE
            check_id[i] = int(chk)
            kind_id[i] = KIND_SYNDROME if kind == "syndrome" else KIND_FLAG if kind == "flag" else KIND_NONE

    key_codes = []
    rx_masks = []
    rz_masks = []
    for key, (rx, rz) in decoder.table.items():
        xs, zs, xf, zf = key
        completed = len(xs)
        key_codes.append(_encode_key_from_lists(xs, zs, xf, zf, completed))
        rx_masks.append(_mask_from_bits(rx))
        rz_masks.append(_mask_from_bits(rz))

    order = np.argsort(np.array(key_codes, dtype=np.int64))
    key_codes = np.array(key_codes, dtype=np.int64)[order]
    rx_masks = np.array(rx_masks, dtype=np.uint16)[order]
    rz_masks = np.array(rz_masks, dtype=np.uint16)[order]

    return {
        "op_type": op_type,
        "q1": q1,
        "q2": q2,
        "round_id": round_id,
        "basis_id": basis_id,
        "check_id": check_id,
        "kind_id": kind_id,
        "min_rounds": int(model.min_rounds),
        "max_rounds": int(model.max_rounds),
        "key_codes": key_codes,
        "rx_masks": rx_masks,
        "rz_masks": rz_masks,
    }


def get_compiled_numba_data(decoder, force_rebuild: bool = False):
    key = (
        int(decoder.model.min_rounds),
        int(decoder.model.max_rounds),
        int(len(decoder.table)),
    )
    if force_rebuild or key not in _COMPILED_CACHE:
        _COMPILED_CACHE[key] = _compile_model_and_decoder(decoder)
    return _COMPILED_CACHE[key]


if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _popcount_u16(x):
        c = 0
        x = int(x)
        while x:
            x &= x - 1
            c += 1
        return c

    @nb.njit(cache=True)
    def _parity_u16(x):
        return _popcount_u16(x) & 1

    @nb.njit(cache=True)
    def _rng_next_u64(state):
        # 64-bit LCG.  Overflow is intentional in uint64 arithmetic.
        state = np.uint64(state) * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
        return state

    @nb.njit(cache=True)
    def _rng_uniform01(state):
        state = _rng_next_u64(state)
        # Use top 53 bits for a double in [0,1).
        x = state >> np.uint64(11)
        return state, float(x) * (1.0 / 9007199254740992.0)

    @nb.njit(cache=True)
    def _rng_int(state, n):
        state, u = _rng_uniform01(state)
        k = int(u * n)
        if k >= n:
            k = n - 1
        return state, k

    @nb.njit(cache=True)
    def _apply_pauli_code(x_mask, z_mask, q, pcode):
        # pcode: 0 I, 1 X, 2 Y, 3 Z
        bit = np.uint16(1 << q)
        if pcode == 1:  # X
            x_mask = np.uint16(x_mask ^ bit)
        elif pcode == 2:  # Y
            x_mask = np.uint16(x_mask ^ bit)
            z_mask = np.uint16(z_mask ^ bit)
        elif pcode == 3:  # Z
            z_mask = np.uint16(z_mask ^ bit)
        return x_mask, z_mask

    @nb.njit(cache=True)
    def _sample_oneq_pauli(state, p):
        state, u = _rng_uniform01(state)
        if u >= p:
            return state, 0
        state, k = _rng_int(state, 3)
        return state, k + 1

    @nb.njit(cache=True)
    def _sample_twoq_pauli(state, p):
        state, u = _rng_uniform01(state)
        if u >= p:
            return state, 0, 0
        state, k = _rng_int(state, 15)
        code = k + 1  # 1..15, with 0=II excluded.
        return state, code // 4, code % 4

    @nb.njit(cache=True)
    def _syndrome_from_mask(e_mask):
        s = 0
        if _parity_u16(e_mask & H_ROW0):
            s |= 1
        if _parity_u16(e_mask & H_ROW1):
            s |= 2
        if _parity_u16(e_mask & H_ROW2):
            s |= 4
        return s

    @nb.njit(cache=True)
    def _ordinary_decode_mask(syndrome):
        if syndrome == 0:
            return np.uint16(0)
        if syndrome == 4:
            return np.uint16(1 << 0)
        if syndrome == 2:
            return np.uint16(1 << 1)
        if syndrome == 6:
            return np.uint16(1 << 2)
        if syndrome == 1:
            return np.uint16(1 << 3)
        if syndrome == 5:
            return np.uint16(1 << 4)
        if syndrome == 3:
            return np.uint16(1 << 5)
        if syndrome == 7:
            return np.uint16(1 << 6)
        return np.uint16(0)

    @nb.njit(cache=True)
    def _encode_key_numba(x_syn, z_syn, x_flags, z_flags, completed):
        code = np.int64(completed & 7)
        shift = 3
        for r in range(4):
            code |= np.int64((x_syn[r] & 7) << shift); shift += 3
        for r in range(4):
            code |= np.int64((z_syn[r] & 7) << shift); shift += 3
        for r in range(4):
            code |= np.int64((x_flags[r] & 7) << shift); shift += 3
        for r in range(4):
            code |= np.int64((z_flags[r] & 7) << shift); shift += 3
        return code

    @nb.njit(cache=True)
    def _lookup_decoder(key, key_codes, rx_masks, rz_masks):
        lo = 0
        hi = key_codes.shape[0] - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            val = key_codes[mid]
            if val == key:
                return True, rx_masks[mid], rz_masks[mid]
            if val < key:
                lo = mid + 1
            else:
                hi = mid - 1
        return False, np.uint16(0), np.uint16(0)

    @nb.njit(cache=True)
    def _terminal_reliable_numba(x_syn, z_syn, x_flags, z_flags, completed, min_rounds):
        if completed < min_rounds or completed < 2:
            return False
        last = completed - 1
        prev = completed - 2
        if x_flags[last] != 0 or z_flags[last] != 0:
            return False
        return x_syn[last] == x_syn[prev] and z_syn[last] == z_syn[prev]

    @nb.njit(cache=True)
    def _apply_1q_gate_noise_all_data_numba(x_mask, z_mask, state, p_1q):
        for q in range(7):
            state, pc = _sample_oneq_pauli(state, p_1q)
            if pc != 0:
                x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, q, pc)
        return x_mask, z_mask, state

    @nb.njit(cache=True)
    def _apply_logical_gate_numba(x_mask, z_mask, gate_code, state, p_1q, noisy):
        # Binary Clifford propagation on the data-frame mask only.
        if gate_code == GATE_RZ90:
            z_mask = np.uint16(z_mask ^ x_mask)
        elif gate_code == GATE_RX90:
            x_mask = np.uint16(x_mask ^ z_mask)
        elif gate_code == GATE_RY90:
            tmp = x_mask; x_mask = z_mask; z_mask = tmp
        if noisy:
            x_mask, z_mask, state = _apply_1q_gate_noise_all_data_numba(x_mask, z_mask, state, p_1q)
        return x_mask, z_mask, state

    @nb.njit(cache=True)
    def _run_ft_ec_numba(
        x_data,
        z_data,
        state,
        p_1q,
        p_2q,
        p_meas,
        p_reset,
        p_idle,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
        fallback_on_unverified,
    ):
        x_mask = np.uint16(x_data)
        z_mask = np.uint16(z_data)
        x_syn = np.zeros(4, dtype=np.int64)
        z_syn = np.zeros(4, dtype=np.int64)
        x_flags = np.zeros(4, dtype=np.int64)
        z_flags = np.zeros(4, dtype=np.int64)
        completed = 0

        for idx in range(op_type.shape[0]):
            typ = op_type[idx]
            if typ == OP_DATA_IDLE:
                for q in range(7):
                    state, pc = _sample_oneq_pauli(state, p_idle)
                    if pc != 0:
                        x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, q, pc)

            elif typ == OP_RESET_X or typ == OP_RESET_Z:
                q = int(q1[idx])
                bit = np.uint16(1 << q)
                x_mask = np.uint16(x_mask & np.uint16(~bit))
                z_mask = np.uint16(z_mask & np.uint16(~bit))
                state, pc = _sample_oneq_pauli(state, p_reset)
                if pc != 0:
                    x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, q, pc)

            elif typ == OP_CX:
                c = int(q1[idx]); t = int(q2[idx])
                cb = np.uint16(1 << c); tb = np.uint16(1 << t)
                if x_mask & cb:
                    x_mask = np.uint16(x_mask ^ tb)
                if z_mask & tb:
                    z_mask = np.uint16(z_mask ^ cb)
                state, pc1, pc2 = _sample_twoq_pauli(state, p_2q)
                if pc1 != 0:
                    x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, c, pc1)
                if pc2 != 0:
                    x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, t, pc2)

            elif typ == OP_MEAS_Z:
                q = int(q1[idx])
                outcome = 1 if (x_mask & np.uint16(1 << q)) else 0
                state, u = _rng_uniform01(state)
                if u < p_meas:
                    outcome ^= 1
                r = int(round_id[idx]); b = int(basis_id[idx]); chk = int(check_id[idx]); k = int(kind_id[idx])
                if b == BASIS_Z_CHECK and k == KIND_SYNDROME:
                    z_syn[r] ^= outcome << chk
                elif b == BASIS_Z_CHECK and k == KIND_FLAG:
                    z_flags[r] ^= outcome << chk
                elif b == BASIS_X_CHECK and k == KIND_FLAG:
                    x_flags[r] ^= outcome << chk
                bit = np.uint16(1 << q)
                x_mask = np.uint16(x_mask & np.uint16(~bit))
                z_mask = np.uint16(z_mask & np.uint16(~bit))

            elif typ == OP_MEAS_X:
                q = int(q1[idx])
                outcome = 1 if (z_mask & np.uint16(1 << q)) else 0
                state, u = _rng_uniform01(state)
                if u < p_meas:
                    outcome ^= 1
                r = int(round_id[idx]); b = int(basis_id[idx]); chk = int(check_id[idx]); k = int(kind_id[idx])
                if b == BASIS_X_CHECK and k == KIND_SYNDROME:
                    x_syn[r] ^= outcome << chk
                elif b == BASIS_X_CHECK and k == KIND_FLAG:
                    x_flags[r] ^= outcome << chk
                elif b == BASIS_Z_CHECK and k == KIND_FLAG:
                    z_flags[r] ^= outcome << chk
                bit = np.uint16(1 << q)
                x_mask = np.uint16(x_mask & np.uint16(~bit))
                z_mask = np.uint16(z_mask & np.uint16(~bit))

                if b == BASIS_Z_CHECK and chk == 2 and k == KIND_FLAG:
                    completed = r + 1
                    if _terminal_reliable_numba(x_syn, z_syn, x_flags, z_flags, completed, min_rounds):
                        break

        if completed == 0:
            completed = max_rounds

        key = _encode_key_numba(x_syn, z_syn, x_flags, z_flags, completed)
        found, rx_mask, rz_mask = _lookup_decoder(key, key_codes, rx_masks, rz_masks)
        unverified = 0
        if not found:
            if fallback_on_unverified:
                unverified = 1
                # final Z-check syndrome detects X errors; final X-check syndrome detects Z errors
                rx_mask = _ordinary_decode_mask(int(z_syn[completed - 1]))
                rz_mask = _ordinary_decode_mask(int(x_syn[completed - 1]))
            else:
                # No exception in numba kernel; mark as unverified and use no correction.
                unverified = 1
                rx_mask = np.uint16(0)
                rz_mask = np.uint16(0)

        x_mask = np.uint16(x_mask ^ rx_mask)
        z_mask = np.uint16(z_mask ^ rz_mask)

        return np.uint16(x_mask & LOGICAL_MASK_7), np.uint16(z_mask & LOGICAL_MASK_7), state, unverified, completed

    @nb.njit(cache=True)
    def _apply_one_prep_gate_numba(x_mask, z_mask, state, prep_code, p_1q, noisy):
        if prep_code == PREP_H:
            tmp = x_mask
            x_mask = z_mask
            z_mask = tmp
        elif prep_code == PREP_S:
            z_mask = np.uint16(z_mask ^ x_mask)
        # Logical X and Z change only Pauli signs in the binary frame.
        if noisy:
            x_mask, z_mask, state = _apply_1q_gate_noise_all_data_numba(
                x_mask, z_mask, state, p_1q
            )
        return x_mask, z_mask, state

    @nb.njit(cache=True)
    def _prepare_initial_frame_numba(
        init_code,
        x_mask,
        z_mask,
        state,
        p_1q,
        p_2q,
        p_meas,
        p_reset,
        p_idle,
        prep_gates_noisy,
        prep_qec_after_gate,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        gates = np.zeros(2, dtype=np.int8)
        ngates = 0
        if init_code == INIT_1:
            gates[0] = PREP_X; ngates = 1
        elif init_code == INIT_PLUS:
            gates[0] = PREP_H; ngates = 1
        elif init_code == INIT_MINUS:
            gates[0] = PREP_H; gates[1] = PREP_Z; ngates = 2
        elif init_code == INIT_PLUS_I:
            gates[0] = PREP_H; gates[1] = PREP_S; ngates = 2
        elif init_code == INIT_MINUS_I:
            gates[0] = PREP_H; gates[1] = PREP_S; ngates = 2

        unverified = 0
        ec_cycles = 0
        for j in range(ngates):
            x_mask, z_mask, state = _apply_one_prep_gate_numba(
                x_mask, z_mask, state, int(gates[j]), p_1q, prep_gates_noisy
            )
            if prep_qec_after_gate:
                x_mask, z_mask, state, unv, _completed = _run_ft_ec_numba(
                    x_mask, z_mask, state,
                    p_1q, p_2q, p_meas, p_reset, p_idle,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                    fallback_on_unverified,
                )
                unverified += unv
                ec_cycles += 1
        return x_mask, z_mask, state, unverified, ec_cycles

    @nb.njit(cache=True)
    def _initial_bloch_code(init_code):
        if init_code == INIT_0:
            return 0, 0, 1
        if init_code == INIT_1:
            return 0, 0, -1
        if init_code == INIT_PLUS:
            return 1, 0, 0
        if init_code == INIT_MINUS:
            return -1, 0, 0
        if init_code == INIT_PLUS_I:
            return 0, 1, 0
        if init_code == INIT_MINUS_I:
            return 0, -1, 0
        return 0, 0, 1

    @nb.njit(cache=True)
    def _update_bloch_code(bx, by, bz, gate_code):
        if gate_code == GATE_RZ90:
            return -by, bx, bz
        if gate_code == GATE_RX90:
            return bx, -bz, by
        if gate_code == GATE_RY90:
            return bz, by, -bx
        return bx, by, bz

    @nb.njit(cache=True)
    def _ideal_outcome_from_bloch(state, bx, by, bz, meas_code):
        if meas_code == MEAS_X:
            mean = bx
        elif meas_code == MEAS_Y:
            mean = by
        else:
            mean = bz
        p_plus = 0.5 * (1.0 + float(mean))
        state, u = _rng_uniform01(state)
        out = 1 if u < p_plus else -1
        return state, out

    @nb.njit(cache=True)
    def _logical_measurement_flip_numba(x_data, z_data, state, meas_code, p_1q, p_final_meas, measurement_basis_noisy):
        tx = np.uint16(x_data)
        tz = np.uint16(z_data)

        if meas_code == MEAS_Z:
            pass
        elif meas_code == MEAS_X:
            tmp = tx; tx = tz; tz = tmp
            if measurement_basis_noisy:
                tx, tz, state = _apply_1q_gate_noise_all_data_numba(tx, tz, state, p_1q)
        elif meas_code == MEAS_Y:
            # Sdg maps Y to X in binary frame, then H maps X to Z.
            tz = np.uint16(tz ^ tx)
            if measurement_basis_noisy:
                tx, tz, state = _apply_1q_gate_noise_all_data_numba(tx, tz, state, p_1q)
            tmp = tx; tx = tz; tz = tmp
            if measurement_basis_noisy:
                tx, tz, state = _apply_1q_gate_noise_all_data_numba(tx, tz, state, p_1q)

        bits = np.uint16(tx)
        for q in range(7):
            state, u = _rng_uniform01(state)
            if u < p_final_meas:
                bits = np.uint16(bits ^ np.uint16(1 << q))

        syn = _syndrome_from_mask(bits)
        corr = _ordinary_decode_mask(syn)
        bits = np.uint16(bits ^ corr)
        flip = _popcount_u16(bits) & 1
        return state, flip

    @nb.njit(cache=True)
    def _simulate_csb_point_numba_kernel(
        gate_code,
        init_code,
        meas_code,
        measure_sign,
        depth,
        shots,
        seed,
        p_1q,
        p_2q,
        p_meas,
        p_reset,
        p_idle,
        p_final_meas,
        prep_gates_noisy,
        prep_qec_after_gate,
        measurement_basis_noisy,
        qec_after_each_logical_gate,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        total = 0.0
        total2 = 0.0
        unverified_total = 0
        ec_cycles_total = 0

        base_state = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)

        for shot in range(shots):
            state = base_state + np.uint64(shot) * np.uint64(0xD1B54A32D192ED03)
            # Warm up/scramble state.
            state = _rng_next_u64(state)
            state = _rng_next_u64(state)

            x_data = np.uint16(0)
            z_data = np.uint16(0)
            x_data, z_data, state, prep_unv, prep_ec = _prepare_initial_frame_numba(
                init_code, x_data, z_data, state,
                p_1q, p_2q, p_meas, p_reset, p_idle,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            unverified_total += prep_unv
            ec_cycles_total += prep_ec

            bx, by, bz = _initial_bloch_code(init_code)

            for _g in range(depth):
                x_data, z_data, state = _apply_logical_gate_numba(x_data, z_data, gate_code, state, p_1q, True)
                bx, by, bz = _update_bloch_code(bx, by, bz, gate_code)
                if qec_after_each_logical_gate:
                    x_data, z_data, state, unv, completed = _run_ft_ec_numba(
                        x_data,
                        z_data,
                        state,
                        p_1q,
                        p_2q,
                        p_meas,
                        p_reset,
                        p_idle,
                        op_type,
                        q1,
                        q2,
                        round_id,
                        basis_id,
                        check_id,
                        kind_id,
                        min_rounds,
                        max_rounds,
                        key_codes,
                        rx_masks,
                        rz_masks,
                        fallback_on_unverified,
                    )
                    unverified_total += unv
                    ec_cycles_total += 1

            state, ideal_out = _ideal_outcome_from_bloch(state, bx, by, bz, meas_code)
            state, flip = _logical_measurement_flip_numba(x_data, z_data, state, meas_code, p_1q, p_final_meas, measurement_basis_noisy)
            out = ideal_out
            if flip & 1:
                out = -out
            out = out * measure_sign
            total += float(out)
            total2 += float(out * out)

        if shots <= 0:
            return np.nan, np.nan, 0.0
        mean = total / shots
        if shots > 1:
            var = max(0.0, (total2 - shots * mean * mean) / (shots - 1))
            stderr = math.sqrt(var / shots)
        else:
            stderr = 0.0
        unv_rate = float(unverified_total) / max(ec_cycles_total, 1)
        return mean, stderr, unv_rate


def simulate_csb_point(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    target_gate: str,
    branch_name: str,
    depth: int,
    shots: int,
    seed: int,
):
    """Simulate one stochastic-Pauli logical CSB branch/depth point."""
    if not NUMBA_AVAILABLE:
        return _PYTHON_REFERENCE_SIMULATE_CSB_POINT(
            decoder, noise, options, target_gate, branch_name, depth, shots, seed
        )

    data = get_compiled_numba_data(decoder)
    branch = get_csb_branches(target_gate)[branch_name]
    return _simulate_csb_point_numba_kernel(
        int(_gate_to_code(target_gate)),
        int(_label_to_init_code(branch["initial"])),
        int(_pauli_to_meas_code(branch["measure"])),
        int(branch.get("measure_sign", +1)),
        int(depth),
        int(shots),
        int(seed),
        float(noise.p_1q),
        float(noise.p_2q),
        float(noise.p_meas),
        float(noise.p_reset),
        float(noise.p_idle),
        float(noise.p_final_meas),
        bool(options["prep_gates_noisy"]),
        bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["qec_after_each_logical_gate"]),
        bool(options["fallback_on_unverified"]),
        data["op_type"], data["q1"], data["q2"], data["round_id"],
        data["basis_id"], data["check_id"], data["kind_id"],
        int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def warmup_numba(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seed: int,
):
    """Compile the stochastic CSB Numba kernel using a notebook-supplied seed."""
    return simulate_csb_point(
        decoder, noise, options, "RZ90", "axis", depth=1, shots=2, seed=int(seed)
    )


def using_numba() -> bool:
    return bool(NUMBA_AVAILABLE)


def _combine_csb_chunk_rows(sub: pd.DataFrame) -> Dict[str, Any]:
    """Combine independent shot chunks for one CSB branch/depth point."""
    sub = sub.sort_values("chunk_id")
    shots_arr = sub["shots"].to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    stderrs = sub["stderr"].to_numpy(dtype=float)

    total_shots = int(np.sum(shots_arr))
    if total_shots <= 0:
        mean = np.nan
        stderr = np.nan
        unv = np.nan
    else:
        mean = float(np.sum(shots_arr * means) / total_shots)

        # Reconstruct within-chunk sample variances from stderr_i = sqrt(var_i / n_i)
        # and combine with between-chunk variation.
        if total_shots > 1 and len(sub) > 0:
            variances = (stderrs * np.sqrt(np.maximum(shots_arr, 1.0))) ** 2
            ss_within = float(np.sum(np.maximum(shots_arr - 1.0, 0.0) * variances))
            ss_between = float(np.sum(shots_arr * (means - mean) ** 2))
            sample_var = max(0.0, (ss_within + ss_between) / max(total_shots - 1, 1))
            stderr = float(np.sqrt(sample_var / total_shots))
        else:
            stderr = 0.0

        unv = float(
            np.sum(shots_arr * sub["unverified_rate"].to_numpy(dtype=float))
            / total_shots
        )

    first = sub.iloc[0].to_dict()
    keep_keys = [
        "target_gate", "branch", "branch_kind", "initial", "measure", "measure_sign",
        "step", "rep", "depth", "requested_total_shots",
    ]
    row = {k: first[k] for k in keep_keys if k in first}
    row["mean"] = mean
    row["stderr"] = stderr
    row["shots"] = total_shots
    row["unverified_rate"] = unv
    row["seed"] = int(first.get("seed", 0))
    row["n_chunks"] = int(len(sub))
    return row


def run_csb_spectrum(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    target_gates: Sequence[str],
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    """Run stochastic-Pauli logical CSB with notebook-supplied settings."""
    rep = int(rep)
    steps = [int(s) for s in steps]
    shots = int(shots)
    chunk_shots = max(1, int(chunk_shots))
    rng = np.random.default_rng(seed)
    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    tasks = []
    point_id = 0
    for target_gate in target_gates:
        for branch_name, branch_spec in get_csb_branches(target_gate).items():
            branch_total = max(1, shots // 2) if branch_spec["kind"].startswith("equatorial") else shots
            for step in steps:
                depth = rep * step
                for chunk_id, n_chunk in enumerate(_chunk_sizes(branch_total, chunk_shots)):
                    tasks.append({
                        "point_id": point_id, "chunk_id": chunk_id,
                        "target_gate": target_gate, "branch": branch_name,
                        "branch_kind": branch_spec["kind"],
                        "initial": branch_spec["initial"],
                        "measure": branch_spec["measure"],
                        "measure_sign": int(branch_spec.get("measure_sign", +1)),
                        "step": step, "rep": rep, "depth": depth,
                        "shots": int(n_chunk), "requested_total_shots": shots,
                        "seed": int(rng.integers(2**32 - 1)),
                    })
                point_id += 1

    if verbose:
        print("Logical CSB")
        print("  target gates =", list(target_gates))
        print("  steps        =", steps)
        print("  rep          =", rep)
        print("  true depths  =", [rep * s for s in steps])
        print("  shots        =", shots)
        print("  chunk shots  =", chunk_shots)
        print("  n_jobs       =", n_jobs)

    def run_task(task):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        mean, se, unv = simulate_csb_point(
            dec, noise, options, task["target_gate"], task["branch"],
            task["depth"], task["shots"], task["seed"]
        )
        return {**task, "mean": mean, "stderr": se, "unverified_rate": unv}

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        chunk_rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(run_task)(task) for task in tasks
        )
    else:
        chunk_rows = [run_task(task) for task in tasks]

    chunk_df = pd.DataFrame(chunk_rows)
    rows = [_combine_csb_chunk_rows(sub) for _, sub in chunk_df.groupby("point_id", sort=True)]
    return pd.DataFrame(rows).sort_values(["target_gate", "branch", "step"]).reset_index(drop=True)


def run_rb_pec_experiment(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    pauli_df: pd.DataFrame,
    rb_depths: Sequence[int],
    nseq: int,
    noisy_shots: int,
    pec_shots: int,
    seed: int,
    n_jobs: int,
    noisy_chunk_shots: int,
    pec_chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    row = pauli_df.iloc[0]
    inverse_q = inverse_pauli_quasiprobabilities_from_probs(
        [row["p_I"], row["p_X"], row["p_Y"], row["p_Z"]]
    )
    eta = {"X": row["f_X"], "Y": row["f_Y"], "Z": row["f_Z"]}
    return run_rb_mitigation_demo(
        decoder, eta, noise, options, rb_depths, nseq, noisy_shots, pec_shots,
        seed, n_jobs, noisy_chunk_shots, pec_chunk_shots, cache_file,
        min_rounds, max_rounds, inverse_q, verbose
    )


# =============================================================================
# Two-logical-qubit CX CSB and PEC
# =============================================================================


PAULI1 = ["I", "X", "Y", "Z"]
PAULI2 = [a + b for a in PAULI1 for b in PAULI1]
PAULI_TO_CODE = {p: i for i, p in enumerate(PAULI2)}
CODE_TO_PAULI = {i: p for p, i in PAULI_TO_CODE.items()}

# Independent phase-twirled effective fidelity classes.
CX_TAU_PARAMETER_ORDER = [
    "IX", "ZI", "ZX", "XI", "IY", "XX", "XY", "ZY"
]

# Spectrum-only measured quantities.
CX_G_ORDER = ["g_IX", "g_ZI", "g_ZX", "g_XY", "g_A", "g_C"]

# g = A_CX tau, where tau follows CX_TAU_PARAMETER_ORDER.
A_CX = np.array([
    [1, 0, 0, 0,   0,   0, 0,   0],  # IX
    [0, 1, 0, 0,   0,   0, 0,   0],  # ZI
    [0, 0, 1, 0,   0,   0, 0,   0],  # ZX
    [0, 0, 0, 0,   0,   0, 1,   0],  # XY class
    [0, 0, 0, 0, 0.5,   0, 0, 0.5],  # A: (IY+ZY)/2
    [0, 0, 0, 0.5, 0, 0.5, 0,   0],  # C: (XI+XX)/2
], dtype=float)


# =============================================================================
# Pauli algebra and channel transforms
# =============================================================================

def _anticommutes_1q(a: str, b: str) -> int:
    if a == "I" or b == "I" or a == b:
        return 0
    return 1


def pauli_character(p: str, e: str) -> int:
    """Return +1 if two two-qubit Paulis commute and -1 if they anticommute."""
    parity = _anticommutes_1q(p[0], e[0]) ^ _anticommutes_1q(p[1], e[1])
    return -1 if parity else +1


def two_qubit_fidelities_from_probs(prob: Mapping[str, float]) -> Dict[str, float]:
    return {
        p: float(sum(pauli_character(p, e) * float(prob[e]) for e in PAULI2))
        for p in PAULI2
    }


def two_qubit_probs_from_fidelities(
    fidelities: Mapping[str, float],
    project_simplex: bool = True,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Inverse two-qubit Walsh-Hadamard transform.

    Returns ``(physical_or_raw, raw)``.  When ``project_simplex=True``, the first
    dictionary is the Euclidean simplex projection of the raw probabilities.
    """
    f = {p: float(fidelities[p]) for p in PAULI2}
    raw_vec = np.array([
        sum(pauli_character(p, e) * f[p] for p in PAULI2) / 16.0
        for e in PAULI2
    ], dtype=float)
    raw = {e: float(raw_vec[i]) for i, e in enumerate(PAULI2)}
    if project_simplex:
        vec = simplex_project(raw_vec)
    else:
        vec = raw_vec
    physical = {e: float(vec[i]) for i, e in enumerate(PAULI2)}
    return physical, raw


def inverse_two_qubit_pauli_quasiprobabilities_from_probs(
    prob: Mapping[str, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """Quasiprobability coefficients of the inverse two-qubit Pauli channel."""
    f = two_qubit_fidelities_from_probs(prob)
    inv_f = {}
    for p in PAULI2:
        value = float(f[p])
        if abs(value) < eps:
            value = eps if value >= 0 else -eps
        inv_f[p] = 1.0 / value
    return {
        e: float(sum(pauli_character(p, e) * inv_f[p] for p in PAULI2) / 16.0)
        for e in PAULI2
    }


def expand_twirled_tau(tau_reduced: Mapping[str, float]) -> Dict[str, float]:
    """Expand the eight phase-twirled effective fidelity classes to all Paulis."""
    t = {k: float(v) for k, v in tau_reduced.items()}
    out = {"II": 1.0}
    out.update({
        "IX": t["IX"],
        "ZI": t["ZI"],
        "ZX": t["ZX"],
        "XI": t["XI"], "YI": t["XI"],
        "IY": t["IY"], "IZ": t["IY"],
        "XX": t["XX"], "YX": t["XX"],
        "XY": t["XY"], "XZ": t["XY"],
        "YY": t["XY"], "YZ": t["XY"],
        "ZY": t["ZY"], "ZZ": t["ZY"],
    })
    return out


def reduced_tau_from_full(full: Mapping[str, float]) -> Dict[str, float]:
    return {p: float(full[p]) for p in CX_TAU_PARAMETER_ORDER}


def reconstruct_cx_tau_from_spectra(g: Mapping[str, float]) -> Dict[str, Any]:
    """Moore-Penrose CSB-only reconstruction of the eight tau classes."""
    b = np.array([float(g[name]) for name in CX_G_ORDER], dtype=float)
    tau_vec = np.linalg.pinv(A_CX) @ b
    reduced = {
        name: float(tau_vec[i])
        for i, name in enumerate(CX_TAU_PARAMETER_ORDER)
    }
    full_raw = expand_twirled_tau(reduced)
    prob, prob_raw = two_qubit_probs_from_fidelities(full_raw, project_simplex=True)
    full_physical = two_qubit_fidelities_from_probs(prob)
    reduced_physical = reduced_tau_from_full(full_physical)
    return {
        "A": A_CX.copy(),
        "b": b,
        "rank": int(np.linalg.matrix_rank(A_CX)),
        "singular_values": np.linalg.svd(A_CX, compute_uv=False),
        "tau_reduced_raw": reduced,
        "tau_full_raw": full_raw,
        "prob_raw": prob_raw,
        "prob_projected": prob,
        "tau_full_projected": full_physical,
        "tau_reduced_projected": reduced_physical,
        "residual_norm_raw": float(np.linalg.norm(A_CX @ tau_vec - b)),
    }


# CNOT label map used in beta_Q=alpha_{pi_C(Q)}.
_CNOT_PI = {
    "II": "II",
    "IX": "IX", "IY": "ZY", "IZ": "ZZ",
    "XI": "XX", "XX": "XI", "XY": "YZ", "XZ": "YY",
    "YI": "YX", "YX": "YI", "YY": "XZ", "YZ": "XY",
    "ZI": "ZI", "ZX": "ZX", "ZY": "IY", "ZZ": "IZ",
}


def beta_from_single_qubit_fidelities(
    control_f: Mapping[str, float],
    target_f: Optional[Mapping[str, float]] = None,
) -> Dict[str, float]:
    """
    Known local-layer contribution beta_Q=alpha_{pi_C(Q)}.

    ``control_f`` is the calibrated Pauli fidelity dictionary of RZ90 on the
    control block and ``target_f`` that of RX(-pi/2) on the target block.  If
    ``target_f`` is omitted, the shared channel used by the single-qubit paper
    workflow is assumed.
    """
    if target_f is None:
        target_f = control_f
    fc = {"I": 1.0, **{k: float(control_f[k]) for k in "XYZ"}}
    ft = {"I": 1.0, **{k: float(target_f[k]) for k in "XYZ"}}
    alpha = {a + b: fc[a] * ft[b] for a in PAULI1 for b in PAULI1}
    return {q: float(alpha[_CNOT_PI[q]]) for q in PAULI2}


def intrinsic_cx_from_effective_tau(
    tau_full: Mapping[str, float],
    beta: Mapping[str, float],
    project_simplex: bool = True,
) -> Dict[str, Any]:
    eta_raw = {p: float(tau_full[p]) / float(beta[p]) for p in PAULI2}
    eta_raw["II"] = 1.0
    prob, prob_raw = two_qubit_probs_from_fidelities(eta_raw, project_simplex=project_simplex)
    eta = two_qubit_fidelities_from_probs(prob) if project_simplex else eta_raw
    return {
        "eta_raw": eta_raw,
        "prob_raw": prob_raw,
        "prob_projected": prob,
        "eta_projected": eta,
    }


# =============================================================================
# CSB branch definitions
# =============================================================================

# branch_code, prep Pauli, measured Pauli, postprocessing sign, kind, ideal phase
CX_CSB_BRANCHES: Dict[str, Dict[str, Any]] = {
    "IX":     dict(code=0, prep="IX", measure="IX", sign=+1, kind="real", omega=0.0),
    "ZI":     dict(code=1, prep="ZI", measure="ZI", sign=+1, kind="real", omega=0.0),
    "ZX":     dict(code=2, prep="ZX", measure="ZX", sign=+1, kind="real", omega=0.0),
    "XY":     dict(code=3, prep="XY", measure="XY", sign=+1, kind="real_minus", omega=np.pi),
    # A: prepare IY; C_A=<IY>-i<ZZ> has ideal phase i^L.
    "A_real": dict(code=4, prep="IY", measure="IY", sign=+1, kind="complex_real", omega=np.pi/2),
    "A_imag": dict(code=5, prep="IY", measure="ZZ", sign=-1, kind="complex_imag", omega=np.pi/2),
    # C: prepare XI; C_C=<XI>+i<YX> has ideal phase i^L.
    "C_real": dict(code=6, prep="XI", measure="XI", sign=+1, kind="complex_real", omega=np.pi/2),
    "C_imag": dict(code=7, prep="XI", measure="YX", sign=+1, kind="complex_imag", omega=np.pi/2),
}

# Pauli code per qubit: 0 I, 1 X, 2 Y, 3 Z.
_P1_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}


def _twoq_code(label: str) -> int:
    return 4 * _P1_CODE[label[0]] + _P1_CODE[label[1]]


def _plus_init_code_for_pauli(label: str) -> int:
    if label in ("I", "Z"):
        return INIT_0
    if label == "X":
        return INIT_PLUS
    if label == "Y":
        return INIT_PLUS_I
    raise ValueError(label)


# =============================================================================
# Numba two-block trajectory kernel
# =============================================================================

if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _apply_binary_rz90(x_mask, z_mask):
        return x_mask, np.uint16(z_mask ^ x_mask)

    @nb.njit(cache=True)
    def _apply_binary_rx90(x_mask, z_mask):
        return np.uint16(x_mask ^ z_mask), z_mask

    @nb.njit(cache=True)
    def _apply_transversal_cx_and_ftec(
        xc, zc, xt, zt, state, noise_arr, fallback_on_unverified,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        p_1q, p_2q, p_meas, p_reset, p_idle, _p_final = noise_arr
        # Seven physical transversal CNOTs, each followed by the same two-qubit
        # depolarizing fault model used elsewhere in the simulator.
        for q in range(7):
            bit = np.uint16(1 << q)
            if xc & bit:
                xt = np.uint16(xt ^ bit)
            if zt & bit:
                zc = np.uint16(zc ^ bit)
            state, pc, pt = _sample_twoq_pauli(state, p_2q)
            if pc != 0:
                xc, zc = _apply_pauli_code(xc, zc, q, pc)
            if pt != 0:
                xt, zt = _apply_pauli_code(xt, zt, q, pt)

        xc, zc, state, unv_c, rounds_c = _run_ft_ec_numba(
            xc, zc, state,
            p_1q, p_2q, p_meas, p_reset, p_idle,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            fallback_on_unverified,
        )
        xt, zt, state, unv_t, rounds_t = _run_ft_ec_numba(
            xt, zt, state,
            p_1q, p_2q, p_meas, p_reset, p_idle,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            fallback_on_unverified,
        )
        return xc, zc, xt, zt, state, unv_c + unv_t, rounds_c + rounds_t

    @nb.njit(cache=True)
    def _apply_rz_power_binary(x_mask, z_mask, exponent):
        """Unsigned Pauli-frame action of RZ(pi/2)^exponent."""
        if (int(exponent) & 1) != 0:
            z_mask = np.uint16(z_mask ^ x_mask)
        return x_mask, z_mask

    @nb.njit(cache=True)
    def _apply_rx_power_binary(x_mask, z_mask, exponent):
        """Unsigned Pauli-frame action of RX(pi/2)^exponent."""
        if (int(exponent) & 1) != 0:
            x_mask = np.uint16(x_mask ^ z_mask)
        return x_mask, z_mask

    @nb.njit(cache=True)
    def _apply_final_cx_twirl_frame(xc, zc, xt, zt, tc, tt):
        """Absorb the final T_L^dagger into the measurement frame."""
        if int(tc) & 1:
            xc, zc = _apply_rz_power_binary(xc, zc, 3)
        if int(tt) & 1:
            xt, zt = _apply_rx_power_binary(xt, zt, 3)
        return xc, zc, xt, zt

    @nb.njit(cache=True)
    def _apply_compiled_predress_layer_and_ftec(
        xc, zc, xt, zt, state, noise_arr,
        previous_tc, previous_tt, current_tc, current_tt,
        fallback_on_unverified,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        """
        Apply A_k^comp = T_k A T_{k-1}^dagger as one local-Clifford slot.

        A = RZ_c(pi/2) tensor RX_t(-pi/2).  The virtual twirls are never
        separate physical gates.  Each nonidentity compiled Clifford receives
        one p_1q fault location; an identity compiled Clifford receives one
        logical-idle p_idle location.  One FTEC gadget follows on each block.
        """
        p_1q, p_2q, p_meas, p_reset, p_idle, _p_final = noise_arr

        m_c = (int(current_tc) + 1 - int(previous_tc)) & 3
        m_t = (int(current_tt) - 1 - int(previous_tt)) & 3

        xc, zc = _apply_rz_power_binary(xc, zc, m_c)
        xt, zt = _apply_rx_power_binary(xt, zt, m_t)

        if m_c == 0:
            xc, zc, state = _apply_idle_faults(xc, zc, state, p_idle)
        else:
            xc, zc, state = _apply_1q_gate_noise_all_data_numba(
                xc, zc, state, p_1q
            )

        if m_t == 0:
            xt, zt, state = _apply_idle_faults(xt, zt, state, p_idle)
        else:
            xt, zt, state = _apply_1q_gate_noise_all_data_numba(
                xt, zt, state, p_1q
            )

        xc, zc, state, unv_c, rounds_c = _run_ft_ec_numba(
            xc, zc, state,
            p_1q, p_2q, p_meas, p_reset, p_idle,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            fallback_on_unverified,
        )
        xt, zt, state, unv_t, rounds_t = _run_ft_ec_numba(
            xt, zt, state,
            p_1q, p_2q, p_meas, p_reset, p_idle,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            fallback_on_unverified,
        )
        return (
            xc, zc, xt, zt, state,
            unv_c + unv_t,
            rounds_c + rounds_t,
        )

    @nb.njit(cache=True)
    def _apply_one_predressed_cx_cycle(
        xc, zc, xt, zt, state, noise_arr,
        previous_tc, previous_tt, current_tc, current_tt,
        fallback_on_unverified,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        """Compiled local Clifford -> FTEC -> transversal CX -> FTEC."""
        xc, zc, xt, zt, state, unv_local, rounds_local = (
            _apply_compiled_predress_layer_and_ftec(
                xc, zc, xt, zt, state, noise_arr,
                previous_tc, previous_tt, current_tc, current_tt,
                fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
        )

        xc, zc, xt, zt, state, unv_cx, rounds_cx = (
            _apply_transversal_cx_and_ftec(
                xc, zc, xt, zt, state, noise_arr, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
        )
        return (
            xc, zc, xt, zt, state,
            unv_local + unv_cx,
            4,
            rounds_local + rounds_cx,
        )

    @nb.njit(cache=True)
    def _ideal_branch_mean(branch_code, depth):
        mod = depth & 3
        if branch_code <= 2:  # IX, ZI, ZX
            return 1.0
        if branch_code == 3:  # XY eigenphase -1
            return 1.0 if (depth & 1) == 0 else -1.0
        if branch_code == 4 or branch_code == 6:  # real cos
            if mod == 0: return 1.0
            if mod == 2: return -1.0
            return 0.0
        # A_imag and C_imag are signed so both equal sin(pi L/2).
        if mod == 1: return 1.0
        if mod == 3: return -1.0
        return 0.0

    @nb.njit(cache=True)
    def _branch_measurement_codes(branch_code):
        # return control measurement code, target measurement code
        if branch_code == 0: return 0, MEAS_X       # IX
        if branch_code == 1: return MEAS_Z, 0       # ZI
        if branch_code == 2: return MEAS_Z, MEAS_X  # ZX
        if branch_code == 3: return MEAS_X, MEAS_Y  # XY
        if branch_code == 4: return 0, MEAS_Y       # IY
        if branch_code == 5: return MEAS_Z, MEAS_Z  # ZZ, sign already in ideal mean
        if branch_code == 6: return MEAS_X, 0       # XI
        return MEAS_Y, MEAS_X                    # YX

    @nb.njit(cache=True)
    def _measure_two_block_pauli(
        xc, zc, xt, zt, state, branch_code, ideal_mean,
        noise_arr, measurement_basis_noisy,
    ):
        p_1q, _p_2q, _p_meas, _p_reset, _p_idle, p_final = noise_arr
        state, u = _rng_uniform01(state)
        ideal_out = 1 if u < 0.5 * (1.0 + ideal_mean) else -1
        mc, mt = _branch_measurement_codes(branch_code)
        flip = 0
        if mc != 0:
            state, f = _logical_measurement_flip_numba(
                xc, zc, state, mc, p_1q, p_final, measurement_basis_noisy
            )
            flip ^= f
        if mt != 0:
            state, f = _logical_measurement_flip_numba(
                xt, zt, state, mt, p_1q, p_final, measurement_basis_noisy
            )
            flip ^= f
        if flip & 1:
            ideal_out = -ideal_out
        return state, ideal_out

    @nb.njit(cache=True)
    def _simulate_cx_csb_point_kernel(
        branch_code, prep_init_c, prep_init_t, depth, shots, seed, noise_arr,
        prep_gates_noisy, prep_qec_after_gate,
        measurement_basis_noisy, fallback_on_unverified,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        total = 0.0
        total2 = 0.0
        unverified_total = 0
        ec_cycles_total = 0
        ec_rounds_total = 0
        p_1q, p_2q, p_meas, p_reset, p_idle, _p_final = noise_arr
        base_state = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)

        for shot in range(shots):
            state = base_state + np.uint64(shot) * np.uint64(0xD1B54A32D192ED03)
            state = _rng_next_u64(_rng_next_u64(state))
            xc = np.uint16(0); zc = np.uint16(0)
            xt = np.uint16(0); zt = np.uint16(0)

            xc, zc, state, u, e = _prepare_initial_frame_numba(
                prep_init_c, xc, zc, state,
                p_1q, p_2q, p_meas, p_reset, p_idle,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            unverified_total += u; ec_cycles_total += e
            xt, zt, state, u, e = _prepare_initial_frame_numba(
                prep_init_t, xt, zt, state,
                p_1q, p_2q, p_meas, p_reset, p_idle,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            unverified_total += u; ec_cycles_total += e

            previous_tc = 0
            previous_tt = 0
            for _ in range(depth):
                state, current_tc = _rng_int(state, 2)
                state, current_tt = _rng_int(state, 2)
                xc, zc, xt, zt, state, unv, ecc, ecr = (
                    _apply_one_predressed_cx_cycle(
                        xc, zc, xt, zt, state, noise_arr,
                        previous_tc, previous_tt, current_tc, current_tt,
                        fallback_on_unverified,
                        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                    )
                )
                unverified_total += unv
                ec_cycles_total += ecc
                ec_rounds_total += ecr
                previous_tc = current_tc
                previous_tt = current_tt

            # Final T_L^dagger is a virtual frame update absorbed into readout.
            if depth > 0:
                xc, zc, xt, zt = _apply_final_cx_twirl_frame(
                    xc, zc, xt, zt, previous_tc, previous_tt
                )

            ideal_mean = _ideal_branch_mean(branch_code, depth)
            state, out = _measure_two_block_pauli(
                xc, zc, xt, zt, state, branch_code, ideal_mean,
                noise_arr, measurement_basis_noisy,
            )
            total += out
            total2 += out * out

        if shots <= 0:
            return np.nan, np.nan, 0.0, 0.0
        mean = total / shots
        if shots > 1:
            var = max(0.0, (total2 - shots * mean * mean) / (shots - 1))
            stderr = math.sqrt(var / shots)
        else:
            stderr = 0.0
        unv_rate = float(unverified_total) / max(ec_cycles_total, 1)
        avg_rounds = float(ec_rounds_total) / max(ec_cycles_total, 1)
        return mean, stderr, unv_rate, avg_rounds

    @nb.njit(cache=True)
    def _apply_logical_twoq_pauli_masks(xc, zc, xt, zt, pcode):
        pc = pcode // 4
        pt = pcode % 4
        if pc == 1 or pc == 2:
            xc = np.uint16(xc ^ LOGICAL_MASK_7)
        if pc == 2 or pc == 3:
            zc = np.uint16(zc ^ LOGICAL_MASK_7)
        if pt == 1 or pt == 2:
            xt = np.uint16(xt ^ LOGICAL_MASK_7)
        if pt == 2 or pt == 3:
            zt = np.uint16(zt ^ LOGICAL_MASK_7)
        return xc, zc, xt, zt

    @nb.njit(cache=True)
    def _sample_categorical(state, cumulative):
        state, u = _rng_uniform01(state)
        for i in range(cumulative.shape[0]):
            if u <= cumulative[i]:
                return state, i
        return state, cumulative.shape[0] - 1

    @nb.njit(cache=True)
    def _measurement_codes_from_pauli_code(pcode):
        pc = pcode // 4
        pt = pcode % 4
        def conv(x):
            if x == 1: return MEAS_X
            if x == 2: return MEAS_Y
            if x == 3: return MEAS_Z
            return 0
        return conv(pc), conv(pt)

    @nb.njit(cache=True)
    def _measure_validation_pauli(
        xc, zc, xt, zt, state, pcode, noise_arr, measurement_basis_noisy,
    ):
        p_1q, _p_2q, _p_meas, _p_reset, _p_idle, p_final = noise_arr
        mc, mt = _measurement_codes_from_pauli_code(pcode)
        flip = 0
        if mc != 0:
            state, f = _logical_measurement_flip_numba(
                xc, zc, state, mc, p_1q, p_final, measurement_basis_noisy
            )
            flip ^= f
        if mt != 0:
            state, f = _logical_measurement_flip_numba(
                xt, zt, state, mt, p_1q, p_final, measurement_basis_noisy
            )
            flip ^= f
        out = -1 if (flip & 1) else +1
        return state, out

    @nb.njit(cache=True)
    def _simulate_cx_pec_point_kernel(
        validation_pcode, prep_init_c, prep_init_t, depth, shots, seed, noise_arr,
        prep_gates_noisy, prep_qec_after_gate,
        measurement_basis_noisy, fallback_on_unverified,
        use_pec, cumulative_q, q_sign, gamma,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        total = 0.0
        total2 = 0.0
        p_1q, p_2q, p_meas, p_reset, p_idle, _p_final = noise_arr
        base_state = np.uint64(seed) + np.uint64(0xA24BAED4963EE407)

        for shot in range(shots):
            state = base_state + np.uint64(shot) * np.uint64(0x9FB21C651E98DF25)
            state = _rng_next_u64(_rng_next_u64(state))
            xc = np.uint16(0); zc = np.uint16(0)
            xt = np.uint16(0); zt = np.uint16(0)
            weight = 1.0

            xc, zc, state, _u, _e = _prepare_initial_frame_numba(
                prep_init_c, xc, zc, state,
                p_1q, p_2q, p_meas, p_reset, p_idle,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            xt, zt, state, _u, _e = _prepare_initial_frame_numba(
                prep_init_t, xt, zt, state,
                p_1q, p_2q, p_meas, p_reset, p_idle,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )

            previous_tc = 0
            previous_tt = 0
            for _ in range(depth):
                state, current_tc = _rng_int(state, 2)
                state, current_tt = _rng_int(state, 2)
                xc, zc, xt, zt, state, _u, _e, _r = (
                    _apply_one_predressed_cx_cycle(
                        xc, zc, xt, zt, state, noise_arr,
                        previous_tc, previous_tt, current_tc, current_tt,
                        fallback_on_unverified,
                        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                    )
                )
                if use_pec:
                    state, idx = _sample_categorical(state, cumulative_q)
                    xc, zc, xt, zt = _apply_logical_twoq_pauli_masks(
                        xc, zc, xt, zt, idx
                    )
                    weight *= gamma * q_sign[idx]
                previous_tc = current_tc
                previous_tt = current_tt

            if depth > 0:
                xc, zc, xt, zt = _apply_final_cx_twirl_frame(
                    xc, zc, xt, zt, previous_tc, previous_tt
                )

            state, out = _measure_validation_pauli(
                xc, zc, xt, zt, state, validation_pcode,
                noise_arr, measurement_basis_noisy,
            )
            value = weight * out
            total += value
            total2 += value * value

        if shots <= 0:
            return np.nan, np.nan
        mean = total / shots
        if shots > 1:
            var = max(0.0, (total2 - shots * mean * mean) / (shots - 1))
            stderr = math.sqrt(var / shots)
        else:
            stderr = 0.0
        return mean, stderr


# =============================================================================
# Python wrappers and parallel spectrum runner
# =============================================================================

def _cx_noise_array(noise: NoiseParams) -> np.ndarray:
    return np.array([
        noise.p_1q, noise.p_2q, noise.p_meas,
        noise.p_reset, noise.p_idle, noise.p_final_meas,
    ], dtype=np.float64)


def _cx_compiled(decoder):
    return get_compiled_numba_data(decoder)


def warmup_cx_numba(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    seed: int,
) -> Tuple[float, float, float, float]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required by the logical-CX simulator.")
    return simulate_cx_csb_point(
        decoder, noise, options,
        "IX", depth=1, shots=2, seed=int(seed),
    )


def simulate_cx_csb_point(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    branch_name: str,
    depth: int,
    shots: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the two-block CX trajectory simulation.")
    data = _cx_compiled(decoder)
    branch = CX_CSB_BRANCHES[branch_name]
    prep = branch["prep"]
    return _simulate_cx_csb_point_kernel(
        int(branch["code"]),
        int(_plus_init_code_for_pauli(prep[0])),
        int(_plus_init_code_for_pauli(prep[1])),
        int(depth), int(shots), int(seed),
        _cx_noise_array(noise),
        bool(options["prep_gates_noisy"]),
        bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        data["op_type"], data["q1"], data["q2"], data["round_id"],
        data["basis_id"], data["check_id"], data["kind_id"],
        int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def _combine_chunks(sub: pd.DataFrame) -> Dict[str, Any]:
    shots = sub["shots"].to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    stderrs = sub["stderr"].to_numpy(dtype=float)
    n = int(np.sum(shots))
    mean = float(np.sum(shots * means) / max(n, 1))
    if n > 1:
        variances = (stderrs * np.sqrt(np.maximum(shots, 1.0))) ** 2
        ss_within = float(np.sum(np.maximum(shots - 1.0, 0.0) * variances))
        ss_between = float(np.sum(shots * (means - mean) ** 2))
        stderr = math.sqrt(max((ss_within + ss_between) / (n - 1), 0.0) / n)
    else:
        stderr = 0.0
    first = sub.iloc[0]
    return {
        "branch": first["branch"],
        "branch_kind": first["branch_kind"],
        "prep_pauli": first["prep_pauli"],
        "measure_pauli": first["measure_pauli"],
        "step": int(first["step"]),
        "rep": int(first["rep"]),
        "depth": int(first["depth"]),
        "mean": mean,
        "stderr": float(stderr),
        "shots": n,
        "requested_total_shots": int(first["requested_total_shots"]),
        "unverified_rate": float(np.sum(shots * sub["unverified_rate"]) / max(n, 1)),
        "average_ec_rounds": float(np.sum(shots * sub["average_ec_rounds"]) / max(n, 1)),
        "n_chunks": int(len(sub)),
    }


def run_cx_csb_spectrum(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    rep = int(rep)
    shots = int(shots)
    steps = [int(s) for s in steps]
    chunk_shots = max(1, int(chunk_shots))
    rng = np.random.default_rng(seed)
    tasks = []
    point_id = 0
    for branch_name, branch in CX_CSB_BRANCHES.items():
        branch_total = max(
            1, shots // 2 if branch["kind"].startswith("complex") else shots
        )
        for step in steps:
            depth = rep * step
            for chunk_id, n in enumerate(_chunk_sizes(branch_total, chunk_shots)):
                tasks.append({
                    "point_id": point_id, "chunk_id": chunk_id,
                    "branch": branch_name, "branch_kind": branch["kind"],
                    "prep_pauli": branch["prep"], "measure_pauli": branch["measure"],
                    "step": step, "rep": rep, "depth": depth,
                    "shots": n, "requested_total_shots": shots,
                    "seed": int(rng.integers(2**32 - 1)),
                })
            point_id += 1

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    if verbose:
        print("Logical-CX CSB")
        print("  steps       =", steps)
        print("  rep         =", rep)
        print("  true depths =", [rep * s for s in steps])
        print("  shots       =", shots)
        print("  n_jobs      =", n_jobs)

    def run_task(task):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        mean, se, unv, avg_rounds = simulate_cx_csb_point(
            dec, noise, options,
            task["branch"], task["depth"], task["shots"], task["seed"]
        )
        return {
            **task, "mean": mean, "stderr": se,
            "unverified_rate": unv, "average_ec_rounds": avg_rounds,
        }

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(run_task)(task) for task in tasks
        )
    else:
        rows = [run_task(task) for task in tasks]

    chunk_df = pd.DataFrame(rows)
    combined = [_combine_chunks(sub) for _, sub in chunk_df.groupby("point_id")]
    return pd.DataFrame(combined).sort_values(["branch", "step"]).reset_index(drop=True)


# =============================================================================
# Spectrum fitting, Pauli learning, and tables
# =============================================================================

def _fit_real_branch(csb_df: pd.DataFrame, branch: str, demodulate_minus: bool = False) -> Dict[str, float]:
    sub = csb_df[csb_df.branch == branch].sort_values("depth")
    L = sub.depth.to_numpy(dtype=float)
    y = sub["mean"].to_numpy(dtype=float)
    if demodulate_minus:
        y = y * ((-1.0) ** L.astype(int))
    fit = fit_axis_decay_signed_continuous(L, y, sub.stderr.to_numpy(dtype=float))
    return {
        "g": float(fit["f"]),
        "g_stderr": float(fit["f_stderr"]),
        "amplitude": float(fit["A"]),
        "success": bool(fit["success"]),
        "cost": float(fit["cost"]),
    }


def _complex_branch_table(csb_df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    real = csb_df[csb_df.branch == f"{prefix}_real"].sort_values("depth")
    imag = csb_df[csb_df.branch == f"{prefix}_imag"].sort_values("depth")
    m = real[["step", "rep", "depth", "mean", "stderr", "shots"]].merge(
        imag[["step", "rep", "depth", "mean", "stderr", "shots"]],
        on=["step", "rep", "depth"], suffixes=("_real", "_imag")
    )
    m["complex_signal"] = m["mean_real"].to_numpy() + 1j * m["mean_imag"].to_numpy()
    m["magnitude"] = np.abs(m["complex_signal"].to_numpy())
    return m


def fit_cx_csb_spectra(csb_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float]]:
    rows = []
    g = {}
    for branch, minus in [("IX", False), ("ZI", False), ("ZX", False), ("XY", True)]:
        fit = _fit_real_branch(csb_df, branch, demodulate_minus=minus)
        g[f"g_{branch}"] = fit["g"]
        rows.append({"mode": branch, "kind": "real", **fit})

    for prefix, label in [("A", "(IY+iZZ) block"), ("C", "(XI-iYX) block")]:
        q = _complex_branch_table(csb_df, prefix)

        # The CX learning equations only require the spectral magnitude.  Fit
        # |C_L|=A g^L directly so that even repetition factors (for example the
        # even repetition blocks do not alias away the imaginary quadrature.
        xr = q["mean_real"].to_numpy(dtype=float)
        yi = q["mean_imag"].to_numpy(dtype=float)
        sr = q["stderr_real"].to_numpy(dtype=float)
        si = q["stderr_imag"].to_numpy(dtype=float)
        mag = np.sqrt(xr*xr + yi*yi)
        mag_safe = np.maximum(mag, 1e-15)
        mag_se = np.sqrt((xr/mag_safe)**2 * sr**2 + (yi/mag_safe)**2 * si**2)
        mag_fit = fit_axis_decay_signed_continuous(
            q["depth"].to_numpy(dtype=float), mag, mag_se
        )

        # Keep the full complex fit as an angle diagnostic when it is
        # well-conditioned, but never use it to replace a finite magnitude fit.
        complex_fit = fit_complex_csb_eigenvalue(q, omega0=np.pi/2)
        g_value = float(mag_fit["f"])
        g_stderr = float(mag_fit["f_stderr"])
        if not np.isfinite(g_value):
            g_value = float(complex_fit["g"])
            g_stderr = float(complex_fit["g_stderr"])

        g[f"g_{prefix}"] = g_value
        rows.append({
            "mode": prefix,
            "kind": "complex",
            "description": label,
            "g": g_value,
            "g_stderr": g_stderr,
            "delta_theta": float(complex_fit["delta_theta"]),
            "delta_theta_stderr": float(complex_fit["delta_theta_stderr"]),
            "lambda_real": float(complex_fit["lambda_real"]),
            "lambda_imag": float(complex_fit["lambda_imag"]),
            "success": bool(mag_fit["success"]),
            "cost": float(mag_fit["cost"]),
        })
    return pd.DataFrame(rows), g


def learning_tables_from_cx_csb(
    csb_df: pd.DataFrame,
    beta: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    spectra_df, g = fit_cx_csb_spectra(csb_df)
    recon = reconstruct_cx_tau_from_spectra(g)

    tau_full = recon["tau_full_projected"]
    tau_prob = recon["prob_projected"]
    tau_q = inverse_two_qubit_pauli_quasiprobabilities_from_probs(tau_prob)
    gamma_tau = float(sum(abs(v) for v in tau_q.values()))

    tau_rows = []
    for p in PAULI2:
        tau_rows.append({
            "pauli": p,
            "tau_raw": float(recon["tau_full_raw"][p]),
            "tau_projected": float(tau_full[p]),
            "probability_raw": float(recon["prob_raw"][p]),
            "probability_projected": float(tau_prob[p]),
            "inverse_q": float(tau_q[p]),
        })
    tau_df = pd.DataFrame(tau_rows)

    intrinsic = None
    eta_df = pd.DataFrame()
    if beta is not None:
        intrinsic = intrinsic_cx_from_effective_tau(tau_full, beta, project_simplex=True)
        eta_q = inverse_two_qubit_pauli_quasiprobabilities_from_probs(intrinsic["prob_projected"])
        eta_rows = []
        for p in PAULI2:
            eta_rows.append({
                "pauli": p,
                "beta": float(beta[p]),
                "eta_raw": float(intrinsic["eta_raw"][p]),
                "eta_projected": float(intrinsic["eta_projected"][p]),
                "probability_raw": float(intrinsic["prob_raw"][p]),
                "probability_projected": float(intrinsic["prob_projected"][p]),
                "inverse_q": float(eta_q[p]),
            })
        eta_df = pd.DataFrame(eta_rows)

    summary = pd.DataFrame([{
        "reconstruction_rank": recon["rank"],
        "reconstruction_nullity": len(CX_TAU_PARAMETER_ORDER) - recon["rank"],
        "reconstruction_residual_norm": recon["residual_norm_raw"],
        "effective_process_fidelity_pII": tau_prob["II"],
        "effective_process_infidelity": 1.0 - tau_prob["II"],
        "effective_gamma_per_cycle": gamma_tau,
        "effective_sampling_overhead_per_cycle": gamma_tau ** 2,
    }])

    return {
        "spectra_df": spectra_df,
        "g": g,
        "reconstruction": recon,
        "tau_df": tau_df,
        "tau_full": tau_full,
        "tau_prob": tau_prob,
        "tau_inverse_q": tau_q,
        "eta_df": eta_df,
        "intrinsic": intrinsic,
        "summary_df": summary,
    }


# =============================================================================
# PEC validation
# =============================================================================

def _prepare_inverse_sampling_arrays(inverse_q: Mapping[str, float]) -> Tuple[np.ndarray, np.ndarray, float]:
    q = np.array([float(inverse_q[p]) for p in PAULI2], dtype=float)
    gamma = float(np.sum(np.abs(q)))
    probs = np.abs(q) / gamma
    cumulative = np.cumsum(probs)
    cumulative[-1] = 1.0
    signs = np.where(q >= 0, 1.0, -1.0)
    return cumulative.astype(np.float64), signs.astype(np.float64), gamma


def simulate_cx_pec_point(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    validation_pauli: str,
    depth: int,
    shots: int,
    seed: int,
    inverse_q: Optional[Mapping[str, float]],
) -> Tuple[float, float]:
    data = _cx_compiled(decoder)
    if inverse_q is None:
        cumulative = np.ones(16, dtype=np.float64)
        signs = np.ones(16, dtype=np.float64)
        gamma = 1.0
        use_pec = False
    else:
        cumulative, signs, gamma = _prepare_inverse_sampling_arrays(inverse_q)
        use_pec = True
    return _simulate_cx_pec_point_kernel(
        int(_twoq_code(validation_pauli)),
        int(_plus_init_code_for_pauli(validation_pauli[0])),
        int(_plus_init_code_for_pauli(validation_pauli[1])),
        int(depth), int(shots), int(seed),
        _cx_noise_array(noise),
        bool(options["prep_gates_noisy"]),
        bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        bool(use_pec), cumulative, signs, float(gamma),
        data["op_type"], data["q1"], data["q2"], data["round_id"],
        data["basis_id"], data["check_id"], data["kind_id"],
        int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def run_cx_pec_validation(
    decoder,
    inverse_q: Mapping[str, float],
    noise: NoiseParams,
    options: Mapping[str, bool],
    depths: Sequence[int],
    validation_paulis: Sequence[str],
    shots_per_pauli: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    depths = [int(d) for d in depths]
    if any(d % 4 != 0 for d in depths):
        raise ValueError("CX PEC validation depths must be multiples of four.")
    rng = np.random.default_rng(seed)
    tasks = []
    for depth in depths:
        for pauli in validation_paulis:
            for mode in ("noisy", "pec"):
                for chunk_id, n in enumerate(_chunk_sizes(shots_per_pauli, chunk_shots)):
                    tasks.append({
                        "depth": depth, "pauli": pauli, "mode": mode,
                        "shots": n, "chunk_id": chunk_id,
                        "seed": int(rng.integers(2**32 - 1)),
                    })

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    def run_task(task):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        inv = inverse_q if task["mode"] == "pec" else None
        mean, se = simulate_cx_pec_point(
            dec, noise, options,
            task["pauli"], task["depth"], task["shots"], task["seed"], inv
        )
        n = int(task["shots"])
        sample_var = se ** 2 * n if n > 1 else 0.0
        sumsq = sample_var * max(n - 1, 0) + n * mean ** 2
        return {**task, "sum": mean * n, "sumsq": sumsq}

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(run_task)(task) for task in tasks
        )
    else:
        rows = [run_task(task) for task in tasks]

    chunk_df = pd.DataFrame(rows)
    per_rows = []
    for (depth, pauli, mode), sub in chunk_df.groupby(["depth", "pauli", "mode"]):
        n = int(sub.shots.sum())
        s = float(sub["sum"].sum())
        ss = float(sub["sumsq"].sum())
        mean = s / max(n, 1)
        var = max((ss - s * s / n) / (n - 1), 0.0) if n > 1 else 0.0
        per_rows.append({
            "depth": int(depth), "pauli": pauli, "mode": mode,
            "expectation": mean,
            "stderr": math.sqrt(var / n) if n > 0 else np.nan,
            "shots": n,
        })
    per_pauli_df = pd.DataFrame(per_rows)

    summary_rows = []
    for depth in depths:
        row = {"depth": depth, "ideal_expectation": 1.0}
        for mode in ("noisy", "pec"):
            sub = per_pauli_df[(per_pauli_df.depth == depth) & (per_pauli_df["mode"] == mode)]
            vals = sub.expectation.to_numpy(dtype=float)
            row[f"{mode}_expectation"] = float(np.mean(vals))
            row[f"{mode}_std_across_paulis"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary_rows.append(row)
    return pd.DataFrame(summary_rows), per_pauli_df


# =============================================================================
# Compiled random local-Clifford + CX validation and CX workflow helpers
# =============================================================================

CX_PULSE_RX90, CX_PULSE_RXM90, CX_PULSE_RY90, CX_PULSE_RYM90 = 0, 1, 2, 3
CX_PULSE_RZ90, CX_PULSE_RZM90 = 4, 5
CX_PULSE_RX180, CX_PULSE_RY180, CX_PULSE_RZ180, CX_PULSE_IDLE = 6, 7, 8, 9


def _cx_clifford_key(matrix):
    return tuple(np.asarray(matrix, dtype=np.int8).reshape(-1))


def _build_cx_clifford_compiler():
    I = np.eye(3, dtype=np.int8)
    RX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int8)
    RY = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8)
    RZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8)
    native = [RX, RX.T, RY, RY.T, RZ, RZ.T, RX @ RX, RY @ RY, RZ @ RZ]

    mats, words, index, queue = [I], [[]], {_cx_clifford_key(I): 0}, [0]
    while queue:
        current = queue.pop(0)
        for pulse, gate in enumerate(native):
            candidate = np.asarray(gate @ mats[current], dtype=np.int8)
            key = _cx_clifford_key(candidate)
            if key not in index:
                index[key] = len(mats)
                mats.append(candidate)
                words.append(words[current] + [pulse])
                queue.append(len(mats) - 1)

    if len(mats) != 24 or max(map(len, words)) > 2:
        raise RuntimeError("Unexpected one-qubit Clifford compilation.")

    mats = np.stack(mats)
    lengths = np.asarray([len(word) for word in words], dtype=np.int8)
    word_array = np.full((24, 2), CX_PULSE_IDLE, dtype=np.int8)
    for i, word in enumerate(words):
        word_array[i, :len(word)] = word

    multiply = np.empty((24, 24), dtype=np.int8)
    inverse = np.empty(24, dtype=np.int8)
    for i in range(24):
        inverse[i] = index[_cx_clifford_key(mats[i].T)]
        for j in range(24):
            multiply[i, j] = index[_cx_clifford_key(mats[i] @ mats[j])]

    gate_mats = {
        "I": I, "RX90": RX, "RY90": RY, "RZ90": RZ,
        "RXM90": RX.T, "RYM90": RY.T, "RZM90": RZ.T,
        "RX180": RX @ RX, "RY180": RY @ RY, "RZ180": RZ @ RZ,
    }
    gate_ids = {name: index[_cx_clifford_key(mat)] for name, mat in gate_mats.items()}
    return mats, word_array, lengths, multiply, inverse, gate_ids


(
    CX_CLIFFORD_MATS,
    CX_CLIFFORD_WORDS,
    CX_CLIFFORD_LENGTHS,
    CX_CLIFFORD_MUL,
    CX_CLIFFORD_INV,
    CX_CLIFFORD_GATE_IDS,
) = _build_cx_clifford_compiler()
CX_CLIFFORD_IDENTITY = int(CX_CLIFFORD_GATE_IDS["I"])
CX_TWIRL_CONTROL_IDS = np.asarray(
    [CX_CLIFFORD_IDENTITY, CX_CLIFFORD_GATE_IDS["RZ90"]], dtype=np.int8
)
CX_TWIRL_TARGET_IDS = np.asarray(
    [CX_CLIFFORD_IDENTITY, CX_CLIFFORD_GATE_IDS["RX90"]], dtype=np.int8
)


# -----------------------------------------------------------------------------
# Global two-qubit Clifford inverse compiler
# -----------------------------------------------------------------------------
#
# A depth-d validation circuit contains d layers of the form
#
#     local Clifford -> CX.
#
# The exact inverse of the *complete* two-qubit Clifford is compiled globally,
# rather than by reversing all d layers one by one.  Any two-qubit Clifford can
# be synthesized with at most three CX gates and local Cliffords.  We build a
# small exact compiler for the 11,520-element two-qubit Clifford group, using
# signed Pauli permutations as an integer representation.  Local Clifford
# generators have zero CX cost and CX has unit cost, so a 0-1 BFS returns a
# minimum-CX decomposition.

_CX_TWOQ_NONIDENTITY = tuple(p for p in PAULI2 if p != "II")
_CX_TWOQ_PAULI_INDEX = {p: i + 1 for i, p in enumerate(_CX_TWOQ_NONIDENTITY)}
_CX_TWOQ_IDENTITY_ACTION = tuple(range(1, 16))
_CX_AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def _cx_twoq_local_action_key(control_id: int, target_id: int) -> Tuple[int, ...]:
    """Signed Pauli permutation for C_control tensor C_target."""
    out = []
    for label in _CX_TWOQ_NONIDENTITY:
        sign = 1
        mapped = []
        for pauli, clifford_id in ((label[0], control_id), (label[1], target_id)):
            if pauli == "I":
                mapped.append("I")
                continue
            matrix = CX_CLIFFORD_MATS[int(clifford_id)]
            column = _CX_AXIS_INDEX[pauli]
            rows = np.flatnonzero(matrix[:, column])
            if len(rows) != 1:
                raise RuntimeError("Invalid one-qubit Clifford action.")
            row = int(rows[0])
            sign *= int(matrix[row, column])
            mapped.append("XYZ"[row])
        out_label = "".join(mapped)
        out.append(sign * _CX_TWOQ_PAULI_INDEX[out_label])
    return tuple(out)


def _cx_build_cx_action_key() -> Tuple[int, ...]:
    """Signed Pauli permutation for a CX with control=first, target=second."""
    i2 = np.eye(2, dtype=np.complex128)
    x2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    oneq = {"I": i2, "X": x2, "Y": y2, "Z": z2}
    basis = [np.kron(oneq[p[0]], oneq[p[1]]) for p in _CX_TWOQ_NONIDENTITY]
    cx = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=np.complex128,
    )
    out = []
    for pauli in basis:
        transformed = cx @ pauli @ cx.conj().T
        coeff = np.asarray(
            [np.trace(candidate.conj().T @ transformed) / 4.0 for candidate in basis]
        )
        index = int(np.argmax(np.abs(coeff)))
        value = coeff[index]
        if not np.isclose(abs(value), 1.0, atol=1e-10):
            raise RuntimeError("Could not identify the CX Pauli action.")
        sign = +1 if float(np.real(value)) > 0 else -1
        out.append(sign * (index + 1))
    return tuple(out)


CX_TWOQ_CX_ACTION = _cx_build_cx_action_key()
CX_TWOQ_LOCAL_ACTION = {
    (c, t): _cx_twoq_local_action_key(c, t)
    for c in range(24)
    for t in range(24)
}


def _cx_compose_twoq_actions(
    left: Tuple[int, ...],
    right: Tuple[int, ...],
) -> Tuple[int, ...]:
    """Return the signed Pauli action of left after right."""
    out = []
    for value in right:
        sign = +1 if value > 0 else -1
        mapped = left[abs(int(value)) - 1]
        out.append(sign * int(mapped))
    return tuple(out)


def _cx_inverse_twoq_action(action: Tuple[int, ...]) -> Tuple[int, ...]:
    """Inverse of a signed Pauli permutation."""
    out = [0] * len(action)
    for source, value in enumerate(action):
        sign = +1 if value > 0 else -1
        target = abs(int(value)) - 1
        out[target] = sign * (source + 1)
    return tuple(out)


def _build_cx_global_inverse_compiler():
    from collections import deque

    identity = CX_CLIFFORD_IDENTITY
    generators = []
    for qubit in (0, 1):
        for name in ("RX90", "RY90", "RZ90"):
            gate_id = int(CX_CLIFFORD_GATE_IDS[name])
            if qubit == 0:
                action = CX_TWOQ_LOCAL_ACTION[(gate_id, identity)]
            else:
                action = CX_TWOQ_LOCAL_ACTION[(identity, gate_id)]
            generators.append((("local", qubit, gate_id), action, 0))
    generators.append((("cx",), CX_TWOQ_CX_ACTION, 1))

    start = _CX_TWOQ_IDENTITY_ACTION
    distance = {start: 0}
    parent = {start: None}
    parent_operation = {}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        current_distance = distance[current]
        for operation, generator_action, cx_cost in generators:
            candidate = _cx_compose_twoq_actions(generator_action, current)
            new_distance = current_distance + cx_cost
            if candidate not in distance or new_distance < distance[candidate]:
                distance[candidate] = new_distance
                parent[candidate] = current
                parent_operation[candidate] = operation
                if cx_cost == 0:
                    queue.appendleft(candidate)
                else:
                    queue.append(candidate)

    if len(distance) != 11520:
        raise RuntimeError(
            f"Two-qubit Clifford compiler generated {len(distance)} elements, expected 11520."
        )
    if max(distance.values()) > 3:
        raise RuntimeError("Two-qubit Clifford inverse compiler exceeded three CX gates.")
    return parent, parent_operation, distance


(
    CX_TWOQ_COMPILER_PARENT,
    CX_TWOQ_COMPILER_OPERATION,
    CX_TWOQ_COMPILER_CX_COUNT,
) = _build_cx_global_inverse_compiler()


def _cx_twoq_compiler_path(action: Tuple[int, ...]) -> List[Tuple[Any, ...]]:
    """Chronological primitive path for one two-qubit Clifford action."""
    if action not in CX_TWOQ_COMPILER_PARENT:
        raise ValueError("Unknown two-qubit Clifford action.")
    path = []
    current = action
    while CX_TWOQ_COMPILER_PARENT[current] is not None:
        path.append(CX_TWOQ_COMPILER_OPERATION[current])
        current = CX_TWOQ_COMPILER_PARENT[current]
    path.reverse()
    return path


def _cx_compress_twoq_compiler_path(path: Sequence[Tuple[Any, ...]]) -> Dict[str, Any]:
    """Compress primitive local generators between CX gates into local Cliffords."""
    local_c = int(CX_CLIFFORD_IDENTITY)
    local_t = int(CX_CLIFFORD_IDENTITY)
    pre_c = []
    pre_t = []

    for operation in path:
        if operation[0] == "cx":
            pre_c.append(local_c)
            pre_t.append(local_t)
            local_c = int(CX_CLIFFORD_IDENTITY)
            local_t = int(CX_CLIFFORD_IDENTITY)
            continue

        _tag, qubit, gate_id = operation
        if int(qubit) == 0:
            local_c = int(CX_CLIFFORD_MUL[int(gate_id), local_c])
        else:
            local_t = int(CX_CLIFFORD_MUL[int(gate_id), local_t])

    return {
        "inverse_pre_c_ids": np.asarray(pre_c, dtype=np.int8),
        "inverse_pre_t_ids": np.asarray(pre_t, dtype=np.int8),
        "inverse_num_cx": int(len(pre_c)),
        "inverse_final_c_id": int(local_c),
        "inverse_final_t_id": int(local_t),
    }


def _cx_action_from_compiled_decomposition(
    pre_c_ids: Sequence[int],
    pre_t_ids: Sequence[int],
    final_c_id: int,
    final_t_id: int,
) -> Tuple[int, ...]:
    """Exact signed-Pauli action of local-CX-...-local decomposition."""
    action = _CX_TWOQ_IDENTITY_ACTION
    for control_id, target_id in zip(pre_c_ids, pre_t_ids):
        local_action = CX_TWOQ_LOCAL_ACTION[(int(control_id), int(target_id))]
        layer_action = _cx_compose_twoq_actions(CX_TWOQ_CX_ACTION, local_action)
        action = _cx_compose_twoq_actions(layer_action, action)
    final_action = CX_TWOQ_LOCAL_ACTION[(int(final_c_id), int(final_t_id))]
    return _cx_compose_twoq_actions(final_action, action)


def compile_global_inverse_cx_sequence(
    control_ids: Sequence[int],
    target_ids: Sequence[int],
) -> Dict[str, Any]:
    """
    Globally compile the exact inverse of a random local-Clifford-plus-CX circuit.

    The returned inverse has the chronological form

        L_0 -> CX -> L_1 -> CX -> ... -> L_{m-1} -> CX -> L_final,

    with m <= 3.  ``L_final`` is intended to be absorbed into the final
    measurement basis, so only the m compiled CX layers are physically run.
    """
    control_ids = np.asarray(control_ids, dtype=np.int8)
    target_ids = np.asarray(target_ids, dtype=np.int8)
    if control_ids.shape != target_ids.shape:
        raise ValueError("control_ids and target_ids must have the same shape.")

    forward_action = _CX_TWOQ_IDENTITY_ACTION
    for control_id, target_id in zip(control_ids, target_ids):
        local_action = CX_TWOQ_LOCAL_ACTION[(int(control_id), int(target_id))]
        layer_action = _cx_compose_twoq_actions(CX_TWOQ_CX_ACTION, local_action)
        forward_action = _cx_compose_twoq_actions(layer_action, forward_action)

    inverse_action = _cx_inverse_twoq_action(forward_action)
    path = _cx_twoq_compiler_path(inverse_action)
    compiled = _cx_compress_twoq_compiler_path(path)

    if compiled["inverse_num_cx"] > 3:
        raise RuntimeError("Globally compiled inverse contains more than three CX gates.")

    check = _cx_action_from_compiled_decomposition(
        compiled["inverse_pre_c_ids"],
        compiled["inverse_pre_t_ids"],
        compiled["inverse_final_c_id"],
        compiled["inverse_final_t_id"],
    )
    if check != inverse_action:
        raise RuntimeError("Global inverse compilation failed exact Clifford verification.")

    return compiled



def _cx_local_gate_ids(local_gate_set):
    return np.asarray([CX_CLIFFORD_GATE_IDS[g] for g in local_gate_set], dtype=np.int8)


def generate_random_cx_sequences(depths, sequences_per_depth, seed, local_gate_set):
    """Generate random forward circuits and globally compile each exact inverse."""
    gate_ids = _cx_local_gate_ids(local_gate_set)
    rng = np.random.default_rng(int(seed))
    records = []
    for depth in map(int, depths):
        for sequence_id in range(int(sequences_per_depth)):
            control_ids = rng.choice(gate_ids, size=depth).astype(np.int8)
            target_ids = rng.choice(gate_ids, size=depth).astype(np.int8)
            inverse = compile_global_inverse_cx_sequence(control_ids, target_ids)
            records.append({
                "depth": depth,
                "sequence_id": sequence_id,
                "control_ids": control_ids,
                "target_ids": target_ids,
                **inverse,
                "total_composite_cx_layers": depth + int(inverse["inverse_num_cx"]),
            })
    return records


if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _cx_apply_native_pulse(x_mask, z_mask, pulse_code, state, noise_arr):
        p_1q = noise_arr[0]
        p_idle = noise_arr[4]
        pulse = int(pulse_code)
        if pulse == CX_PULSE_IDLE:
            return _apply_idle_faults(x_mask, z_mask, state, p_idle)
        if pulse == CX_PULSE_RX90 or pulse == CX_PULSE_RXM90:
            x_mask = np.uint16(x_mask ^ z_mask)
        elif pulse == CX_PULSE_RY90 or pulse == CX_PULSE_RYM90:
            tmp = x_mask; x_mask = z_mask; z_mask = tmp
        elif pulse == CX_PULSE_RZ90 or pulse == CX_PULSE_RZM90:
            z_mask = np.uint16(z_mask ^ x_mask)
        return _apply_1q_gate_noise_all_data_numba(x_mask, z_mask, state, p_1q)

    @nb.njit(cache=True)
    def _cx_apply_local_clifford_and_ec(
        xc, zc, xt, zt, clifford_c, clifford_t, state, noise_arr,
        fallback_on_unverified, clifford_words, clifford_lengths,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        n_c = int(clifford_lengths[int(clifford_c)])
        n_t = int(clifford_lengths[int(clifford_t)])
        n_slots = n_c if n_c >= n_t else n_t
        if n_slots == 0:
            n_slots = 1
        for slot in range(n_slots):
            pc = CX_PULSE_IDLE if slot >= n_c else int(clifford_words[int(clifford_c), slot])
            pt = CX_PULSE_IDLE if slot >= n_t else int(clifford_words[int(clifford_t), slot])
            xc, zc, state = _cx_apply_native_pulse(xc, zc, pc, state, noise_arr)
            xt, zt, state = _cx_apply_native_pulse(xt, zt, pt, state, noise_arr)

        xc, zc, state, unv_c, rounds_c = _run_one_ftec(
            xc, zc, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        xt, zt, state, unv_t, rounds_t = _run_one_ftec(
            xt, zt, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        return xc, zc, xt, zt, state, unv_c + unv_t, rounds_c + rounds_t, n_slots

    @nb.njit(cache=True)
    def _cx_compiled_local_id(current_twirl_id, local_id, previous_twirl_id, multiply, inverse):
        return int(multiply[int(current_twirl_id), int(multiply[int(local_id), int(inverse[int(previous_twirl_id)])])])

    @nb.njit(cache=True)
    def _cx_apply_random_layer(
        xc, zc, xt, zt, state, noise_arr, local_c, local_t,
        previous_twirl_c_bit, previous_twirl_t_bit,
        current_twirl_c_bit, current_twirl_t_bit,
        fallback_on_unverified, clifford_words, clifford_lengths,
        multiply, inverse, twirl_control_ids, twirl_target_ids,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        previous_c = int(twirl_control_ids[int(previous_twirl_c_bit)])
        previous_t = int(twirl_target_ids[int(previous_twirl_t_bit)])
        current_c = int(twirl_control_ids[int(current_twirl_c_bit)])
        current_t = int(twirl_target_ids[int(current_twirl_t_bit)])
        compiled_c = _cx_compiled_local_id(current_c, local_c, previous_c, multiply, inverse)
        compiled_t = _cx_compiled_local_id(current_t, local_t, previous_t, multiply, inverse)
        xc, zc, xt, zt, state, unv_local, rounds_local, slots = _cx_apply_local_clifford_and_ec(
            xc, zc, xt, zt, compiled_c, compiled_t, state, noise_arr,
            fallback_on_unverified, clifford_words, clifford_lengths,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        xc, zc, xt, zt, state, unv_cx, rounds_cx = _apply_transversal_cx_and_ftec(
            xc, zc, xt, zt, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        return xc, zc, xt, zt, state, unv_local + unv_cx, rounds_local + rounds_cx, slots

    @nb.njit(cache=True)
    def _cx_measurement_code_from_axis(axis_code):
        if int(axis_code) == 1: return MEAS_X
        if int(axis_code) == 2: return MEAS_Y
        return MEAS_Z

    @nb.njit(cache=True)
    def _cx_positive_axis_after_local_clifford(pauli_axis_code, clifford_id, clifford_mats):
        axis = int(pauli_axis_code) - 1
        matrix = clifford_mats[int(clifford_id)]
        for row in range(3):
            if matrix[row, axis] != 0:
                return row + 1
        return 3

    @nb.njit(cache=True)
    def _simulate_random_cx_circuit_kernel(
        forward_c_ids, forward_t_ids,
        inverse_pre_c_ids, inverse_pre_t_ids, inverse_num_cx,
        inverse_final_c_id, inverse_final_t_id,
        shots, seed, noise_arr,
        measurement_basis_noisy, fallback_on_unverified,
        use_pec, cumulative_q, q_sign, gamma,
        clifford_identity, clifford_mats, clifford_words, clifford_lengths,
        mul_table, inv_table, twirl_control_ids, twirl_target_ids,
        op_type, q1, q2, round_id, basis_id, check_id, kind_id,
        min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
    ):
        total = 0.0
        total2 = 0.0
        pulse_slots_total = 0.0
        depth = int(forward_c_ids.shape[0])
        base_state = np.uint64(seed) + np.uint64(0xA24BAED4963EE407)

        for shot in range(int(shots)):
            state = base_state + np.uint64(shot) * np.uint64(0x9FB21C651E98DF25)
            state = _rng_next_u64(_rng_next_u64(state))
            xc = np.uint16(0); zc = np.uint16(0)
            xt = np.uint16(0); zt = np.uint16(0)
            weight = 1.0
            previous_tc = 0; previous_tt = 0

            for layer in range(depth):
                state, current_tc = _rng_int(state, 2)
                state, current_tt = _rng_int(state, 2)
                xc, zc, xt, zt, state, _u, _r, slots = _cx_apply_random_layer(
                    xc, zc, xt, zt, state, noise_arr,
                    int(forward_c_ids[layer]), int(forward_t_ids[layer]),
                    previous_tc, previous_tt, current_tc, current_tt,
                    fallback_on_unverified, clifford_words, clifford_lengths,
                    mul_table, inv_table, twirl_control_ids, twirl_target_ids,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )
                pulse_slots_total += slots
                previous_tc = current_tc; previous_tt = current_tt
                if use_pec:
                    state, idx = _sample_categorical(state, cumulative_q)
                    xc, zc, xt, zt = _apply_logical_twoq_pauli_masks(xc, zc, xt, zt, idx)
                    weight *= gamma * q_sign[idx]

            # Apply the globally compiled exact inverse.  Unlike the old
            # implementation, this does not reverse all ``depth`` layers.
            # The inverse contains at most three additional CX layers.
            for inv_index in range(int(inverse_num_cx)):
                local_c = int(inverse_pre_c_ids[inv_index])
                local_t = int(inverse_pre_t_ids[inv_index])
                state, current_tc = _rng_int(state, 2)
                state, current_tt = _rng_int(state, 2)
                xc, zc, xt, zt, state, _u, _r, slots = _cx_apply_random_layer(
                    xc, zc, xt, zt, state, noise_arr, local_c, local_t,
                    previous_tc, previous_tt, current_tc, current_tt,
                    fallback_on_unverified, clifford_words, clifford_lengths,
                    mul_table, inv_table, twirl_control_ids, twirl_target_ids,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )
                pulse_slots_total += slots
                previous_tc = current_tc; previous_tt = current_tt
                if use_pec:
                    state, idx = _sample_categorical(state, cumulative_q)
                    xc, zc, xt, zt = _apply_logical_twoq_pauli_masks(xc, zc, xt, zt, idx)
                    weight *= gamma * q_sign[idx]

            # The compiler leaves one final local Clifford after the last CX.
            # It is absorbed into the measurement basis together with the
            # final virtual-twirl frame.  ``frame`` is the dagger of the
            # omitted final correction, which is exactly what is needed to
            # transform the measured Z observable.
            frame_c = int(
                mul_table[
                    int(twirl_control_ids[int(previous_tc)]),
                    int(inv_table[int(inverse_final_c_id)]),
                ]
            )
            frame_t = int(
                mul_table[
                    int(twirl_target_ids[int(previous_tt)]),
                    int(inv_table[int(inverse_final_t_id)]),
                ]
            )

            mc = _cx_measurement_code_from_axis(
                _cx_positive_axis_after_local_clifford(3, frame_c, clifford_mats)
            )
            mt = _cx_measurement_code_from_axis(
                _cx_positive_axis_after_local_clifford(3, frame_t, clifford_mats)
            )
            p1 = noise_arr[0]; pf = noise_arr[5]
            state, fc = _logical_measurement_flip_numba(xc, zc, state, mc, p1, pf, measurement_basis_noisy)
            state, ft = _logical_measurement_flip_numba(xt, zt, state, mt, p1, pf, measurement_basis_noisy)
            indicator = 1.0 if int(fc) == 0 and int(ft) == 0 else 0.0
            value = weight * indicator
            total += value; total2 += value * value

        if shots <= 0:
            return np.nan, np.nan, np.nan
        mean = total / shots
        if shots > 1:
            var = max((total2 - shots * mean * mean) / (shots - 1), 0.0)
            stderr = math.sqrt(var / shots)
        else:
            stderr = 0.0
        return mean, stderr, pulse_slots_total / max(shots, 1)


def simulate_random_cx_circuit(
    decoder, noise: NoiseParams, options: Mapping[str, bool],
    control_ids, target_ids, shots: int, seed: int,
    inverse_q: Optional[Mapping[str, float]] = None,
    inverse_pre_c_ids: Optional[Sequence[int]] = None,
    inverse_pre_t_ids: Optional[Sequence[int]] = None,
    inverse_num_cx: Optional[int] = None,
    inverse_final_c_id: Optional[int] = None,
    inverse_final_t_id: Optional[int] = None,
):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for the random logical-CX simulator.")

    control_ids = np.asarray(control_ids, dtype=np.int8)
    target_ids = np.asarray(target_ids, dtype=np.int8)
    if (
        inverse_pre_c_ids is None
        or inverse_pre_t_ids is None
        or inverse_num_cx is None
        or inverse_final_c_id is None
        or inverse_final_t_id is None
    ):
        compiled_inverse = compile_global_inverse_cx_sequence(control_ids, target_ids)
        inverse_pre_c_ids = compiled_inverse["inverse_pre_c_ids"]
        inverse_pre_t_ids = compiled_inverse["inverse_pre_t_ids"]
        inverse_num_cx = compiled_inverse["inverse_num_cx"]
        inverse_final_c_id = compiled_inverse["inverse_final_c_id"]
        inverse_final_t_id = compiled_inverse["inverse_final_t_id"]

    data = _cx_compiled(decoder)
    if inverse_q is None:
        cumulative = np.ones(16, dtype=np.float64)
        signs = np.ones(16, dtype=np.float64)
        gamma = 1.0
        use_pec = False
    else:
        cumulative, signs, gamma = _prepare_inverse_sampling_arrays(inverse_q)
        use_pec = True
    return _simulate_random_cx_circuit_kernel(
        control_ids, target_ids,
        np.asarray(inverse_pre_c_ids, dtype=np.int8),
        np.asarray(inverse_pre_t_ids, dtype=np.int8),
        int(inverse_num_cx), int(inverse_final_c_id), int(inverse_final_t_id),
        int(shots), int(seed), _cx_noise_array(noise),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        bool(use_pec), cumulative, signs, float(gamma),
        int(CX_CLIFFORD_IDENTITY), CX_CLIFFORD_MATS, CX_CLIFFORD_WORDS,
        CX_CLIFFORD_LENGTHS, CX_CLIFFORD_MUL, CX_CLIFFORD_INV,
        CX_TWIRL_CONTROL_IDS, CX_TWIRL_TARGET_IDS,
        data["op_type"], data["q1"], data["q2"], data["round_id"],
        data["basis_id"], data["check_id"], data["kind_id"],
        int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def warmup_random_cx_circuit(decoder, noise, options, seed):
    ids = _cx_local_gate_ids(["RX90"])
    return simulate_random_cx_circuit(
        decoder, noise, options, ids, ids, shots=2, seed=int(seed), inverse_q=None
    )


def _cx_g_vector_from_tau(tau_full):
    return np.array([
        float(tau_full["IX"]), float(tau_full["ZI"]), float(tau_full["ZX"]),
        float(tau_full["XY"]),
        0.5 * (float(tau_full["IY"]) + float(tau_full["ZY"])),
        0.5 * (float(tau_full["XI"]) + float(tau_full["XX"])),
    ])


def _cx_two_block_correlated_probability(prob):
    return float(sum(float(prob[p]) for p in PAULI2 if p != "II" and p[0] != "I" and p[1] != "I"))


def _cx_single_block_probability(prob):
    return float(sum(float(prob[p]) for p in PAULI2 if p != "II" and ((p[0] == "I") ^ (p[1] == "I"))))


def build_csb_equivalent_cx_pauli_models(learning, tolerance: float = 1e-10):
    """Construct Moore-Penrose and extremal correlated representatives of the CSB nullspace."""
    if linprog is None:
        raise RuntimeError("scipy.optimize.linprog is required for CX model comparison.")
    base_prob = {p: float(learning["tau_prob"][p]) for p in PAULI2}
    base_tau = two_qubit_fidelities_from_probs(base_prob)
    base_reduced = np.array([float(base_tau[p]) for p in CX_TAU_PARAMETER_ORDER])
    n1 = np.array([0.,0.,0.,0.,1.,0.,0.,-1.])
    n2 = np.array([0.,0.,0.,1.,0.,-1.,0.,0.])
    basis = np.column_stack([n1, n2])

    def pvec(offsets):
        vec = base_reduced + basis @ np.asarray(offsets, dtype=float)
        reduced = {p: float(vec[i]) for i, p in enumerate(CX_TAU_PARAMETER_ORDER)}
        full = expand_twirled_tau(reduced)
        _, raw = two_qubit_probs_from_fidelities(full, project_simplex=False)
        return np.array([float(raw[p]) for p in PAULI2])

    p0 = pvec([0.,0.])
    response = np.column_stack([pvec([1.,0.])-p0, pvec([0.,1.])-p0])
    correlated = np.array([p != "II" and p[0] != "I" and p[1] != "I" for p in PAULI2])
    objective = np.sum(response[correlated], axis=0)
    kwargs = dict(A_ub=-response, b_ub=p0 + float(tolerance), bounds=[(None,None),(None,None)], method="highs")
    minimum = linprog(objective, **kwargs)
    maximum = linprog(-objective, **kwargs)
    if not minimum.success or not maximum.success:
        raise RuntimeError("Could not find positivity-constrained CX nullspace representatives.")

    offsets = {
        "Moore-Penrose": np.zeros(2),
        "Minimum-correlated": np.asarray(minimum.x),
        "Maximum-correlated": np.asarray(maximum.x),
    }
    models = {}
    diagnostics = []
    probability_rows = []
    reference_g = _cx_g_vector_from_tau(base_tau)
    for name, off in offsets.items():
        vec = p0 + response @ off
        vec[np.abs(vec) < 5e-13] = 0.0
        if np.min(vec) < -5e-9:
            raise RuntimeError(f"{name} produced a negative probability.")
        vec = np.maximum(vec, 0.0); vec /= np.sum(vec)
        prob = {p: float(vec[i]) for i,p in enumerate(PAULI2)}
        tau = two_qubit_fidelities_from_probs(prob)
        inverse_q = inverse_two_qubit_pauli_quasiprobabilities_from_probs(prob)
        gamma = float(sum(abs(float(v)) for v in inverse_q.values()))
        mismatch = float(np.max(np.abs(_cx_g_vector_from_tau(tau)-reference_g)))
        if mismatch > 1e-8:
            raise RuntimeError(f"{name} changed measured CSB spectrum by {mismatch:.3e}.")
        models[name] = {"prob":prob, "tau":tau, "inverse_q":inverse_q, "offsets":off, "gamma":gamma}
        diagnostics.append({
            "model":name, "p_II":prob["II"], "total_nonidentity_probability":1-prob["II"],
            "single_block_probability":_cx_single_block_probability(prob),
            "two_block_correlated_probability":_cx_two_block_correlated_probability(prob),
            "p_XI":prob["XI"], "p_XX":prob["XX"], "p_IY":prob["IY"], "p_ZY":prob["ZY"],
            "gamma_per_composite_layer":gamma,
            "null_offset_IY_minus_ZY":float(off[0]), "null_offset_XI_minus_XX":float(off[1]),
            "max_abs_CSB_spectrum_difference":mismatch,
        })
        for p in PAULI2:
            probability_rows.append({"model":name, "pauli":p, "probability":prob[p]})
    return models, pd.DataFrame(diagnostics), pd.DataFrame(probability_rows)


def _combine_weighted_chunks(sub):
    shots = sub["shots"].to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    stderrs = sub["stderr"].to_numpy(dtype=float)
    n = int(np.sum(shots)); mean = float(np.sum(shots*means)/max(n,1))
    if n > 1:
        variances = (stderrs*np.sqrt(np.maximum(shots,1.0)))**2
        ss = float(np.sum(np.maximum(shots-1,0)*variances) + np.sum(shots*(means-mean)**2))
        stderr = math.sqrt(max(ss/(n-1),0.0)/n)
    else:
        stderr = 0.0
    return mean, stderr, n


def run_cx_multi_model_validation(
    decoder, inverse_models, noise, options, depths, sequences_per_depth,
    noisy_shots_per_sequence, pec_shots_per_sequence, local_gate_set,
    seed, n_jobs, chunk_shots, cache_file, min_rounds, max_rounds, verbose=False,
):
    sequences = generate_random_cx_sequences(depths, sequences_per_depth, seed, local_gate_set)
    rng = np.random.default_rng(int(seed)+17)
    tasks = []
    noisy_sizes = _chunk_sizes(noisy_shots_per_sequence, chunk_shots)
    pec_sizes = _chunk_sizes(pec_shots_per_sequence, chunk_shots)
    for seq in sequences:
        noisy_seeds = [int(rng.integers(2**32-1)) for _ in noisy_sizes]
        pec_seeds = [int(rng.integers(2**32-1)) for _ in pec_sizes]
        for cid,(n,sd) in enumerate(zip(noisy_sizes,noisy_seeds)):
            tasks.append({**seq,"model":"Noisy","shots":n,"chunk_id":cid,"seed":sd})
        for name in inverse_models:
            for cid,(n,sd) in enumerate(zip(pec_sizes,pec_seeds)):
                tasks.append({**seq,"model":name,"shots":n,"chunk_id":cid,"seed":sd})
    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder,"save"):
        decoder.save(cache_path)

    def one(task):
        dec = get_worker_decoder(cache_path,min_rounds,max_rounds)
        inv = None if task["model"]=="Noisy" else inverse_models[task["model"]]
        mean,se,slots = simulate_random_cx_circuit(
            dec, noise, options, task["control_ids"], task["target_ids"],
            task["shots"], task["seed"], inv,
            inverse_pre_c_ids=task["inverse_pre_c_ids"],
            inverse_pre_t_ids=task["inverse_pre_t_ids"],
            inverse_num_cx=task["inverse_num_cx"],
            inverse_final_c_id=task["inverse_final_c_id"],
            inverse_final_t_id=task["inverse_final_t_id"],
        )
        return {
            "depth": task["depth"],
            "sequence_id": task["sequence_id"],
            "model": task["model"],
            "shots": task["shots"],
            "chunk_id": task["chunk_id"],
            "mean": mean,
            "stderr": se,
            "average_compiled_pulse_slots": slots,
            "inverse_cx_count": int(task["inverse_num_cx"]),
            "total_composite_cx_layers": int(task["total_composite_cx_layers"]),
        }
    if verbose:
        print("CX Pauli-model PEC comparison", list(map(int,depths)), list(inverse_models))
    if JOBLIB_AVAILABLE and int(n_jobs)!=1 and len(tasks)>1:
        rows = Parallel(n_jobs=int(n_jobs),backend="loky",batch_size=1)(delayed(one)(t) for t in tasks)
    else:
        rows = [one(t) for t in tasks]
    chunk_df = pd.DataFrame(rows)
    seq_rows=[]
    for (depth,sid,model),sub in chunk_df.groupby(["depth","sequence_id","model"],sort=True):
        mean,se,n = _combine_weighted_chunks(sub)
        first = sub.iloc[0]
        seq_rows.append({
            "depth": int(depth),
            "sequence_id": int(sid),
            "model": model,
            "survival_probability": mean,
            "sampling_stderr": se,
            "shots": n,
            "inverse_cx_count": int(first["inverse_cx_count"]),
            "total_composite_cx_layers": int(first["total_composite_cx_layers"]),
        })
    sequence_df=pd.DataFrame(seq_rows)
    summary=[]
    for depth in map(int,depths):
        for name in ["Noisy"]+list(inverse_models):
            sub=sequence_df[(sequence_df.depth==depth)&(sequence_df.model==name)]
            vals=sub.survival_probability.to_numpy(float); within=sub.sampling_stderr.to_numpy(float)
            seq_se=float(np.std(vals,ddof=1)/math.sqrt(len(vals))) if len(vals)>1 else 0.0
            sampling_se=float(np.sqrt(np.sum(within**2))/max(len(vals),1))
            gamma=1.0 if name=="Noisy" else float(sum(abs(float(v)) for v in inverse_models[name].values()))
            inverse_counts = sub.inverse_cx_count.to_numpy(dtype=float)
            total_layers = sub.total_composite_cx_layers.to_numpy(dtype=float)
            overhead = np.power(float(gamma), 2.0 * total_layers)
            summary.append({
                "depth": depth,
                "model": name,
                "ideal_survival_probability": 1.0,
                "num_sequences": len(vals),
                "mean_inverse_cx_count": float(np.mean(inverse_counts)),
                "min_inverse_cx_count": int(np.min(inverse_counts)) if len(inverse_counts) else 0,
                "max_inverse_cx_count": int(np.max(inverse_counts)) if len(inverse_counts) else 0,
                "mean_total_composite_cx_layers": float(np.mean(total_layers)),
                # Backward-compatible alias.  Because the globally compiled
                # inverse can use 0--3 CX gates depending on the sequence,
                # this is the mean total layer count at a fixed forward depth.
                "total_composite_cx_layers": float(np.mean(total_layers)),
                "survival_probability": float(np.mean(vals)),
                "sequence_stderr": seq_se,
                "sampling_stderr": sampling_se,
                "gamma_per_composite_layer": gamma,
                "estimated_sampling_overhead": float(np.mean(overhead)),
            })
    return pd.DataFrame(summary), sequence_df


def cx_standard_pec_summary(comparison_summary, sequences_per_depth, pec_model="Moore-Penrose"):
    noisy = comparison_summary[comparison_summary.model=="Noisy"][
        ["depth", "survival_probability"]
    ].rename(columns={"survival_probability":"noisy_survival_probability"})
    pec_columns = [
        "depth", "survival_probability", "gamma_per_composite_layer",
        "estimated_sampling_overhead", "mean_inverse_cx_count",
        "min_inverse_cx_count", "max_inverse_cx_count",
        "mean_total_composite_cx_layers", "total_composite_cx_layers",
    ]
    pec = comparison_summary[comparison_summary.model==pec_model][pec_columns].rename(
        columns={"survival_probability":"pec_survival_probability"}
    )
    out = noisy.merge(pec, on="depth")
    out["ideal_survival_probability"] = 1.0
    out["num_sequences"] = int(sequences_per_depth)
    return out.sort_values("depth").reset_index(drop=True)


def _summary_mean_std(df, group_cols, value_cols):
    agg={}
    for col in value_cols:
        agg[f"{col}_mean"]=(col,"mean"); agg[f"{col}_std"]=(col,"std")
    out=df.groupby(group_cols,as_index=False).agg(**agg)
    for col in out.filter(regex="_std$"):
        out[col]=out[col].fillna(0.0)
    return out


def run_cx_ft_scaling(
    decoder, options, noise_rep_schedule, steps, shots, chunk_shots,
    n_repeats, fit_max_p, seed, n_jobs, cache_file, min_rounds, max_rounds,
    two_qubit_factor=2.0, final_meas_factor=1.0, verbose=False,
):
    repeat_rows=[]
    csb_rows=[]
    modes=["IX","ZI","ZX","XY","A","C"]
    for p_index,config in enumerate(noise_rep_schedule):
        p=float(config["p"]); rep=int(config["rep"]); local_steps=config.get("steps",steps)
        noise=make_homogeneous_noise(p,two_qubit_factor,final_meas_factor)
        for repeat_id in range(int(n_repeats)):
            sd=int(seed+p_index*100_000_000+repeat_id*1_000_003)
            if verbose: print(f"FT p={p:.3g} repeat {repeat_id+1}/{n_repeats}")
            csb=run_cx_csb_spectrum(decoder,noise,options,local_steps,rep,shots,sd,n_jobs,chunk_shots,
                                    cache_file,min_rounds,max_rounds,False)
            csb=csb.copy(); csb["physical_noise_strength"]=p; csb["repeat_id"]=repeat_id
            csb_rows.append(csb)
            spectra,g=fit_cx_csb_spectra(csb); by=spectra.set_index("mode")
            F=(1+g["g_IX"]+g["g_ZI"]+g["g_ZX"]+4*g["g_XY"]+4*g["g_A"]+4*g["g_C"])/16
            row={"physical_noise_strength":p,"repeat_id":repeat_id,"seed":sd,"rep":rep,
                 "max_depth":rep*max(map(int,local_steps)),"process_fidelity":F,"process_infidelity":1-F,
                 "abs_delta_A":abs(float(by.loc["A","delta_theta"])),"abs_delta_C":abs(float(by.loc["C","delta_theta"])),
                 "mean_unverified_rate":float(csb.unverified_rate.mean()),"mean_ec_rounds":float(csb.average_ec_rounds.mean())}
            for mode in modes:
                row[f"g_{mode}"]=float(g[f"g_{mode}"]); row[f"loss_{mode}"]=1-float(g[f"g_{mode}"])
            repeat_rows.append(row)
    raw=pd.DataFrame(repeat_rows).sort_values(["physical_noise_strength","repeat_id"]).reset_index(drop=True)
    values=[f"loss_{m}" for m in modes]+["abs_delta_A","abs_delta_C","process_fidelity","process_infidelity","mean_unverified_rate","mean_ec_rounds"]
    summary=_summary_mean_std(raw,["physical_noise_strength"],values)
    meta=raw.groupby("physical_noise_strength",as_index=False).agg(n_repeats=("repeat_id","nunique"),rep=("rep","first"),max_depth=("max_depth","first"))
    summary=meta.merge(summary,on="physical_noise_strength").sort_values("physical_noise_strength").reset_index(drop=True)
    p=summary.physical_noise_strength.to_numpy(float); y=summary.process_infidelity_mean.to_numpy(float); ys=summary.process_infidelity_std.to_numpy(float)
    mask=np.isfinite(p)&np.isfinite(y)&(p>0)&(y>0)&(p<=float(fit_max_p))
    if np.count_nonzero(mask)<2: raise RuntimeError("At least two positive FT-scaling points are required.")
    fp=p[mask]; fy=y[mask]; fs=ys[mask]
    w=np.ones_like(fs); valid_w=np.isfinite(fs)&(fs>0); w[valid_w]=1.0/fs[valid_w]**2
    x=fp**2
    A=float(np.sum(w*x*fy)/np.sum(w*x*x)); slope,intercept=np.polyfit(np.log(fp),np.log(fy),1)
    fit=pd.DataFrame([{"fit_max_noise_strength":float(fit_max_p),"num_fit_points":len(fp),"quadratic_coefficient_A":A,
                       "free_power_slope":float(slope),"free_power_prefactor":float(np.exp(intercept))}])
    return {"csb_df":pd.concat(csb_rows,ignore_index=True),"raw_df":raw,"summary_df":summary,"fit_df":fit}


def run_cx_pauli_model_comparison_workflow(
    decoder, noise, options, csb_steps, csb_rep, csb_shots, csb_chunk_shots,
    n_repeats, depths, sequences_per_depth, noisy_shots_per_sequence,
    pec_shots_per_sequence, local_gate_set, seed, n_jobs, pec_chunk_shots,
    cache_file, min_rounds, max_rounds, verbose=False,
):
    spectra=[]; tau=[]; pec=[]; comparison=[]; probs=[]; diagnostics=[]
    for repeat_id in range(int(n_repeats)):
        sd=int(seed+repeat_id*1_000_003)
        if verbose: print(f"PEC repeat {repeat_id+1}/{n_repeats}")
        csb=run_cx_csb_spectrum(decoder,noise,options,csb_steps,csb_rep,csb_shots,sd,n_jobs,csb_chunk_shots,
                                cache_file,min_rounds,max_rounds,False)
        learning=learning_tables_from_cx_csb(csb,beta=None)
        models,diag_df,prob_df=build_csb_equivalent_cx_pauli_models(learning)
        inverse_models={name:model["inverse_q"] for name,model in models.items()}
        comp_df,_seq=run_cx_multi_model_validation(
            decoder,inverse_models,noise,options,depths,sequences_per_depth,
            noisy_shots_per_sequence,pec_shots_per_sequence,local_gate_set,sd+999,n_jobs,
            pec_chunk_shots,cache_file,min_rounds,max_rounds,verbose,
        )
        pec_df=cx_standard_pec_summary(comp_df,sequences_per_depth)
        for df,target in [(learning["spectra_df"],spectra),(learning["tau_df"],tau),(pec_df,pec),
                          (comp_df,comparison),(prob_df,probs),(diag_df,diagnostics)]:
            d=df.copy(); d["repeat_id"]=repeat_id; target.append(d)
    spectra_df=pd.concat(spectra,ignore_index=True); tau_df=pd.concat(tau,ignore_index=True); pec_df=pd.concat(pec,ignore_index=True)
    comparison_df=pd.concat(comparison,ignore_index=True); prob_df=pd.concat(probs,ignore_index=True); diag_df=pd.concat(diagnostics,ignore_index=True)
    return {
        "spectra_repeats_df":spectra_df,"tau_repeats_df":tau_df,"pec_repeats_df":pec_df,
        "model_comparison_repeats_df":comparison_df,"model_probability_repeats_df":prob_df,"model_diagnostics_repeats_df":diag_df,
        "spectra_summary_df":_summary_mean_std(spectra_df,["mode"],["g","delta_theta"]).sort_values("mode").reset_index(drop=True),
        "tau_summary_df":_summary_mean_std(tau_df,["pauli"],["tau_projected","probability_projected","inverse_q"]).sort_values("pauli").reset_index(drop=True),
        "pec_summary_df":_summary_mean_std(pec_df,["depth"],["noisy_survival_probability","pec_survival_probability"]).sort_values("depth").reset_index(drop=True),
        "model_comparison_summary_df":_summary_mean_std(comparison_df,["depth","model"],["survival_probability","gamma_per_composite_layer"]).sort_values(["depth","model"]).reset_index(drop=True),
        "model_probability_summary_df":_summary_mean_std(prob_df,["model","pauli"],["probability"]).sort_values(["model","pauli"]).reset_index(drop=True),
        "model_diagnostics_summary_df":_summary_mean_std(diag_df,["model"],["p_II","total_nonidentity_probability","single_block_probability","two_block_correlated_probability","p_XI","p_XX","p_IY","p_ZY","gamma_per_composite_layer","max_abs_CSB_spectrum_difference"]).sort_values("model").reset_index(drop=True),
    }


# =============================================================================
# Logical T-gate CSB and PEC
# =============================================================================


# =============================================================================
# Source-separated noise model
# =============================================================================

@dataclass(frozen=True)
class TGateNoiseModel:
    """Noise parameters for one complete logical T-gate cycle."""

    gadget_noise: NoiseParams
    magic_angle_error: float
    magic_error_rate: float
    magic_nu_x: float
    magic_nu_y: float
    magic_nu_z: float
    phase_twirl: bool

    def validated(self) -> "TGateNoiseModel":
        r = float(self.magic_error_rate)
        if not (0.0 <= r <= 1.0):
            raise ValueError("magic_error_rate must be in [0,1].")
        nu = np.array([self.magic_nu_x, self.magic_nu_y, self.magic_nu_z], dtype=float)
        if np.any(nu < -1e-15):
            raise ValueError("magic-state Pauli composition must be nonnegative.")
        if r > 0 and not np.isclose(float(np.sum(nu)), 1.0, atol=1e-10):
            raise ValueError("magic_nu_x + magic_nu_y + magic_nu_z must equal 1.")
        return self


def make_t_noise_model(
    gadget_strength: float,
    two_qubit_factor: float,
    final_meas_factor: float,
    magic_angle_error: float,
    magic_error_rate: float,
    magic_nu: Mapping[str, float],
    phase_twirl: bool,
) -> TGateNoiseModel:
    return TGateNoiseModel(
        gadget_noise=make_homogeneous_noise(
            gadget_strength, two_qubit_factor, final_meas_factor
        ),
        magic_angle_error=float(magic_angle_error),
        magic_error_rate=float(magic_error_rate),
        magic_nu_x=float(magic_nu.get("X", 0.0)),
        magic_nu_y=float(magic_nu.get("Y", 0.0)),
        magic_nu_z=float(magic_nu.get("Z", 0.0)),
        phase_twirl=bool(phase_twirl),
    ).validated()


def ideal_t_noise_model(
    magic_angle_error: float,
    magic_error_rate: float,
    magic_nu: Mapping[str, float],
    phase_twirl: bool,
) -> TGateNoiseModel:
    return TGateNoiseModel(
        gadget_noise=NoiseParams(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        magic_angle_error=float(magic_angle_error),
        magic_error_rate=float(magic_error_rate),
        magic_nu_x=float(magic_nu.get("X", 0.0)),
        magic_nu_y=float(magic_nu.get("Y", 0.0)),
        magic_nu_z=float(magic_nu.get("Z", 0.0)),
        phase_twirl=bool(phase_twirl),
    ).validated()


# =============================================================================
# Branch definitions and small utilities
# =============================================================================

T_CSB_BRANCHES: Dict[str, Dict[str, Any]] = {
    "transverse_real": {
        "code": 0,
        "initial": "+",
        "measure": "X",
        "kind": "complex_real",
    },
    "transverse_imag": {
        "code": 1,
        "initial": "+",
        "measure": "Y",
        "kind": "complex_imag",
    },
    "axis_0": {
        "code": 2,
        "initial": "0",
        "measure": "Z",
        "kind": "axis_0",
    },
    "axis_1": {
        "code": 3,
        "initial": "1",
        "measure": "Z",
        "kind": "axis_1",
    },
}


def _t_noise_array(noise: NoiseParams) -> np.ndarray:
    return np.array(
        [
            noise.p_1q,
            noise.p_2q,
            noise.p_meas,
            noise.p_reset,
            noise.p_idle,
            noise.p_final_meas,
        ],
        dtype=np.float64,
    )


def _magic_array(model: TGateNoiseModel) -> np.ndarray:
    return np.array(
        [
            model.magic_angle_error,
            model.magic_error_rate,
            model.magic_nu_x,
            model.magic_nu_y,
            model.magic_nu_z,
        ],
        dtype=np.float64,
    )


def _t_compiled(decoder):
    return get_compiled_numba_data(decoder)


def _t_chunk_sizes(total: int, chunk: int) -> List[int]:
    total = int(total)
    chunk = max(1, int(chunk))
    out: List[int] = []
    while total > 0:
        n = min(total, chunk)
        out.append(n)
        total -= n
    return out


def _aggregate_sample_chunks(sub: pd.DataFrame) -> Dict[str, float]:
    n_arr = sub["shots"].to_numpy(dtype=float)
    means = sub["mean"].to_numpy(dtype=float)
    stderrs = sub["stderr"].to_numpy(dtype=float)
    n = int(np.sum(n_arr))
    total = float(np.sum(n_arr * means))
    mean = total / max(n, 1)
    if n > 1:
        sample_vars = (stderrs * np.sqrt(np.maximum(n_arr, 1.0))) ** 2
        ss_within = float(np.sum(np.maximum(n_arr - 1.0, 0.0) * sample_vars))
        ss_between = float(np.sum(n_arr * (means - mean) ** 2))
        var = max((ss_within + ss_between) / (n - 1), 0.0)
        stderr = math.sqrt(var / n)
    else:
        stderr = 0.0
    return {"shots": n, "mean": mean, "stderr": float(stderr)}


def _phase_wrap(x: float) -> float:
    return float((x + np.pi) % (2 * np.pi) - np.pi)


# =============================================================================
# Numba hybrid state / Pauli-frame simulator
# =============================================================================

if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _apply_pauli_1q_state(a0, a1, pcode):
        p = int(pcode)
        if p == 0:
            return a0, a1
        if p == 1:  # X
            return a1, a0
        if p == 2:  # Y, global phase convention irrelevant
            return -1j * a1, 1j * a0
        # Z
        return a0, -a1


    @nb.njit(cache=True)
    def _apply_s_power_state(a0, a1, exponent):
        e = int(exponent) & 3
        if e == 0:
            return a0, a1
        if e == 1:
            return a0, 1j * a1
        if e == 2:
            return a0, -a1
        return a0, -1j * a1


    @nb.njit(cache=True)
    def _apply_rx90_state(a0, a1):
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return (
            inv_sqrt2 * (a0 - 1j * a1),
            inv_sqrt2 * (-1j * a0 + a1),
        )


    @nb.njit(cache=True)
    def _apply_ry90_state(a0, a1):
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        return (
            inv_sqrt2 * (a0 - a1),
            inv_sqrt2 * (a0 + a1),
        )


    @nb.njit(cache=True)
    def _apply_base_clifford_state_and_frame(
        a0, a1, x_mask, z_mask, gate_code
    ):
        """Apply RX90, RY90, or RZ90 without adding a noise location."""
        g = int(gate_code)
        if g == 0:  # RX90
            a0, a1 = _apply_rx90_state(a0, a1)
            x_mask = np.uint16(x_mask ^ z_mask)
        elif g == 1:  # RY90
            a0, a1 = _apply_ry90_state(a0, a1)
            tmp = x_mask
            x_mask = z_mask
            z_mask = tmp
        else:  # RZ90 = S up to global phase
            a0, a1 = _apply_s_power_state(a0, a1, 1)
            z_mask = np.uint16(z_mask ^ x_mask)
        return a0, a1, x_mask, z_mask


    @nb.njit(cache=True)
    def _apply_pauli_to_state_only(a0, a1, pcode):
        # Conjugating a binary physical Pauli frame by a logical Pauli changes
        # only signs, so the frame masks do not change.
        return _apply_pauli_1q_state(a0, a1, pcode)


    @nb.njit(cache=True)
    def _apply_pauli_data_2q(a00, a01, a10, a11, pcode):
        p = int(pcode)
        if p == 0:
            return a00, a01, a10, a11
        if p == 1:  # X on data
            return a10, a11, a00, a01
        if p == 2:  # Y on data
            return -1j * a10, -1j * a11, 1j * a00, 1j * a01
        return a00, a01, -a10, -a11


    @nb.njit(cache=True)
    def _apply_pauli_ancilla_2q(a00, a01, a10, a11, pcode):
        p = int(pcode)
        if p == 0:
            return a00, a01, a10, a11
        if p == 1:  # X on ancilla
            return a01, a00, a11, a10
        if p == 2:  # Y on ancilla
            return -1j * a01, 1j * a00, -1j * a11, 1j * a10
        return a00, -a01, a10, -a11


    @nb.njit(cache=True)
    def _canonical_logical_code(x_mask, z_mask):
        """
        Decompose a seven-qubit Pauli frame into

            canonical syndrome representative * stabilizer * logical Pauli.

        For the Steane CSS code, row-space stabilizers have even parity and the
        all-ones logical representatives have odd parity after the syndrome is
        removed.
        """
        sx = _syndrome_from_mask(x_mask)
        sz = _syndrome_from_mask(z_mask)
        x_can = _ordinary_decode_mask(sx)
        z_can = _ordinary_decode_mask(sz)
        x_zero = np.uint16(x_mask ^ x_can)
        z_zero = np.uint16(z_mask ^ z_can)
        lx = _parity_u16(x_zero)
        lz = _parity_u16(z_zero)
        pcode = 0
        if lx and lz:
            pcode = 2
        elif lx:
            pcode = 1
        elif lz:
            pcode = 3
        return np.uint16(x_can), np.uint16(z_can), pcode


    @nb.njit(cache=True)
    def _canonicalize_1q_frame(x_mask, z_mask, a0, a1):
        x_can, z_can, pcode = _canonical_logical_code(x_mask, z_mask)
        a0, a1 = _apply_pauli_1q_state(a0, a1, pcode)
        return x_can, z_can, a0, a1, pcode


    @nb.njit(cache=True)
    def _canonicalize_2q_frames(
        xd, zd, xa, za, a00, a01, a10, a11
    ):
        xd, zd, pd = _canonical_logical_code(xd, zd)
        a00, a01, a10, a11 = _apply_pauli_data_2q(
            a00, a01, a10, a11, pd
        )
        xa, za, pa = _canonical_logical_code(xa, za)
        a00, a01, a10, a11 = _apply_pauli_ancilla_2q(
            a00, a01, a10, a11, pa
        )
        return xd, zd, xa, za, a00, a01, a10, a11, pd, pa


    @nb.njit(cache=True)
    def _run_one_ftec(
        x_mask,
        z_mask,
        state,
        noise_arr,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        p1, p2, pm, pr, pi, _pf = noise_arr
        return _run_ft_ec_numba(
            x_mask,
            z_mask,
            state,
            p1,
            p2,
            pm,
            pr,
            pi,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
            fallback_on_unverified,
        )


    @nb.njit(cache=True)
    def _apply_idle_faults(x_mask, z_mask, state, p_idle):
        for q in range(7):
            state, pc = _sample_oneq_pauli(state, p_idle)
            if pc != 0:
                x_mask, z_mask = _apply_pauli_code(x_mask, z_mask, q, pc)
        return x_mask, z_mask, state


    @nb.njit(cache=True)
    def _apply_transversal_cnot_faults(xd, zd, xa, za, state, p_2q):
        for q in range(7):
            bit = np.uint16(1 << q)
            if xd & bit:
                xa = np.uint16(xa ^ bit)
            if za & bit:
                zd = np.uint16(zd ^ bit)
            state, pd, pa = _sample_twoq_pauli(state, p_2q)
            if pd != 0:
                xd, zd = _apply_pauli_code(xd, zd, q, pd)
            if pa != 0:
                xa, za = _apply_pauli_code(xa, za, q, pa)
        return xd, zd, xa, za, state


    @nb.njit(cache=True)
    def _sample_magic_pauli(state, magic_arr):
        _eps, r, nx, ny, nz = magic_arr
        state, u = _rng_uniform01(state)
        if u >= r:
            return state, 0
        state, v = _rng_uniform01(state)
        if v < nx:
            return state, 1
        if v < nx + ny:
            return state, 2
        return state, 3


    @nb.njit(cache=True)
    def _prepare_input_state(branch_code):
        b = int(branch_code)
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        if b == 0 or b == 1:  # |+>
            return complex(inv_sqrt2, 0.0), complex(inv_sqrt2, 0.0)
        if b == 2:  # |0>
            return complex(1.0, 0.0), complex(0.0, 0.0)
        return complex(0.0, 0.0), complex(1.0, 0.0)


    @nb.njit(cache=True)
    def _prepare_noisy_input(
        branch_code,
        xd,
        zd,
        d0,
        d1,
        state,
        noise_arr,
        prep_gates_noisy,
        prep_qec_after_gate,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        p1 = noise_arr[0]
        b = int(branch_code)
        has_gate = False
        if b == 0 or b == 1:  # |+> = H|0>
            tmp = xd
            xd = zd
            zd = tmp
            has_gate = True
        elif b == 3:  # |1> = X|0>; binary Pauli-frame action is sign only
            has_gate = True

        if has_gate and prep_gates_noisy:
            xd, zd, state = _apply_1q_gate_noise_all_data_numba(
                xd, zd, state, p1
            )

        unv = 0
        rounds = 0
        if has_gate and prep_qec_after_gate:
            xd, zd, state, unv, rounds = _run_one_ftec(
                xd, zd, state, noise_arr, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            xd, zd, d0, d1, _p = _canonicalize_1q_frame(xd, zd, d0, d1)
        return xd, zd, d0, d1, state, unv, rounds


    @nb.njit(cache=True)
    def _one_t_cycle(
        xd,
        zd,
        d0,
        d1,
        state,
        noise_arr,
        magic_arr,
        correction_offset,
        measurement_basis_noisy,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """
        Apply one complete noisy logical T-injection cycle.

        ``correction_offset`` is an S-power in Z4 that has been compiled into
        the existing feed-forward correction slot.  The actual correction is

            S^(observed_m + correction_offset).

        No ideal S gate is inserted before or after the noisy gadget.  For a
        T-only sequence the caller uses k_current-k_previous.  For a C+T random
        circuit the previous post-twirl is compiled into the current Clifford,
        so the caller uses k_current.
        """
        p1, p2, _pm, _pr, pi, pf = noise_arr
        eps = magic_arr[0]
        unverified = 0
        ec_cycles = 0
        ec_rounds = 0

        # Fresh accepted magic state |A(eps)> and optional accepted-state Pauli.
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        r0 = complex(inv_sqrt2, 0.0)
        phase = np.pi / 4.0 + eps
        r1 = inv_sqrt2 * complex(math.cos(phase), math.sin(phase))
        state, magic_p = _sample_magic_pauli(state, magic_arr)
        r0, r1 = _apply_pauli_1q_state(r0, r1, magic_p)

        # Product state in |data,ancilla> order.
        a00 = d0 * r0
        a01 = d0 * r1
        a10 = d1 * r0
        a11 = d1 * r1
        xa = np.uint16(0)
        za = np.uint16(0)

        # Ideal logical CNOT D -> A and its transversal physical implementation.
        tmp = a10
        a10 = a11
        a11 = tmp
        xd, zd, xa, za, state = _apply_transversal_cnot_faults(
            xd, zd, xa, za, state, p2
        )

        # FTEC on both blocks.
        xd, zd, state, u, r = _run_one_ftec(
            xd, zd, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xa, za, state, u, r = _run_one_ftec(
            xa, za, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r

        (
            xd, zd, xa, za,
            a00, a01, a10, a11,
            _pd, _pa,
        ) = _canonicalize_2q_frames(
            xd, zd, xa, za, a00, a01, a10, a11
        )

        # Ideal ancilla-Z branch followed by decoded physical readout noise.
        p_m0 = (
            a00.real * a00.real + a00.imag * a00.imag
            + a10.real * a10.real + a10.imag * a10.imag
        )
        p_m0 = min(max(p_m0, 0.0), 1.0)
        state, u01 = _rng_uniform01(state)
        ideal_m = 0 if u01 < p_m0 else 1
        if ideal_m == 0:
            norm = math.sqrt(max(p_m0, 1e-300))
            d0 = a00 / norm
            d1 = a10 / norm
        else:
            norm = math.sqrt(max(1.0 - p_m0, 1e-300))
            d0 = a01 / norm
            d1 = a11 / norm

        state, meas_flip = _logical_measurement_flip_numba(
            xa, za, state, MEAS_Z, p1, pf,
            measurement_basis_noisy,
        )
        observed_m = ideal_m ^ int(meas_flip)

        # Data idle during destructive resource measurement, then FTEC.
        xd, zd, state = _apply_idle_faults(xd, zd, state, pi)
        xd, zd, state, u, r = _run_one_ftec(
            xd, zd, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xd, zd, d0, d1, _p = _canonicalize_1q_frame(xd, zd, d0, d1)

        # The pre-T C4 frame is compiled into the existing injection correction.
        # All four exponents occupy exactly one synchronized correction slot.
        correction_exponent = (int(observed_m) + int(correction_offset)) & 3
        if correction_exponent == 0:
            xd, zd, state = _apply_idle_faults(xd, zd, state, pi)
        else:
            d0, d1 = _apply_s_power_state(d0, d1, correction_exponent)
            if correction_exponent & 1:
                zd = np.uint16(zd ^ xd)
            xd, zd, state = _apply_1q_gate_noise_all_data_numba(
                xd, zd, state, p1
            )

        xd, zd, state, u, r = _run_one_ftec(
            xd, zd, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xd, zd, d0, d1, _p = _canonicalize_1q_frame(xd, zd, d0, d1)

        return (
            xd,
            zd,
            d0,
            d1,
            state,
            unverified,
            ec_cycles,
            ec_rounds,
            observed_m,
            correction_exponent,
        )


    @nb.njit(cache=True)
    def _apply_compiled_clifford_and_ec(
        xd,
        zd,
        d0,
        d1,
        state,
        base_gate_code,
        incoming_t_pauli,
        incoming_post_twirl_k,
        clifford_pec_pauli,
        noise_arr,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """
        Compile one random Clifford layer as

            P_C^PEC * C * P_T(previous) * S^(-k_previous)

        and then apply the physical logical-Clifford noise once, followed by one
        FTEC.  The factors are applied sequentially only as ideal algebra; there
        is no noise or QEC between them.  Even if the product is identity, this
        remains one fixed Clifford slot with the same p_1q noise, which is needed
        for the gate-independent PEC channel model E_C o E_C^{-1}.
        """
        p1 = noise_arr[0]

        # Previous T post-twirl, then previous T PEC Pauli.
        e = (-int(incoming_post_twirl_k)) & 3
        if e != 0:
            d0, d1 = _apply_s_power_state(d0, d1, e)
            if e & 1:
                zd = np.uint16(zd ^ xd)
        d0, d1 = _apply_pauli_to_state_only(
            d0, d1, int(incoming_t_pauli)
        )

        # Base random Clifford, then its sampled inverse-noise Pauli.
        d0, d1, xd, zd = _apply_base_clifford_state_and_frame(
            d0, d1, xd, zd, int(base_gate_code)
        )
        d0, d1 = _apply_pauli_to_state_only(
            d0, d1, int(clifford_pec_pauli)
        )

        # One physical noise location for the compiled logical Clifford.
        xd, zd, state = _apply_1q_gate_noise_all_data_numba(
            xd, zd, state, p1
        )
        xd, zd, state, unv, rounds = _run_one_ftec(
            xd, zd, state, noise_arr, fallback_on_unverified,
            op_type, q1, q2, round_id, basis_id, check_id, kind_id,
            min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
        )
        xd, zd, d0, d1, _p = _canonicalize_1q_frame(
            xd, zd, d0, d1
        )
        return xd, zd, d0, d1, state, unv, rounds


    @nb.njit(cache=True)
    def _absorb_final_t_frame(
        xd, zd, d0, d1, post_twirl_k, t_pec_pauli
    ):
        """Absorb P_T S^(-k) into the final measurement frame."""
        e = (-int(post_twirl_k)) & 3
        if e != 0:
            d0, d1 = _apply_s_power_state(d0, d1, e)
            if e & 1:
                zd = np.uint16(zd ^ xd)
        d0, d1 = _apply_pauli_to_state_only(
            d0, d1, int(t_pec_pauli)
        )
        return xd, zd, d0, d1


    @nb.njit(cache=True)
    def _bloch_mean(d0, d1, meas_code):
        cross = np.conjugate(d0) * d1
        if meas_code == MEAS_X:
            return 2.0 * cross.real
        if meas_code == MEAS_Y:
            return 2.0 * cross.imag
        return (
            d0.real * d0.real + d0.imag * d0.imag
            - d1.real * d1.real - d1.imag * d1.imag
        )


    @nb.njit(cache=True)
    def _measure_output(
        xd, zd, d0, d1, state, meas_code, noise_arr, measurement_basis_noisy
    ):
        p1, _p2, _pm, _pr, _pi, pf = noise_arr
        mean = min(max(_bloch_mean(d0, d1, meas_code), -1.0), 1.0)
        state, u = _rng_uniform01(state)
        out = 1 if u < 0.5 * (1.0 + mean) else -1

        # Use literal measurement codes at the call site.  Some Numba
        # versions/cache states otherwise specialize
        # _logical_measurement_flip_numba to a Literal[int] signature and
        # reject a dynamically typed int64 measurement code.
        if meas_code == MEAS_X:
            state, flip = _logical_measurement_flip_numba(
                xd, zd, state, MEAS_X, p1, pf, measurement_basis_noisy
            )
        elif meas_code == MEAS_Y:
            state, flip = _logical_measurement_flip_numba(
                xd, zd, state, MEAS_Y, p1, pf, measurement_basis_noisy
            )
        else:
            state, flip = _logical_measurement_flip_numba(
                xd, zd, state, MEAS_Z, p1, pf, measurement_basis_noisy
            )

        if flip & 1:
            out = -out
        return state, out


    @nb.njit(cache=True)
    def _branch_to_meas_code(branch_code):
        b = int(branch_code)
        if b == 0:
            return MEAS_X
        if b == 1:
            return MEAS_Y
        return MEAS_Z


    @nb.njit(cache=True)
    def _simulate_t_csb_point_kernel(
        branch_code,
        depth,
        shots,
        seed,
        noise_arr,
        magic_arr,
        use_phase_twirl,
        prep_gates_noisy,
        prep_qec_after_gate,
        measurement_basis_noisy,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        total = 0.0
        total2 = 0.0
        unverified_total = 0
        ec_cycles_total = 0
        ec_rounds_total = 0
        m1_total = 0
        base_state = np.uint64(seed) + np.uint64(0xD6E8FEB86659FD93)

        for shot in range(int(shots)):
            state = base_state + np.uint64(shot) * np.uint64(0x9E3779B97F4A7C15)
            state = _rng_next_u64(_rng_next_u64(state))
            xd = np.uint16(0)
            zd = np.uint16(0)
            d0, d1 = _prepare_input_state(branch_code)

            xd, zd, d0, d1, state, u, r = _prepare_noisy_input(
                branch_code, xd, zd, d0, d1, state, noise_arr,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            unverified_total += u
            if r > 0:
                ec_cycles_total += 1
                ec_rounds_total += r

            previous_twirl_k = 0
            for _cycle in range(int(depth)):
                current_twirl_k = 0
                correction_offset = 0
                if use_phase_twirl:
                    state, current_twirl_k = _rng_int(state, 4)
                    correction_offset = (
                        int(current_twirl_k) - int(previous_twirl_k)
                    ) & 3

                (
                    xd, zd, d0, d1, state,
                    u, e, r, observed_m, _corr,
                ) = _one_t_cycle(
                    xd, zd, d0, d1, state, noise_arr, magic_arr,
                    correction_offset, measurement_basis_noisy, fallback_on_unverified,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )
                unverified_total += u
                ec_cycles_total += e
                ec_rounds_total += r
                m1_total += observed_m
                previous_twirl_k = current_twirl_k

            # The final S^(-k_L) is absorbed into the measurement frame.
            if use_phase_twirl and int(depth) > 0:
                xd, zd, d0, d1 = _absorb_final_t_frame(
                    xd, zd, d0, d1, previous_twirl_k, 0
                )

            meas_code = _branch_to_meas_code(branch_code)
            state, out = _measure_output(
                xd, zd, d0, d1, state, meas_code, noise_arr, measurement_basis_noisy
            )
            total += out
            total2 += out * out

        n = int(shots)
        if n <= 0:
            return np.nan, np.nan, 0.0, 0.0, 0.0
        mean = total / n
        if n > 1:
            var = max((total2 - n * mean * mean) / (n - 1), 0.0)
            stderr = math.sqrt(var / n)
        else:
            stderr = 0.0
        unv_rate = float(unverified_total) / max(ec_cycles_total, 1)
        avg_rounds = float(ec_rounds_total) / max(ec_cycles_total, 1)
        m1_rate = float(m1_total) / max(n * int(depth), 1)
        return mean, stderr, unv_rate, avg_rounds, m1_rate


    @nb.njit(cache=True)
    def _sample_inverse_pauli_numba(state, cumulative, signs, gamma):
        state, u = _rng_uniform01(state)
        idx = cumulative.shape[0] - 1
        for j in range(cumulative.shape[0]):
            if u <= cumulative[j]:
                idx = j
                break
        return state, idx, gamma * signs[idx]


    @nb.njit(cache=True)
    def _simulate_t_pec_point_kernel(
        depth,
        shots,
        seed,
        noise_arr,
        magic_arr,
        use_phase_twirl,
        prep_gates_noisy,
        prep_qec_after_gate,
        measurement_basis_noisy,
        fallback_on_unverified,
        use_pec,
        cumulative_q,
        q_signs,
        gamma,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """Legacy repeated-T PEC check, retained for compatibility."""
        total = 0.0
        total2 = 0.0
        base_state = np.uint64(seed) + np.uint64(0xA24BAED4963EE407)

        for shot in range(int(shots)):
            state = base_state + np.uint64(shot) * np.uint64(0x9FB21C651E98DF25)
            state = _rng_next_u64(_rng_next_u64(state))
            xd = np.uint16(0)
            zd = np.uint16(0)
            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            d0 = complex(inv_sqrt2, 0.0)
            d1 = complex(inv_sqrt2, 0.0)
            xd, zd, d0, d1, state, _u, _r = _prepare_noisy_input(
                0, xd, zd, d0, d1, state, noise_arr,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )
            weight = 1.0
            previous_twirl_k = 0
            outgoing_t_pauli = 0

            for _cycle in range(int(depth)):
                current_twirl_k = 0
                correction_offset = 0
                if use_phase_twirl:
                    state, current_twirl_k = _rng_int(state, 4)
                    correction_offset = (
                        int(current_twirl_k) - int(previous_twirl_k)
                    ) & 3

                (
                    xd, zd, d0, d1, state,
                    _u, _e, _r, _m, _corr,
                ) = _one_t_cycle(
                    xd, zd, d0, d1, state, noise_arr, magic_arr,
                    correction_offset, measurement_basis_noisy, fallback_on_unverified,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )
                outgoing_t_pauli = 0
                if use_pec:
                    state, outgoing_t_pauli, qweight = _sample_inverse_pauli_numba(
                        state, cumulative_q, q_signs, gamma
                    )
                    d0, d1 = _apply_pauli_1q_state(
                        d0, d1, outgoing_t_pauli
                    )
                    outgoing_t_pauli = 0
                    weight *= qweight
                previous_twirl_k = current_twirl_k

            if use_phase_twirl and int(depth) > 0:
                xd, zd, d0, d1 = _absorb_final_t_frame(
                    xd, zd, d0, d1, previous_twirl_k, outgoing_t_pauli
                )

            state, out = _measure_output(
                xd, zd, d0, d1, state, MEAS_X, noise_arr, measurement_basis_noisy
            )
            value = weight * out
            total += value
            total2 += value * value

        n = int(shots)
        if n <= 0:
            return np.nan, np.nan
        mean = total / n
        if n > 1:
            var = max((total2 - n * mean * mean) / (n - 1), 0.0)
            stderr = math.sqrt(var / n)
        else:
            stderr = 0.0
        return mean, stderr




if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _simulate_random_ct_sequence_kernel(
        gate_codes,
        shots,
        seed,
        noise_arr,
        magic_arr,
        use_phase_twirl,
        prep_gates_noisy,
        prep_qec_after_gate,
        measurement_basis_noisy,
        fallback_on_unverified,
        use_pec,
        cliff_cumulative,
        cliff_signs,
        cliff_gamma,
        t_cumulative,
        t_signs,
        t_gamma,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """
        Simulate one fixed random circuit

            C_1 -> T_1 -> ... -> C_L -> T_L.

        The previous T post-twirl and T-PEC Pauli are compiled into the current
        Clifford.  The current Clifford PEC Pauli is included in that same
        logical gate before its single physical noise channel.  The current T
        pre-twirl is compiled into the injection correction slot.
        """
        total = 0.0
        total2 = 0.0
        abs_weight_total = 0.0
        base_state = np.uint64(seed) + np.uint64(0xB5AD4ECEDA1CE2A9)

        for shot in range(int(shots)):
            state = base_state + np.uint64(shot) * np.uint64(0x9E3779B97F4A7C15)
            state = _rng_next_u64(_rng_next_u64(state))
            xd = np.uint16(0)
            zd = np.uint16(0)
            d0 = complex(1.0, 0.0)
            d1 = complex(0.0, 0.0)
            xd, zd, d0, d1, state, _u, _r = _prepare_noisy_input(
                2, xd, zd, d0, d1, state, noise_arr,
                prep_gates_noisy, prep_qec_after_gate, fallback_on_unverified,
                op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
            )

            weight = 1.0
            previous_twirl_k = 0
            incoming_t_pauli = 0

            for layer in range(gate_codes.shape[0]):
                clifford_pec_pauli = 0
                if use_pec:
                    state, clifford_pec_pauli, qweight = _sample_inverse_pauli_numba(
                        state, cliff_cumulative, cliff_signs, cliff_gamma
                    )
                    weight *= qweight

                (
                    xd, zd, d0, d1, state, _unv, _rounds,
                ) = _apply_compiled_clifford_and_ec(
                    xd, zd, d0, d1, state,
                    int(gate_codes[layer]),
                    int(incoming_t_pauli),
                    int(previous_twirl_k),
                    int(clifford_pec_pauli),
                    noise_arr, fallback_on_unverified,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )

                current_twirl_k = 0
                if use_phase_twirl:
                    state, current_twirl_k = _rng_int(state, 4)

                (
                    xd, zd, d0, d1, state,
                    _u, _e, _r, _m, _corr,
                ) = _one_t_cycle(
                    xd, zd, d0, d1, state, noise_arr, magic_arr,
                    int(current_twirl_k), measurement_basis_noisy, fallback_on_unverified,
                    op_type, q1, q2, round_id, basis_id, check_id, kind_id,
                    min_rounds, max_rounds, key_codes, rx_masks, rz_masks,
                )

                incoming_t_pauli = 0
                if use_pec:
                    state, incoming_t_pauli, qweight = _sample_inverse_pauli_numba(
                        state, t_cumulative, t_signs, t_gamma
                    )
                    weight *= qweight
                previous_twirl_k = current_twirl_k

            # The final P_T S^(-k) is absorbed into the Z measurement frame.
            if gate_codes.shape[0] > 0:
                xd, zd, d0, d1 = _absorb_final_t_frame(
                    xd, zd, d0, d1,
                    previous_twirl_k,
                    incoming_t_pauli,
                )

            state, out = _measure_output(
                xd, zd, d0, d1, state, MEAS_Z, noise_arr, measurement_basis_noisy
            )
            indicator_zero = 1.0 if out > 0 else 0.0
            value = weight * indicator_zero
            total += value
            total2 += value * value
            abs_weight_total += abs(weight)

        n = int(shots)
        if n <= 0:
            return np.nan, np.nan, np.nan
        mean = total / n
        if n > 1:
            var = max((total2 - n * mean * mean) / (n - 1), 0.0)
            stderr = math.sqrt(var / n)
        else:
            stderr = 0.0
        return mean, stderr, abs_weight_total / n


# =============================================================================
# Python wrappers
# =============================================================================


def _t_require_numba() -> None:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The logical T simulator requires Numba.")


def warmup_t_numba(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    seed: int,
) -> Tuple[float, float, float, float, float]:
    """Compile the T-gate Numba path using notebook-supplied model/options/seed."""
    _t_require_numba()
    model = model.validated()
    data = _t_compiled(decoder)
    return _simulate_t_csb_point_kernel(
        0,
        1,
        2,
        int(seed),
        _t_noise_array(model.gadget_noise),
        _magic_array(model),
        bool(model.phase_twirl),
        bool(options["prep_gates_noisy"]),
        bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        data["op_type"],
        data["q1"],
        data["q2"],
        data["round_id"],
        data["basis_id"],
        data["check_id"],
        data["kind_id"],
        int(data["min_rounds"]),
        int(data["max_rounds"]),
        data["key_codes"],
        data["rx_masks"],
        data["rz_masks"],
    )


def simulate_t_csb_point(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    branch_name: str,
    depth: int,
    shots: int,
    seed: int,
) -> Tuple[float, float, float, float, float]:
    _t_require_numba()
    model = model.validated()
    data = _t_compiled(decoder)
    return _simulate_t_csb_point_kernel(
        int(T_CSB_BRANCHES[branch_name]["code"]),
        int(depth),
        int(shots),
        int(seed),
        _t_noise_array(model.gadget_noise),
        _magic_array(model),
        bool(model.phase_twirl),
        bool(options["prep_gates_noisy"]),
        bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        data["op_type"],
        data["q1"],
        data["q2"],
        data["round_id"],
        data["basis_id"],
        data["check_id"],
        data["kind_id"],
        int(data["min_rounds"]),
        int(data["max_rounds"]),
        data["key_codes"],
        data["rx_masks"],
        data["rz_masks"],
    )


def run_t_csb_spectrum(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    """Run the four CSB branches for a complete logical T-gate cycle."""
    _t_require_numba()
    model = model.validated()
    steps = [int(s) for s in steps]
    rep = int(rep)
    shots = int(shots)
    chunk_shots = max(1, int(chunk_shots))
    rng = np.random.default_rng(seed)

    tasks: List[Dict[str, Any]] = []
    point_id = 0
    for branch_name, spec in T_CSB_BRANCHES.items():
        branch_total = max(1, shots // 2)
        for step in steps:
            depth = rep * step
            for chunk_id, n in enumerate(_t_chunk_sizes(branch_total, chunk_shots)):
                tasks.append(
                    {
                        "point_id": point_id,
                        "chunk_id": chunk_id,
                        "branch": branch_name,
                        "branch_kind": spec["kind"],
                        "initial": spec["initial"],
                        "measure": spec["measure"],
                        "step": step,
                        "rep": rep,
                        "depth": depth,
                        "shots": n,
                        "requested_total_shots": shots,
                        "seed": int(rng.integers(2**32 - 1)),
                    }
                )
            point_id += 1

    tasks.sort(key=lambda t: int(t["depth"]) * int(t["shots"]), reverse=True)

    if verbose:
        print("Logical T CSB: true depth L = rep * step")
        print("  steps            =", steps)
        print("  rep              =", rep)
        print("  true depths      =", [rep * s for s in steps])
        print("  requested shots  =", shots)
        print("  shots per branch =", max(1, shots // 2))
        print("  chunk shots      =", chunk_shots)
        print("  n_jobs           =", n_jobs)
        print("  model            =", model)

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    def run_task(task):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        mean, se, unv, rounds, m1_rate = simulate_t_csb_point(
            dec,
            model,
            options,
            task["branch"],
            task["depth"],
            task["shots"],
            task["seed"],
        )
        return {
            **task,
            "mean": mean,
            "stderr": se,
            "unverified_rate": unv,
            "average_ec_rounds": rounds,
            "m1_rate": m1_rate,
        }

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(run_task)(task) for task in tasks
        )
    else:
        rows = [run_task(task) for task in tasks]

    chunk_df = pd.DataFrame(rows)
    combined_rows = []
    for _, sub in chunk_df.groupby("point_id", sort=False):
        agg = _aggregate_sample_chunks(sub)
        first = sub.iloc[0]
        weights = sub["shots"].to_numpy(dtype=float)
        combined_rows.append(
            {
                "branch": first["branch"],
                "branch_kind": first["branch_kind"],
                "initial": first["initial"],
                "measure": first["measure"],
                "step": int(first["step"]),
                "rep": int(first["rep"]),
                "depth": int(first["depth"]),
                "mean": agg["mean"],
                "stderr": agg["stderr"],
                "shots": agg["shots"],
                "requested_total_shots": int(first["requested_total_shots"]),
                "unverified_rate": float(np.average(sub["unverified_rate"], weights=weights)),
                "average_ec_rounds": float(np.average(sub["average_ec_rounds"], weights=weights)),
                "m1_rate": float(np.average(sub["m1_rate"], weights=weights)),
                "n_chunks": int(len(sub)),
            }
        )

    out = pd.DataFrame(combined_rows).sort_values(["branch", "step"]).reset_index(drop=True)
    if verbose:
        for _, row in out.iterrows():
            print(
                f"T-CSB {row.branch:17s} step {int(row.step):3d} "
                f"rep {int(row.rep):3d} depth {int(row.depth):5d} "
                f"shots {int(row.shots):6d} mean {row['mean']:+.6f} "
                f"stderr {row.stderr:.6f} unv {row.unverified_rate:.3g}"
            )
    return out


# =============================================================================
# Spectrum fitting and phase-covariant reconstruction
# =============================================================================


def merge_t_transverse_quadratures(csb_df: pd.DataFrame) -> pd.DataFrame:
    real = csb_df[csb_df.branch == "transverse_real"].sort_values("depth")
    imag = csb_df[csb_df.branch == "transverse_imag"].sort_values("depth")
    cols = ["step", "rep", "depth"]
    out = real[cols + ["mean", "stderr", "shots"]].merge(
        imag[cols + ["mean", "stderr", "shots"]],
        on=cols,
        suffixes=("_real", "_imag"),
    )
    out["complex_signal"] = (
        out["mean_real"].to_numpy(dtype=float)
        + 1j * out["mean_imag"].to_numpy(dtype=float)
    )
    out["magnitude"] = np.abs(out["complex_signal"].to_numpy())
    return out


def t_axis_table(csb_df: pd.DataFrame) -> pd.DataFrame:
    z0 = csb_df[csb_df.branch == "axis_0"].sort_values("depth")
    z1 = csb_df[csb_df.branch == "axis_1"].sort_values("depth")
    cols = ["step", "rep", "depth"]
    out = z0[cols + ["mean", "stderr", "shots"]].merge(
        z1[cols + ["mean", "stderr", "shots"]],
        on=cols,
        suffixes=("_z0", "_z1"),
    )
    out["differential"] = 0.5 * (out["mean_z0"] - out["mean_z1"])
    out["common_mode"] = 0.5 * (out["mean_z0"] + out["mean_z1"])
    out["differential_stderr"] = 0.5 * np.sqrt(
        out["stderr_z0"] ** 2 + out["stderr_z1"] ** 2
    )
    out["common_mode_stderr"] = out["differential_stderr"]
    return out


def _fit_nonunital_t(axis_df: pd.DataFrame, f: float) -> Dict[str, float]:
    L = axis_df["depth"].to_numpy(dtype=float)
    y = axis_df["common_mode"].to_numpy(dtype=float)
    sigma = np.maximum(
        axis_df["common_mode_stderr"].to_numpy(dtype=float), 1e-12
    )
    if abs(1.0 - f) < 1e-10:
        h = L
    else:
        h = (1.0 - np.power(float(f), L)) / (1.0 - float(f))
    mask = np.isfinite(h) & np.isfinite(y) & np.isfinite(sigma) & (L > 0)
    h = h[mask]
    y = y[mask]
    sigma = sigma[mask]
    if len(h) == 0 or np.all(np.abs(h) < 1e-15):
        return {"t": np.nan, "t_stderr": np.nan}
    w = 1.0 / (sigma * sigma)
    denom = float(np.sum(w * h * h))
    t = float(np.sum(w * h * y) / denom)
    t_se = math.sqrt(1.0 / denom)
    return {"t": t, "t_stderr": t_se}


def phase_covariant_pauli_from_gf(
    g: float, f: float, project_simplex: bool = True
) -> Dict[str, float]:
    raw = np.array(
        [
            (1.0 + f + 2.0 * g) / 4.0,
            (1.0 - f) / 4.0,
            (1.0 - f) / 4.0,
            (1.0 + f - 2.0 * g) / 4.0,
        ],
        dtype=float,
    )
    physical = simplex_project(raw) if project_simplex else raw.copy()
    return {
        "raw_p_I": float(raw[0]),
        "raw_p_X": float(raw[1]),
        "raw_p_Y": float(raw[2]),
        "raw_p_Z": float(raw[3]),
        "p_I": float(physical[0]),
        "p_X": float(physical[1]),
        "p_Y": float(physical[2]),
        "p_Z": float(physical[3]),
        "simplex_projection_distance": float(np.linalg.norm(physical - raw)),
    }


def inverse_phase_covariant_pauli_from_gf(
    g: float, f: float, eps: float = 1e-12
) -> Dict[str, float]:
    g_safe = float(g)
    f_safe = float(f)
    if abs(g_safe) < eps:
        g_safe = eps if g_safe >= 0 else -eps
    if abs(f_safe) < eps:
        f_safe = eps if f_safe >= 0 else -eps
    return {
        "I": 0.25 * (1.0 + 2.0 / g_safe + 1.0 / f_safe),
        "X": 0.25 * (1.0 - 1.0 / f_safe),
        "Y": 0.25 * (1.0 - 1.0 / f_safe),
        "Z": 0.25 * (1.0 - 2.0 / g_safe + 1.0 / f_safe),
    }


def fit_t_csb_spectra(csb_df: pd.DataFrame) -> Dict[str, Any]:
    q = merge_t_transverse_quadratures(csb_df)
    complex_fit = fit_complex_csb_eigenvalue(q, omega0=np.pi / 4.0)

    axis = t_axis_table(csb_df)
    axis_fit = fit_axis_decay_signed_continuous(
        axis["depth"].to_numpy(dtype=float),
        axis["differential"].to_numpy(dtype=float),
        axis["differential_stderr"].to_numpy(dtype=float),
    )

    g = float(complex_fit["g"])
    delta = float(complex_fit["delta_theta"])
    f = float(axis_fit["f"])
    nonunital = _fit_nonunital_t(axis, f)
    fidelity = process_fidelity_from_complex_offdiag(g, delta, f)
    pauli = phase_covariant_pauli_from_gf(g, f, project_simplex=True)
    inverse_q = inverse_phase_covariant_pauli_from_gf(g, f)
    gamma = float(sum(abs(v) for v in inverse_q.values()))

    summary = pd.DataFrame(
        [
            {
                "g": g,
                "g_local_stderr": float(complex_fit.get("g_stderr", np.nan)),
                "theta": float(complex_fit["theta"]),
                "delta_theta": delta,
                "delta_theta_local_stderr": float(
                    complex_fit.get("delta_theta_stderr", np.nan)
                ),
                "f": f,
                "f_local_stderr": float(axis_fit.get("f_stderr", np.nan)),
                "t": float(nonunital["t"]),
                "t_local_stderr": float(nonunital["t_stderr"]),
                "process_fidelity": float(fidelity["process_fidelity"]),
                "process_infidelity": float(fidelity["process_infidelity"]),
                "average_gate_fidelity": float(fidelity["average_gate_fidelity"]),
                "average_gate_infidelity": float(
                    fidelity["average_gate_infidelity"]
                ),
                **pauli,
                "q_I": float(inverse_q["I"]),
                "q_X": float(inverse_q["X"]),
                "q_Y": float(inverse_q["Y"]),
                "q_Z": float(inverse_q["Z"]),
                "gamma_per_cycle": gamma,
                "sampling_overhead_per_cycle": gamma * gamma,
            }
        ]
    )
    return {
        "summary_df": summary,
        "transverse_df": q,
        "axis_df": axis,
        "complex_fit": complex_fit,
        "axis_fit": axis_fit,
        "inverse_q": inverse_q,
    }


# =============================================================================
# Outcome-resolved resource-angle diagnostic
# =============================================================================




# =============================================================================
# PEC validation
# =============================================================================


def _inverse_sampling_arrays(
    inverse_q: Mapping[str, float]
) -> Tuple[np.ndarray, np.ndarray, float]:
    q = np.array(
        [inverse_q["I"], inverse_q["X"], inverse_q["Y"], inverse_q["Z"]],
        dtype=float,
    )
    gamma = float(np.sum(np.abs(q)))
    probs = np.abs(q) / gamma
    cumulative = np.cumsum(probs)
    cumulative[-1] = 1.0
    signs = np.where(q >= 0.0, 1.0, -1.0)
    return cumulative.astype(np.float64), signs.astype(np.float64), gamma


def simulate_t_pec_point(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    depth: int,
    shots: int,
    seed: int,
    inverse_q: Optional[Mapping[str, float]],
) -> Tuple[float, float]:
    _t_require_numba()
    model = model.validated()
    data = _t_compiled(decoder)
    if inverse_q is None:
        cumulative = np.ones(4, dtype=np.float64)
        signs = np.ones(4, dtype=np.float64)
        gamma = 1.0
        use_pec = False
    else:
        cumulative, signs, gamma = _inverse_sampling_arrays(inverse_q)
        use_pec = True
    return _simulate_t_pec_point_kernel(
        int(depth), int(shots), int(seed),
        _t_noise_array(model.gadget_noise), _magic_array(model),
        bool(model.phase_twirl),
        bool(options["prep_gates_noisy"]), bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]),
        bool(options["fallback_on_unverified"]),
        bool(use_pec), cumulative, signs, float(gamma),
        data["op_type"], data["q1"], data["q2"], data["round_id"],
        data["basis_id"], data["check_id"], data["kind_id"],
        int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def run_t_pec_validation(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    inverse_q: Mapping[str, float],
    depths: Sequence[int],
    noisy_shots: int,
    pec_shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    depths = [int(d) for d in depths]
    if any(d % 8 != 0 for d in depths):
        raise ValueError("Repeated-T PEC depths must be multiples of 8.")
    rng = np.random.default_rng(seed)
    tasks = []
    for depth in depths:
        for mode, total_shots in (("noisy", noisy_shots), ("pec", pec_shots)):
            for chunk_id, n in enumerate(_t_chunk_sizes(total_shots, chunk_shots)):
                tasks.append({"depth": depth, "mode": mode, "chunk_id": chunk_id,
                              "shots": n, "seed": int(rng.integers(2**32 - 1))})
    tasks.sort(key=lambda t: int(t["depth"]) * int(t["shots"]), reverse=True)
    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)
    def run_task(t):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        inv = inverse_q if t["mode"] == "pec" else None
        mean, se = simulate_t_pec_point(dec, model, options, t["depth"], t["shots"], t["seed"], inv)
        n = int(t["shots"])
        sample_var = se * se * n if n > 1 else 0.0
        return {**t, "sum": mean * n, "sumsq": sample_var * max(n - 1, 0) + n * mean * mean}
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(delayed(run_task)(t) for t in tasks)
    else:
        rows = [run_task(t) for t in tasks]
    chunk_df = pd.DataFrame(rows)
    out = []
    for depth in depths:
        row = {"depth": depth, "ideal_expectation": 1.0}
        for mode in ("noisy", "pec"):
            sub = chunk_df[(chunk_df.depth == depth) & (chunk_df["mode"] == mode)]
            n = int(sub.shots.sum()); s = float(sub["sum"].sum()); ss = float(sub["sumsq"].sum())
            mean = s / max(n, 1)
            var = max((ss - s * s / n) / (n - 1), 0.0) if n > 1 else 0.0
            row[f"{mode}_expectation"] = mean
            row[f"{mode}_stderr"] = math.sqrt(var / n) if n > 1 else 0.0
            row[f"{mode}_shots"] = n
        out.append(row)
        if verbose:
            print(f"depth {depth:3d} | noisy {row['noisy_expectation']:+.6f} ± {row['noisy_stderr']:.6f} | PEC {row['pec_expectation']:+.6f} ± {row['pec_stderr']:.6f}")
    return pd.DataFrame(out)


# =============================================================================
# Compiled-twirl random Clifford-plus-T PEC
# =============================================================================

RANDOM_CLIFFORD_LABELS = ("RX90", "RY90", "RZ90")
_RANDOM_CLIFFORD_TO_CODE = {name: i for i, name in enumerate(RANDOM_CLIFFORD_LABELS)}


def _apply_ideal_gate_python(state: np.ndarray, gate_code: int) -> np.ndarray:
    a0, a1 = complex(state[0]), complex(state[1])
    if int(gate_code) == 0:
        inv = 1.0 / math.sqrt(2.0)
        return np.array([inv * (a0 - 1j * a1), inv * (-1j * a0 + a1)])
    if int(gate_code) == 1:
        inv = 1.0 / math.sqrt(2.0)
        return np.array([inv * (a0 - a1), inv * (a0 + a1)])
    return np.array([a0, 1j * a1])


def ideal_random_ct_probability(gate_codes: Sequence[int]) -> float:
    """Ideal P(0) for |0> followed by C_1,T_1,...,C_L,T_L."""
    state = np.array([1.0 + 0.0j, 0.0 + 0.0j])
    t_phase = complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
    for gate_code in gate_codes:
        state = _apply_ideal_gate_python(state, int(gate_code))
        state[1] *= t_phase
    return float(abs(state[0]) ** 2)


def learn_clifford_pauli_model(
    decoder,
    noise: NoiseParams,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    chunk_shots: int,
    seed: int,
    n_jobs: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> Dict[str, Any]:
    csb_df = run_csb_spectrum(
        decoder=decoder, noise=noise, options=options, steps=steps, rep=rep, shots=shots,
        seed=seed, n_jobs=n_jobs, chunk_shots=chunk_shots,
        target_gates=("RZ90", "RX90", "RY90"), cache_file=cache_file,
        min_rounds=min_rounds, max_rounds=max_rounds, verbose=verbose,
    )
    spectra_df = fit_note_style_spectra(csb_df)
    learned, design_matrix, data_vector = solve_pauli_from_rz_rx_spectra(spectra_df, enforce_simplex=True)
    validation = validate_with_ry_spectra(spectra_df, learned)
    pauli_df = pd.DataFrame([{**learned, **validation}])
    probabilities = np.array([learned["p_I"], learned["p_X"], learned["p_Y"], learned["p_Z"]], dtype=float)
    inverse_q = inverse_pauli_quasiprobabilities_from_probs(probabilities)
    gamma = float(sum(abs(v) for v in inverse_q.values()))
    for label in ("I", "X", "Y", "Z"):
        pauli_df[f"q_{label}"] = float(inverse_q[label])
    pauli_df["gamma_per_cycle"] = gamma
    pauli_df["sampling_overhead_per_cycle"] = gamma * gamma
    return {"csb_df": csb_df, "spectra_df": spectra_df, "pauli_df": pauli_df,
            "inverse_q": inverse_q, "design_matrix": design_matrix, "data_vector": data_vector}


def simulate_random_ct_sequence(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    gate_codes: Sequence[int],
    shots: int,
    seed: int,
    clifford_inverse_q: Optional[Mapping[str, float]],
    t_inverse_q: Optional[Mapping[str, float]],
) -> Tuple[float, float, float]:
    _t_require_numba()
    model = model.validated()
    data = _t_compiled(decoder)
    use_pec = clifford_inverse_q is not None or t_inverse_q is not None
    if use_pec and (clifford_inverse_q is None or t_inverse_q is None):
        raise ValueError("Complete PEC requires both Clifford and T inverse models.")
    if use_pec:
        c_cum, c_sign, c_gamma = _inverse_sampling_arrays(clifford_inverse_q)
        t_cum, t_sign, t_gamma = _inverse_sampling_arrays(t_inverse_q)
    else:
        c_cum = np.ones(4); c_sign = np.ones(4); c_gamma = 1.0
        t_cum = np.ones(4); t_sign = np.ones(4); t_gamma = 1.0
    return _simulate_random_ct_sequence_kernel(
        np.asarray(gate_codes, dtype=np.int64), int(shots), int(seed),
        _t_noise_array(model.gadget_noise), _magic_array(model), bool(model.phase_twirl),
        bool(options["prep_gates_noisy"]), bool(options["prep_qec_after_gate"]),
        bool(options["measurement_basis_gates_noisy"]), bool(options["fallback_on_unverified"]),
        bool(use_pec), c_cum, c_sign, float(c_gamma), t_cum, t_sign, float(t_gamma),
        data["op_type"], data["q1"], data["q2"], data["round_id"], data["basis_id"],
        data["check_id"], data["kind_id"], int(data["min_rounds"]), int(data["max_rounds"]),
        data["key_codes"], data["rx_masks"], data["rz_masks"],
    )


def run_random_circuit_pec_validation(
    decoder,
    model: TGateNoiseModel,
    options: Mapping[str, bool],
    clifford_inverse_q: Mapping[str, float],
    t_inverse_q: Mapping[str, float],
    depths: Sequence[int],
    sequences_per_depth: int,
    noisy_shots: int,
    pec_shots: int,
    seed: int,
    n_jobs: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> Dict[str, pd.DataFrame]:
    depths = [int(d) for d in depths]
    rng = np.random.default_rng(seed)
    specs = []
    for depth in depths:
        for sequence_id in range(int(sequences_per_depth)):
            codes = rng.integers(0, 3, size=depth, dtype=np.int64)
            specs.append({"depth": depth, "sequence_id": sequence_id, "gate_codes": codes,
                          "sequence": " ".join(RANDOM_CLIFFORD_LABELS[int(c)] for c in codes),
                          "ideal_probability": ideal_random_ct_probability(codes),
                          "noisy_seed": int(rng.integers(2**32 - 1)), "pec_seed": int(rng.integers(2**32 - 1))})
    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)
    def one(spec, mode):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        if mode == "pec":
            mean, se, w = simulate_random_ct_sequence(dec, model, options, spec["gate_codes"], pec_shots, spec["pec_seed"], clifford_inverse_q, t_inverse_q)
        else:
            mean, se, w = simulate_random_ct_sequence(dec, model, options, spec["gate_codes"], noisy_shots, spec["noisy_seed"], None, None)
        return {"depth": spec["depth"], "sequence_id": spec["sequence_id"], "mode": mode,
                "probability": mean, "stderr": se, "mean_abs_weight": w}
    tasks = [(s,m) for s in specs for m in ("noisy","pec")]
    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks)>1:
        rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(delayed(one)(s,m) for s,m in tasks)
    else:
        rows = [one(s,m) for s,m in tasks]
    mode_df = pd.DataFrame(rows)
    wide = mode_df.pivot(index=["depth","sequence_id"], columns="mode", values=["probability","stderr","mean_abs_weight"])
    wide.columns = [f"{m}_{q}" for q,m in wide.columns]; wide=wide.reset_index()
    meta = pd.DataFrame([{"depth":s["depth"],"sequence_id":s["sequence_id"],"sequence":s["sequence"],
                          "ideal_probability":s["ideal_probability"]} for s in specs])
    seq = meta.merge(wide,on=["depth","sequence_id"],how="left")
    for m in ("noisy","pec"):
        seq[f"{m}_abs_error"] = np.abs(seq[f"{m}_probability"]-seq["ideal_probability"])
        seq[f"{m}_signed_error"] = seq[f"{m}_probability"]-seq["ideal_probability"]
    summary = seq.groupby("depth",as_index=False).agg(
        n_sequences=("sequence_id","nunique"), ideal_probability_mean=("ideal_probability","mean"),
        noisy_abs_error_mean=("noisy_abs_error","mean"), noisy_abs_error_std=("noisy_abs_error","std"),
        pec_abs_error_mean=("pec_abs_error","mean"), pec_abs_error_std=("pec_abs_error","std"),
        noisy_signed_error_mean=("noisy_signed_error","mean"), pec_signed_error_mean=("pec_signed_error","mean"),
        noisy_probability_mean=("noisy_probability","mean"), pec_probability_mean=("pec_probability","mean"),
        pec_mean_abs_weight=("pec_mean_abs_weight","mean"),
    ).sort_values("depth").reset_index(drop=True)
    summary[["noisy_abs_error_std","pec_abs_error_std"]] = summary[["noisy_abs_error_std","pec_abs_error_std"]].fillna(0.0)
    if verbose: print(summary.to_string(index=False))
    return {"sequence_df":seq,"summary_df":summary}






# =============================================================================
# Analytic resource-only predictions and diagnostics
# =============================================================================


def analytic_resource_only_parameters(
    magic_angle_error: float,
    magic_error_rate: float,
    magic_nu: Mapping[str, float],
) -> Dict[str, float]:
    r = float(magic_error_rate)
    nu_x = float(magic_nu.get("X", 0.0))
    nu_y = float(magic_nu.get("Y", 0.0))
    nu_z = float(magic_nu.get("Z", 0.0))
    eps = float(magic_angle_error)

    # Exact resource-only expression.  Before the accepted-state Pauli mixture,
    # the resource Bloch vector is
    #   (cos(pi/4+eps), sin(pi/4+eps), 0).
    # X, Y, and Z errors flip the corresponding equatorial components.  The
    # ideal consuming circuit depends only on mu=(r_x+r_y)/sqrt(2).
    rx = math.cos(math.pi / 4.0 + eps)
    ry = math.sin(math.pi / 4.0 + eps)
    rx_after = (1.0 - 2.0 * r * (nu_y + nu_z)) * rx
    ry_after = (1.0 - 2.0 * r * (nu_x + nu_z)) * ry
    mu = (rx_after + ry_after) / math.sqrt(2.0)
    return {
        "g": float(mu),
        "f": 1.0,
        "delta_theta": 0.0,
        "t": 0.0,
        "process_fidelity": 0.5 * (1.0 + float(mu)),
        "process_infidelity": 0.5 * (1.0 - float(mu)),
        "effective_Z_probability": 0.5 * (1.0 - float(mu)),
    }




# =============================================================================
# Smoke tests
# =============================================================================




# =============================================================================
# Universal C-CX-C-T logical PEC
# =============================================================================


LOCAL_GATE_LABELS = ("RX90", "RY90", "RZ90")
LOCAL_GATE_TO_CODE = {name: i for i, name in enumerate(LOCAL_GATE_LABELS)}
BITSTRING_LABELS = ("00", "01", "10", "11")


@dataclass(frozen=True)
class UniversalNoiseModel:
    """Physical circuit noise and accepted-magic-state noise."""

    physical_noise: NoiseParams
    magic_angle_error: float
    magic_error_rate: float
    magic_nu_x: float
    magic_nu_y: float
    magic_nu_z: float
    t_phase_twirl: bool
    cx_phase_twirl: bool

    def validated(self) -> "UniversalNoiseModel":
        r = float(self.magic_error_rate)
        if not 0.0 <= r <= 1.0:
            raise ValueError("magic_error_rate must be in [0,1].")
        nu = np.asarray([self.magic_nu_x, self.magic_nu_y, self.magic_nu_z], dtype=float)
        if np.any(nu < -1e-15):
            raise ValueError("Magic-state Pauli composition must be nonnegative.")
        if r > 0.0 and not np.isclose(np.sum(nu), 1.0, atol=1e-10):
            raise ValueError("magic_nu_x + magic_nu_y + magic_nu_z must equal 1.")
        return self

    def t_model(self) -> TGateNoiseModel:
        return TGateNoiseModel(
            gadget_noise=self.physical_noise,
            magic_angle_error=float(self.magic_angle_error),
            magic_error_rate=float(self.magic_error_rate),
            magic_nu_x=float(self.magic_nu_x),
            magic_nu_y=float(self.magic_nu_y),
            magic_nu_z=float(self.magic_nu_z),
            phase_twirl=bool(self.t_phase_twirl),
        ).validated()


def make_universal_noise_model(
    physical_strength: float,
    two_qubit_factor: float,
    final_meas_factor: float,
    magic_angle_error: float,
    magic_error_rate: float,
    magic_nu: Mapping[str, float],
    t_phase_twirl: bool,
    cx_phase_twirl: bool,
) -> UniversalNoiseModel:
    return UniversalNoiseModel(
        physical_noise=make_homogeneous_noise(physical_strength, two_qubit_factor, final_meas_factor),
        magic_angle_error=float(magic_angle_error),
        magic_error_rate=float(magic_error_rate),
        magic_nu_x=float(magic_nu.get("X", 0.0)),
        magic_nu_y=float(magic_nu.get("Y", 0.0)),
        magic_nu_z=float(magic_nu.get("Z", 0.0)),
        t_phase_twirl=bool(t_phase_twirl),
        cx_phase_twirl=bool(cx_phase_twirl),
    ).validated()


def _universal_noise_array(noise: NoiseParams) -> np.ndarray:
    return np.asarray(
        [
            noise.p_1q,
            noise.p_2q,
            noise.p_meas,
            noise.p_reset,
            noise.p_idle,
            noise.p_final_meas,
        ],
        dtype=np.float64,
    )


def _universal_magic_array(model: UniversalNoiseModel) -> np.ndarray:
    return np.asarray(
        [
            model.magic_angle_error,
            model.magic_error_rate,
            model.magic_nu_x,
            model.magic_nu_y,
            model.magic_nu_z,
        ],
        dtype=np.float64,
    )


def _inverse_1q_arrays(
    inverse_q: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    return _inverse_sampling_arrays(inverse_q)


def _inverse_2q_arrays(
    inverse_q: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, float]:
    return _prepare_inverse_sampling_arrays(inverse_q)


def _universal_compiled(decoder) -> Dict[str, Any]:
    return get_compiled_numba_data(decoder)


def _universal_require_numba() -> None:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("The universal simulator requires Numba.")


if NUMBA_AVAILABLE:

    @nb.njit(cache=True)
    def _apply_pauli_2q_state(a00, a01, a10, a11, qubit, pcode):
        p = int(pcode)
        q = int(qubit)
        if p == 0:
            return a00, a01, a10, a11
        if q == 0:
            if p == 1:
                return a10, a11, a00, a01
            if p == 2:
                return -1j * a10, -1j * a11, 1j * a00, 1j * a01
            return a00, a01, -a10, -a11
        if p == 1:
            return a01, a00, a11, a10
        if p == 2:
            return -1j * a01, 1j * a00, -1j * a11, 1j * a10
        return a00, -a01, a10, -a11


    @nb.njit(cache=True)
    def _apply_s_power_2q_state(a00, a01, a10, a11, qubit, exponent):
        e = int(exponent) & 3
        if e == 0:
            return a00, a01, a10, a11
        if e == 1:
            phase = 1j
        elif e == 2:
            phase = -1.0 + 0.0j
        else:
            phase = -1j
        if int(qubit) == 0:
            return a00, a01, phase * a10, phase * a11
        return a00, phase * a01, a10, phase * a11


    @nb.njit(cache=True)
    def _apply_rx_quarter_2q_state(a00, a01, a10, a11, qubit, sign):
        inv = 1.0 / math.sqrt(2.0)
        s = 1 if int(sign) >= 0 else -1
        if int(qubit) == 0:
            b00 = inv * (a00 - 1j * s * a10)
            b01 = inv * (a01 - 1j * s * a11)
            b10 = inv * (-1j * s * a00 + a10)
            b11 = inv * (-1j * s * a01 + a11)
            return b00, b01, b10, b11
        b00 = inv * (a00 - 1j * s * a01)
        b01 = inv * (-1j * s * a00 + a01)
        b10 = inv * (a10 - 1j * s * a11)
        b11 = inv * (-1j * s * a10 + a11)
        return b00, b01, b10, b11


    @nb.njit(cache=True)
    def _apply_ry90_2q_state(a00, a01, a10, a11, qubit):
        inv = 1.0 / math.sqrt(2.0)
        if int(qubit) == 0:
            return (
                inv * (a00 - a10),
                inv * (a01 - a11),
                inv * (a00 + a10),
                inv * (a01 + a11),
            )
        return (
            inv * (a00 - a01),
            inv * (a00 + a01),
            inv * (a10 - a11),
            inv * (a10 + a11),
        )


    @nb.njit(cache=True)
    def _apply_base_local_gate(
        a00, a01, a10, a11, x_mask, z_mask, qubit, gate_code
    ):
        g = int(gate_code)
        if g == 0:
            a00, a01, a10, a11 = _apply_rx_quarter_2q_state(
                a00, a01, a10, a11, qubit, 1
            )
            x_mask = np.uint16(x_mask ^ z_mask)
        elif g == 1:
            a00, a01, a10, a11 = _apply_ry90_2q_state(
                a00, a01, a10, a11, qubit
            )
            tmp = x_mask
            x_mask = z_mask
            z_mask = tmp
        else:
            a00, a01, a10, a11 = _apply_s_power_2q_state(
                a00, a01, a10, a11, qubit, 1
            )
            z_mask = np.uint16(z_mask ^ x_mask)
        return a00, a01, a10, a11, x_mask, z_mask


    @nb.njit(cache=True)
    def _apply_s_power_2q_with_frame(
        a00, a01, a10, a11, x_mask, z_mask, qubit, exponent
    ):
        e = int(exponent) & 3
        if e != 0:
            a00, a01, a10, a11 = _apply_s_power_2q_state(
                a00, a01, a10, a11, qubit, e
            )
            if e & 1:
                z_mask = np.uint16(z_mask ^ x_mask)
        return a00, a01, a10, a11, x_mask, z_mask


    @nb.njit(cache=True)
    def _apply_rx_quarter_2q_with_frame(
        a00, a01, a10, a11, x_mask, z_mask, qubit, sign
    ):
        a00, a01, a10, a11 = _apply_rx_quarter_2q_state(
            a00, a01, a10, a11, qubit, sign
        )
        x_mask = np.uint16(x_mask ^ z_mask)
        return a00, a01, a10, a11, x_mask, z_mask


    @nb.njit(cache=True)
    def _canonicalize_block_2q(
        x_mask, z_mask, a00, a01, a10, a11, qubit
    ):
        x_can, z_can, pcode = _canonical_logical_code(x_mask, z_mask)
        a00, a01, a10, a11 = _apply_pauli_2q_state(
            a00, a01, a10, a11, qubit, pcode
        )
        return x_can, z_can, a00, a01, a10, a11, pcode


    @nb.njit(cache=True)
    def _apply_compiled_local_layer_and_ec(
        xc,
        zc,
        xt,
        zt,
        a00,
        a01,
        a10,
        a11,
        state,
        base_c,
        base_t,
        incoming_t_pauli_c,
        incoming_t_pauli_t,
        incoming_t_k_c,
        incoming_t_k_t,
        cx_twirl_c,
        cx_twirl_t,
        cx_inverse_pauli_c,
        cx_inverse_pauli_t,
        local_inverse_pauli_c,
        local_inverse_pauli_t,
        mode,
        noise_arr,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """
        Apply one parallel compiled local layer and one FTEC per block.

        mode=1 is C1.  The chronological ideal sequence on each block is

            previous T post-twirl
              -> previous T inverse Pauli
              -> base C1
              -> current CX pre-twirl.

        Hence the compiled operator is

            V_pre * C1 * P_T(previous) * S^(-k_previous).

        mode=2 is C2.  The chronological ideal sequence on each block is

            current CX post-twirl
              -> current dressed-CX inverse Pauli
              -> base C2
              -> current C2 inverse Pauli.

        Hence the compiled operator is

            P_C2 * C2 * P_DCX * V_post.

        All ideal factors are merged into one logical Clifford slot.  Each
        block receives exactly one p_1q noise layer and one FTEC gadget.
        """
        p1 = noise_arr[0]

        if int(mode) == 1:
            # Previous T output frame, carried into the current C1 layer.
            a00, a01, a10, a11, xc, zc = _apply_s_power_2q_with_frame(
                a00, a01, a10, a11, xc, zc, 0, -int(incoming_t_k_c)
            )
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 0, int(incoming_t_pauli_c)
            )
            a00, a01, a10, a11, xt, zt = _apply_s_power_2q_with_frame(
                a00, a01, a10, a11, xt, zt, 1, -int(incoming_t_k_t)
            )
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 1, int(incoming_t_pauli_t)
            )

            # Base random C1.
            a00, a01, a10, a11, xc, zc = _apply_base_local_gate(
                a00, a01, a10, a11, xc, zc, 0, int(base_c)
            )
            a00, a01, a10, a11, xt, zt = _apply_base_local_gate(
                a00, a01, a10, a11, xt, zt, 1, int(base_t)
            )

            # Current CX pre-twirl is part of the same C1 Clifford slot.
            if int(cx_twirl_c) & 1:
                a00, a01, a10, a11, xc, zc = _apply_s_power_2q_with_frame(
                    a00, a01, a10, a11, xc, zc, 0, 1
                )
            if int(cx_twirl_t) & 1:
                a00, a01, a10, a11, xt, zt = _apply_rx_quarter_2q_with_frame(
                    a00, a01, a10, a11, xt, zt, 1, 1
                )

        else:
            # Current CX post-twirl is the first factor after the noisy CX.
            if int(cx_twirl_c) & 1:
                a00, a01, a10, a11, xc, zc = _apply_s_power_2q_with_frame(
                    a00, a01, a10, a11, xc, zc, 0, 3
                )
            if int(cx_twirl_t) & 1:
                a00, a01, a10, a11, xt, zt = _apply_rx_quarter_2q_with_frame(
                    a00, a01, a10, a11, xt, zt, 1, -1
                )

            # The dressed-CX inverse is defined after the post-twirl.
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 0, int(cx_inverse_pauli_c)
            )
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 1, int(cx_inverse_pauli_t)
            )

            # Base random C2.
            a00, a01, a10, a11, xc, zc = _apply_base_local_gate(
                a00, a01, a10, a11, xc, zc, 0, int(base_c)
            )
            a00, a01, a10, a11, xt, zt = _apply_base_local_gate(
                a00, a01, a10, a11, xt, zt, 1, int(base_t)
            )

            # The C2 inverse sample is the last ideal factor before C2 noise.
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 0, int(local_inverse_pauli_c)
            )
            a00, a01, a10, a11 = _apply_pauli_2q_state(
                a00, a01, a10, a11, 1, int(local_inverse_pauli_t)
            )

        # One physical Clifford noise slot and one FTEC on each block.
        xc, zc, state = _apply_1q_gate_noise_all_data_numba(
            xc, zc, state, p1
        )
        xt, zt, state = _apply_1q_gate_noise_all_data_numba(
            xt, zt, state, p1
        )

        xc, zc, state, unv_c, rounds_c = _run_one_ftec(
            xc,
            zc,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        xc, zc, a00, a01, a10, a11, _pc = _canonicalize_block_2q(
            xc, zc, a00, a01, a10, a11, 0
        )

        xt, zt, state, unv_t, rounds_t = _run_one_ftec(
            xt,
            zt,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        xt, zt, a00, a01, a10, a11, _pt = _canonicalize_block_2q(
            xt, zt, a00, a01, a10, a11, 1
        )

        return (
            xc,
            zc,
            xt,
            zt,
            a00,
            a01,
            a10,
            a11,
            state,
            unv_c + unv_t,
            rounds_c + rounds_t,
        )


    @nb.njit(cache=True)
    def _apply_ideal_cx_2q(a00, a01, a10, a11):
        return a00, a01, a11, a10


    @nb.njit(cache=True)
    def _apply_cx_and_ec_2q(
        xc,
        zc,
        xt,
        zt,
        a00,
        a01,
        a10,
        a11,
        state,
        noise_arr,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        a00, a01, a10, a11 = _apply_ideal_cx_2q(a00, a01, a10, a11)
        xc, zc, xt, zt, state, unv, rounds = _apply_transversal_cx_and_ftec(
            xc,
            zc,
            xt,
            zt,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        xc, zc, a00, a01, a10, a11, _pc = _canonicalize_block_2q(
            xc, zc, a00, a01, a10, a11, 0
        )
        xt, zt, a00, a01, a10, a11, _pt = _canonicalize_block_2q(
            xt, zt, a00, a01, a10, a11, 1
        )
        return (
            xc,
            zc,
            xt,
            zt,
            a00,
            a01,
            a10,
            a11,
            state,
            unv,
            rounds,
        )


    @nb.njit(cache=True)
    def _apply_pauli_3q_inplace(psi, qubit, pcode):
        p = int(pcode)
        if p == 0:
            return
        q = int(qubit)
        mask = 4 if q == 0 else (2 if q == 1 else 1)
        for i in range(8):
            if (i & mask) == 0:
                j = i | mask
                v0 = psi[i]
                v1 = psi[j]
                if p == 1:
                    psi[i] = v1
                    psi[j] = v0
                elif p == 2:
                    psi[i] = -1j * v1
                    psi[j] = 1j * v0
                else:
                    psi[i] = v0
                    psi[j] = -v1


    @nb.njit(cache=True)
    def _one_t_cycle_on_data_qubit(
        data_qubit,
        xc,
        zc,
        xt,
        zt,
        a00,
        a01,
        a10,
        a11,
        state,
        noise_arr,
        magic_arr,
        correction_offset,
        measurement_basis_noisy,
        fallback_on_unverified,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        """Apply one local T-injection channel to an entangled two-qubit state."""
        p1, p2, _pm, _pr, pi, pf = noise_arr
        eps = magic_arr[0]
        q = int(data_qubit)
        unverified = 0
        ec_cycles = 0
        ec_rounds = 0

        inv = 1.0 / math.sqrt(2.0)
        r0 = complex(inv, 0.0)
        phase = np.pi / 4.0 + eps
        r1 = inv * complex(math.cos(phase), math.sin(phase))
        state, magic_p = _sample_magic_pauli(state, magic_arr)
        r0, r1 = _apply_pauli_1q_state(r0, r1, magic_p)

        psi = np.empty(8, dtype=np.complex128)
        data = np.empty(4, dtype=np.complex128)
        data[0] = a00
        data[1] = a01
        data[2] = a10
        data[3] = a11
        for d in range(4):
            psi[2 * d] = data[d] * r0
            psi[2 * d + 1] = data[d] * r1

        # Ideal CNOT from the selected data qubit to the resource ancilla.
        if q == 0:
            tmp = psi[4]
            psi[4] = psi[5]
            psi[5] = tmp
            tmp = psi[6]
            psi[6] = psi[7]
            psi[7] = tmp
            xq = xc
            zq = zc
        else:
            tmp = psi[2]
            psi[2] = psi[3]
            psi[3] = tmp
            tmp = psi[6]
            psi[6] = psi[7]
            psi[7] = tmp
            xq = xt
            zq = zt

        xa = np.uint16(0)
        za = np.uint16(0)
        xq, zq, xa, za, state = _apply_transversal_cnot_faults(
            xq, zq, xa, za, state, p2
        )

        xq, zq, state, u, r = _run_one_ftec(
            xq,
            zq,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xa, za, state, u, r = _run_one_ftec(
            xa,
            za,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r

        xq, zq, pd = _canonical_logical_code(xq, zq)
        xa, za, pa = _canonical_logical_code(xa, za)
        _apply_pauli_3q_inplace(psi, q, pd)
        _apply_pauli_3q_inplace(psi, 2, pa)

        p_m0 = 0.0
        for i in range(0, 8, 2):
            v = psi[i]
            p_m0 += v.real * v.real + v.imag * v.imag
        p_m0 = min(max(p_m0, 0.0), 1.0)
        state, u01 = _rng_uniform01(state)
        ideal_m = 0 if u01 < p_m0 else 1
        norm = math.sqrt(max(p_m0 if ideal_m == 0 else 1.0 - p_m0, 1e-300))
        a00 = psi[ideal_m] / norm
        a01 = psi[2 + ideal_m] / norm
        a10 = psi[4 + ideal_m] / norm
        a11 = psi[6 + ideal_m] / norm

        state, meas_flip = _logical_measurement_flip_numba(
            xa,
            za,
            state,
            MEAS_Z,
            p1,
            pf,
            measurement_basis_noisy,
        )
        observed_m = ideal_m ^ int(meas_flip)

        xq, zq, state = _apply_idle_faults(xq, zq, state, pi)
        xq, zq, state, u, r = _run_one_ftec(
            xq,
            zq,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xq, zq, pl = _canonical_logical_code(xq, zq)
        a00, a01, a10, a11 = _apply_pauli_2q_state(
            a00, a01, a10, a11, q, pl
        )

        correction_exponent = (int(observed_m) + int(correction_offset)) & 3
        if correction_exponent == 0:
            xq, zq, state = _apply_idle_faults(xq, zq, state, pi)
        else:
            a00, a01, a10, a11 = _apply_s_power_2q_state(
                a00, a01, a10, a11, q, correction_exponent
            )
            if correction_exponent & 1:
                zq = np.uint16(zq ^ xq)
            xq, zq, state = _apply_1q_gate_noise_all_data_numba(
                xq, zq, state, p1
            )

        xq, zq, state, u, r = _run_one_ftec(
            xq,
            zq,
            state,
            noise_arr,
            fallback_on_unverified,
            op_type,
            q1,
            q2,
            round_id,
            basis_id,
            check_id,
            kind_id,
            min_rounds,
            max_rounds,
            key_codes,
            rx_masks,
            rz_masks,
        )
        unverified += u
        ec_cycles += 1
        ec_rounds += r
        xq, zq, pl = _canonical_logical_code(xq, zq)
        a00, a01, a10, a11 = _apply_pauli_2q_state(
            a00, a01, a10, a11, q, pl
        )

        if q == 0:
            xc = xq
            zc = zq
        else:
            xt = xq
            zt = zq

        return (
            xc,
            zc,
            xt,
            zt,
            a00,
            a01,
            a10,
            a11,
            state,
            unverified,
            ec_cycles,
            ec_rounds,
            observed_m,
            correction_exponent,
        )


    @nb.njit(cache=True)
    def _absorb_final_t_frames_2q(
        xc,
        zc,
        xt,
        zt,
        a00,
        a01,
        a10,
        a11,
        k_c,
        p_c,
        k_t,
        p_t,
    ):
        a00, a01, a10, a11, xc, zc = _apply_s_power_2q_with_frame(
            a00, a01, a10, a11, xc, zc, 0, -int(k_c)
        )
        a00, a01, a10, a11 = _apply_pauli_2q_state(
            a00, a01, a10, a11, 0, int(p_c)
        )
        a00, a01, a10, a11, xt, zt = _apply_s_power_2q_with_frame(
            a00, a01, a10, a11, xt, zt, 1, -int(k_t)
        )
        a00, a01, a10, a11 = _apply_pauli_2q_state(
            a00, a01, a10, a11, 1, int(p_t)
        )
        return xc, zc, xt, zt, a00, a01, a10, a11


    @nb.njit(cache=True)
    def _measure_computational_2q(
        xc, zc, xt, zt, a00, a01, a10, a11, state, noise_arr, measurement_basis_noisy
    ):
        probs = np.empty(4, dtype=np.float64)
        vals = np.empty(4, dtype=np.complex128)
        vals[0] = a00
        vals[1] = a01
        vals[2] = a10
        vals[3] = a11
        total = 0.0
        for i in range(4):
            v = vals[i]
            probs[i] = v.real * v.real + v.imag * v.imag
            total += probs[i]
        if total <= 0.0:
            probs[0] = 1.0
            probs[1] = 0.0
            probs[2] = 0.0
            probs[3] = 0.0
        else:
            for i in range(4):
                probs[i] /= total
        cumulative = np.empty(4, dtype=np.float64)
        running = 0.0
        for i in range(4):
            running += probs[i]
            cumulative[i] = running
        cumulative[3] = 1.0
        state, u = _rng_uniform01(state)
        outcome = 3
        for i in range(4):
            if u <= cumulative[i]:
                outcome = i
                break
        bit_c = (outcome >> 1) & 1
        bit_t = outcome & 1
        p1, _p2, _pm, _pr, _pi, pf = noise_arr
        state, flip_c = _logical_measurement_flip_numba(
            xc,
            zc,
            state,
            MEAS_Z,
            p1,
            pf,
            measurement_basis_noisy,
        )
        state, flip_t = _logical_measurement_flip_numba(
            xt,
            zt,
            state,
            MEAS_Z,
            p1,
            pf,
            measurement_basis_noisy,
        )
        bit_c ^= int(flip_c) & 1
        bit_t ^= int(flip_t) & 1
        return state, 2 * bit_c + bit_t


    @nb.njit(cache=True)
    def _sample_inverse(state, cumulative, signs, gamma):
        state, u = _rng_uniform01(state)
        index = cumulative.shape[0] - 1
        for j in range(cumulative.shape[0]):
            if u <= cumulative[j]:
                index = j
                break
        return state, index, gamma * signs[index]


    @nb.njit(cache=True)
    def _simulate_universal_sequence_kernel(
        c1_codes,
        c2_codes,
        shots,
        seed,
        noise_arr,
        magic_arr,
        use_t_twirl,
        use_cx_twirl,
        use_pec,
        measurement_basis_noisy,
        fallback_on_unverified,
        cliff_cumulative,
        cliff_signs,
        cliff_gamma,
        cx_cumulative,
        cx_signs,
        cx_gamma,
        t_cumulative,
        t_signs,
        t_gamma,
        op_type,
        q1,
        q2,
        round_id,
        basis_id,
        check_id,
        kind_id,
        min_rounds,
        max_rounds,
        key_codes,
        rx_masks,
        rz_masks,
    ):
        sums = np.zeros(4, dtype=np.float64)
        sums2 = np.zeros(4, dtype=np.float64)
        abs_weight_total = 0.0
        unverified_total = 0
        ec_cycles_total = 0
        ec_rounds_total = 0
        depth = int(c1_codes.shape[0])
        base_state = np.uint64(seed) + np.uint64(0xD2B74407B1CE6E93)

        for shot in range(int(shots)):
            state = base_state + np.uint64(shot) * np.uint64(0x9E3779B97F4A7C15)
            state = _rng_next_u64(_rng_next_u64(state))

            xc = np.uint16(0)
            zc = np.uint16(0)
            xt = np.uint16(0)
            zt = np.uint16(0)
            a00 = complex(1.0, 0.0)
            a01 = complex(0.0, 0.0)
            a10 = complex(0.0, 0.0)
            a11 = complex(0.0, 0.0)

            incoming_t_pauli_c = 0
            incoming_t_pauli_t = 0
            incoming_t_k_c = 0
            incoming_t_k_t = 0
            weight = 1.0

            for layer in range(depth):
                twirl_c = 0
                twirl_t = 0
                if use_cx_twirl:
                    state, twirl_c = _rng_int(state, 2)
                    state, twirl_t = _rng_int(state, 2)

                # C1 contains the previous T output frame and the current
                # dressed-CX pre-twirl.  The dressed-CX inverse is not here;
                # it is sampled after the noisy CX and compiled into C2.
                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    u,
                    r,
                ) = _apply_compiled_local_layer_and_ec(
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    int(c1_codes[layer, 0]),
                    int(c1_codes[layer, 1]),
                    incoming_t_pauli_c,
                    incoming_t_pauli_t,
                    incoming_t_k_c,
                    incoming_t_k_t,
                    twirl_c,
                    twirl_t,
                    0,
                    0,
                    0,
                    0,
                    1,
                    noise_arr,
                    fallback_on_unverified,
                    op_type,
                    q1,
                    q2,
                    round_id,
                    basis_id,
                    check_id,
                    kind_id,
                    min_rounds,
                    max_rounds,
                    key_codes,
                    rx_masks,
                    rz_masks,
                )
                unverified_total += u
                ec_cycles_total += 2
                ec_rounds_total += r

                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    u,
                    r,
                ) = _apply_cx_and_ec_2q(
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    noise_arr,
                    fallback_on_unverified,
                    op_type,
                    q1,
                    q2,
                    round_id,
                    basis_id,
                    check_id,
                    kind_id,
                    min_rounds,
                    max_rounds,
                    key_codes,
                    rx_masks,
                    rz_masks,
                )
                unverified_total += u
                ec_cycles_total += 2
                ec_rounds_total += r

                # The dressed-CX inverse is defined at the output of the
                # complete twirled dressed-CX primitive.  Chronologically it
                # follows V_post and is then merged into the C2 Clifford slot.
                p_cx_c = 0
                p_cx_t = 0
                if use_pec:
                    state, sampled_cx_pauli, factor = _sample_inverse(
                        state, cx_cumulative, cx_signs, cx_gamma
                    )
                    weight *= factor
                    p_cx_c = int(sampled_cx_pauli) // 4
                    p_cx_t = int(sampled_cx_pauli) % 4

                p_c2_c = 0
                p_c2_t = 0
                if use_pec:
                    state, p_c2_c, factor = _sample_inverse(
                        state, cliff_cumulative, cliff_signs, cliff_gamma
                    )
                    weight *= factor
                    state, p_c2_t, factor = _sample_inverse(
                        state, cliff_cumulative, cliff_signs, cliff_gamma
                    )
                    weight *= factor

                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    u,
                    r,
                ) = _apply_compiled_local_layer_and_ec(
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    int(c2_codes[layer, 0]),
                    int(c2_codes[layer, 1]),
                    0,
                    0,
                    0,
                    0,
                    twirl_c,
                    twirl_t,
                    p_cx_c,
                    p_cx_t,
                    p_c2_c,
                    p_c2_t,
                    2,
                    noise_arr,
                    fallback_on_unverified,
                    op_type,
                    q1,
                    q2,
                    round_id,
                    basis_id,
                    check_id,
                    kind_id,
                    min_rounds,
                    max_rounds,
                    key_codes,
                    rx_masks,
                    rz_masks,
                )
                unverified_total += u
                ec_cycles_total += 2
                ec_rounds_total += r

                k_c = 0
                k_t = 0
                if use_t_twirl:
                    state, k_c = _rng_int(state, 4)
                    state, k_t = _rng_int(state, 4)

                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    u,
                    e,
                    r,
                    _m,
                    _corr,
                ) = _one_t_cycle_on_data_qubit(
                    0,
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    noise_arr,
                    magic_arr,
                    k_c,
                    measurement_basis_noisy,
                    fallback_on_unverified,
                    op_type,
                    q1,
                    q2,
                    round_id,
                    basis_id,
                    check_id,
                    kind_id,
                    min_rounds,
                    max_rounds,
                    key_codes,
                    rx_masks,
                    rz_masks,
                )
                unverified_total += u
                ec_cycles_total += e
                ec_rounds_total += r

                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    u,
                    e,
                    r,
                    _m,
                    _corr,
                ) = _one_t_cycle_on_data_qubit(
                    1,
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    state,
                    noise_arr,
                    magic_arr,
                    k_t,
                    measurement_basis_noisy,
                    fallback_on_unverified,
                    op_type,
                    q1,
                    q2,
                    round_id,
                    basis_id,
                    check_id,
                    kind_id,
                    min_rounds,
                    max_rounds,
                    key_codes,
                    rx_masks,
                    rz_masks,
                )
                unverified_total += u
                ec_cycles_total += e
                ec_rounds_total += r

                incoming_t_pauli_c = 0
                incoming_t_pauli_t = 0
                if use_pec:
                    state, incoming_t_pauli_c, factor = _sample_inverse(
                        state, t_cumulative, t_signs, t_gamma
                    )
                    weight *= factor
                    state, incoming_t_pauli_t, factor = _sample_inverse(
                        state, t_cumulative, t_signs, t_gamma
                    )
                    weight *= factor
                incoming_t_k_c = k_c
                incoming_t_k_t = k_t

            if depth > 0:
                (
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                ) = _absorb_final_t_frames_2q(
                    xc,
                    zc,
                    xt,
                    zt,
                    a00,
                    a01,
                    a10,
                    a11,
                    incoming_t_k_c,
                    incoming_t_pauli_c,
                    incoming_t_k_t,
                    incoming_t_pauli_t,
                )

            state, outcome = _measure_computational_2q(
                xc, zc, xt, zt, a00, a01, a10, a11, state, noise_arr, measurement_basis_noisy
            )
            sums[outcome] += weight
            sums2[outcome] += weight * weight
            abs_weight_total += abs(weight)

        means = np.empty(4, dtype=np.float64)
        stderrs = np.empty(4, dtype=np.float64)
        n = int(shots)
        for i in range(4):
            if n <= 0:
                means[i] = np.nan
                stderrs[i] = np.nan
            else:
                means[i] = sums[i] / n
                if n > 1:
                    var = max((sums2[i] - n * means[i] * means[i]) / (n - 1), 0.0)
                    stderrs[i] = math.sqrt(var / n)
                else:
                    stderrs[i] = 0.0
        mean_abs_weight = abs_weight_total / max(n, 1)
        unverified_rate = float(unverified_total) / max(ec_cycles_total, 1)
        average_ec_rounds = float(ec_rounds_total) / max(ec_cycles_total, 1)
        return means, stderrs, mean_abs_weight, unverified_rate, average_ec_rounds


def _apply_ideal_local_python(state: np.ndarray, qubit: int, gate_code: int) -> np.ndarray:
    state = np.asarray(state, dtype=np.complex128).copy()
    out = np.zeros(4, dtype=np.complex128)
    inv = 1.0 / math.sqrt(2.0)
    for other in range(2):
        if qubit == 0:
            i0, i1 = other, 2 + other
        else:
            i0, i1 = 2 * other, 2 * other + 1
        a0, a1 = state[i0], state[i1]
        if int(gate_code) == 0:
            out[i0] = inv * (a0 - 1j * a1)
            out[i1] = inv * (-1j * a0 + a1)
        elif int(gate_code) == 1:
            out[i0] = inv * (a0 - a1)
            out[i1] = inv * (a0 + a1)
        else:
            out[i0] = a0
            out[i1] = 1j * a1
    return out


def ideal_universal_distribution(
    c1_codes: np.ndarray,
    c2_codes: np.ndarray,
) -> np.ndarray:
    """Ideal computational-basis distribution for the fixed macro circuit."""
    c1 = np.asarray(c1_codes, dtype=np.int64)
    c2 = np.asarray(c2_codes, dtype=np.int64)
    if c1.shape != c2.shape or c1.ndim != 2 or c1.shape[1] != 2:
        raise ValueError("c1_codes and c2_codes must both have shape (depth,2).")
    state = np.asarray([1.0 + 0.0j, 0.0j, 0.0j, 0.0j], dtype=np.complex128)
    t_phase = complex(math.cos(math.pi / 4.0), math.sin(math.pi / 4.0))
    for layer in range(c1.shape[0]):
        state = _apply_ideal_local_python(state, 0, int(c1[layer, 0]))
        state = _apply_ideal_local_python(state, 1, int(c1[layer, 1]))
        state = state[[0, 1, 3, 2]]
        state = _apply_ideal_local_python(state, 0, int(c2[layer, 0]))
        state = _apply_ideal_local_python(state, 1, int(c2[layer, 1]))
        state[1] *= t_phase
        state[2] *= t_phase
        state[3] *= t_phase * t_phase
    prob = np.abs(state) ** 2
    return prob / np.sum(prob)


def generate_universal_sequences(
    depths: Sequence[int],
    sequences_per_depth: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    records: List[Dict[str, Any]] = []
    for depth in map(int, depths):
        if depth < 0:
            raise ValueError("Circuit depths must be nonnegative.")
        for sequence_id in range(int(sequences_per_depth)):
            c1 = rng.integers(0, 3, size=(depth, 2), dtype=np.int8)
            c2 = rng.integers(0, 3, size=(depth, 2), dtype=np.int8)
            ideal = ideal_universal_distribution(c1, c2)
            records.append(
                {
                    "depth": depth,
                    "sequence_id": sequence_id,
                    "c1_codes": c1,
                    "c2_codes": c2,
                    "ideal_distribution": ideal,
                    "c1_control": ",".join(LOCAL_GATE_LABELS[int(v)] for v in c1[:, 0]),
                    "c1_target": ",".join(LOCAL_GATE_LABELS[int(v)] for v in c1[:, 1]),
                    "c2_control": ",".join(LOCAL_GATE_LABELS[int(v)] for v in c2[:, 0]),
                    "c2_target": ",".join(LOCAL_GATE_LABELS[int(v)] for v in c2[:, 1]),
                }
            )
    return records


def simulate_universal_sequence(
    decoder,
    model: UniversalNoiseModel,
    options: Mapping[str, bool],
    c1_codes: np.ndarray,
    c2_codes: np.ndarray,
    shots: int,
    seed: int,
    clifford_inverse_q: Optional[Mapping[str, float]] = None,
    dressed_cx_inverse_q: Optional[Mapping[str, float]] = None,
    t_inverse_q: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    _universal_require_numba()
    model = model.validated()
    use_pec = any(x is not None for x in (clifford_inverse_q, dressed_cx_inverse_q, t_inverse_q))
    if use_pec and any(x is None for x in (clifford_inverse_q, dressed_cx_inverse_q, t_inverse_q)):
        raise ValueError("Complete universal PEC requires Clifford, dressed-CX, and T inverse models.")
    if use_pec:
        c_cum, c_sign, c_gamma = _inverse_1q_arrays(clifford_inverse_q)
        cx_cum, cx_sign, cx_gamma = _inverse_2q_arrays(dressed_cx_inverse_q)
        t_cum, t_sign, t_gamma = _inverse_1q_arrays(t_inverse_q)
    else:
        c_cum=np.ones(4); c_sign=np.ones(4); c_gamma=1.0
        cx_cum=np.ones(16); cx_sign=np.ones(16); cx_gamma=1.0
        t_cum=np.ones(4); t_sign=np.ones(4); t_gamma=1.0
    data = _universal_compiled(decoder)
    means, stderrs, mean_abs_weight, unv, avg_rounds = _simulate_universal_sequence_kernel(
        np.asarray(c1_codes,dtype=np.int8), np.asarray(c2_codes,dtype=np.int8), int(shots), int(seed),
        _universal_noise_array(model.physical_noise), _universal_magic_array(model),
        bool(model.t_phase_twirl), bool(model.cx_phase_twirl), bool(use_pec),
        bool(options["measurement_basis_gates_noisy"]), bool(options["fallback_on_unverified"]),
        c_cum,c_sign,float(c_gamma), cx_cum,cx_sign,float(cx_gamma), t_cum,t_sign,float(t_gamma),
        data["op_type"],data["q1"],data["q2"],data["round_id"],data["basis_id"],data["check_id"],data["kind_id"],
        int(data["min_rounds"]),int(data["max_rounds"]),data["key_codes"],data["rx_masks"],data["rz_masks"],
    )
    return {"probabilities":np.asarray(means,dtype=float),"stderrs":np.asarray(stderrs,dtype=float),
            "mean_abs_weight":float(mean_abs_weight),"unverified_rate":float(unv),
            "average_ec_rounds":float(avg_rounds),"gamma_per_macro_layer":float(cx_gamma*c_gamma**2*t_gamma**2)}


def warmup_universal_numba(
    decoder,
    model: UniversalNoiseModel,
    options: Mapping[str, bool],
    seed: int,
) -> Dict[str, Any]:
    """Compile the universal-circuit path using a notebook-supplied warmup seed."""
    _universal_require_numba()
    return simulate_universal_sequence(
        decoder, model, options,
        np.asarray([[0, 1]], dtype=np.int8),
        np.asarray([[2, 0]], dtype=np.int8),
        2, int(seed), None, None, None,
    )








def run_universal_pec_validation(
    decoder,
    model: UniversalNoiseModel,
    options: Mapping[str, bool],
    clifford_inverse_q: Mapping[str, float],
    dressed_cx_inverse_q: Mapping[str, float],
    t_inverse_q: Mapping[str, float],
    depths: Sequence[int],
    sequences_per_depth: int,
    noisy_shots: int,
    pec_shots: int,
    circuit_seed: int,
    sampling_seed: int,
    n_jobs: int,
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> Dict[str, pd.DataFrame]:
    records = generate_universal_sequences(
        depths=depths,
        sequences_per_depth=sequences_per_depth,
        seed=circuit_seed,
    )
    rng = np.random.default_rng(int(sampling_seed))
    tasks: List[Dict[str, Any]] = []
    for rec in records:
        tasks.append({**rec, "mode": "noisy", "shots": int(noisy_shots), "seed": int(rng.integers(2**32 - 1))})
        tasks.append({**rec, "mode": "pec", "shots": int(pec_shots), "seed": int(rng.integers(2**32 - 1))})

    if verbose:
        print("Universal PEC validation")
        print("  macro layer          = (C1 + V_pre) -> CX -> (V_post + P_DCX + C2 + P_C2) -> T x T")
        print("  depths               =", list(map(int, depths)))
        print("  sequences per depth  =", int(sequences_per_depth))
        print("  noisy shots/sequence =", int(noisy_shots))
        print("  PEC shots/sequence   =", int(pec_shots))
        print("  save data            = False")

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    def run_one(task: Mapping[str, Any]) -> Dict[str, Any]:
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        if task["mode"] == "pec":
            result = simulate_universal_sequence(
                dec,
                model,
                options,
                task["c1_codes"],
                task["c2_codes"],
                shots=int(task["shots"]),
                seed=int(task["seed"]),
                clifford_inverse_q=clifford_inverse_q,
                dressed_cx_inverse_q=dressed_cx_inverse_q,
                t_inverse_q=t_inverse_q,
            )
        else:
            result = simulate_universal_sequence(
                dec,
                model,
                options,
                task["c1_codes"],
                task["c2_codes"],
                shots=int(task["shots"]),
                seed=int(task["seed"]),
                clifford_inverse_q=None,
                dressed_cx_inverse_q=None,
                t_inverse_q=None,
            )
        row: Dict[str, Any] = {
            "depth": int(task["depth"]),
            "sequence_id": int(task["sequence_id"]),
            "mode": str(task["mode"]),
            "mean_abs_weight": result["mean_abs_weight"],
            "gamma_per_macro_layer": result["gamma_per_macro_layer"],
            "unverified_rate": result["unverified_rate"],
            "average_ec_rounds": result["average_ec_rounds"],
        }
        for i, label in enumerate(BITSTRING_LABELS):
            row[f"p_{label}"] = float(result["probabilities"][i])
            row[f"se_{label}"] = float(result["stderrs"][i])
        return row

    if JOBLIB_AVAILABLE and int(n_jobs) != 1 and len(tasks) > 1:
        rows = Parallel(n_jobs=int(n_jobs), backend="loky", batch_size=1)(
            delayed(run_one)(task) for task in tasks
        )
    else:
        rows = [run_one(task) for task in tasks]

    mode_df = pd.DataFrame(rows)
    metadata_rows: List[Dict[str, Any]] = []
    for rec in records:
        row = {
            "depth": int(rec["depth"]),
            "sequence_id": int(rec["sequence_id"]),
            "c1_control": rec["c1_control"],
            "c1_target": rec["c1_target"],
            "c2_control": rec["c2_control"],
            "c2_target": rec["c2_target"],
        }
        for i, label in enumerate(BITSTRING_LABELS):
            row[f"ideal_p_{label}"] = float(rec["ideal_distribution"][i])
        metadata_rows.append(row)
    sequence_df = pd.DataFrame(metadata_rows)

    for mode in ("noisy", "pec"):
        sub = mode_df[mode_df["mode"] == mode].drop(columns="mode")
        rename = {
            column: f"{mode}_{column}"
            for column in sub.columns
            if column not in ("depth", "sequence_id")
        }
        sequence_df = sequence_df.merge(
            sub.rename(columns=rename),
            on=["depth", "sequence_id"],
            how="left",
            validate="one_to_one",
        )

    for mode in ("noisy", "pec"):
        absolute_columns = []
        for label in BITSTRING_LABELS:
            error_column = f"{mode}_abs_error_{label}"
            sequence_df[error_column] = np.abs(
                sequence_df[f"{mode}_p_{label}"] - sequence_df[f"ideal_p_{label}"]
            )
            absolute_columns.append(error_column)
        sequence_df[f"{mode}_tv_distance"] = 0.5 * sequence_df[absolute_columns].sum(axis=1)
        sequence_df[f"{mode}_mean_abs_probability_error"] = sequence_df[absolute_columns].mean(axis=1)
        sequence_df[f"{mode}_probability_sum"] = sequence_df[
            [f"{mode}_p_{label}" for label in BITSTRING_LABELS]
        ].sum(axis=1)

    summary_df = (
        sequence_df.groupby("depth", as_index=False)
        .agg(
            n_sequences=("sequence_id", "nunique"),
            noisy_tv_mean=("noisy_tv_distance", "mean"),
            noisy_tv_std=("noisy_tv_distance", "std"),
            pec_tv_mean=("pec_tv_distance", "mean"),
            pec_tv_std=("pec_tv_distance", "std"),
            noisy_map_error_mean=("noisy_mean_abs_probability_error", "mean"),
            noisy_map_error_std=("noisy_mean_abs_probability_error", "std"),
            pec_map_error_mean=("pec_mean_abs_probability_error", "mean"),
            pec_map_error_std=("pec_mean_abs_probability_error", "std"),
            noisy_probability_sum_mean=("noisy_probability_sum", "mean"),
            pec_probability_sum_mean=("pec_probability_sum", "mean"),
            pec_mean_abs_weight=("pec_mean_abs_weight", "mean"),
            pec_gamma_per_macro_layer=("pec_gamma_per_macro_layer", "mean"),
        )
        .sort_values("depth")
        .reset_index(drop=True)
    )
    for column in summary_df.columns:
        if column.endswith("_std"):
            summary_df[column] = summary_df[column].fillna(0.0)

    if verbose:
        print(summary_df.to_string(index=False))
    return {"sequence_df": sequence_df, "summary_df": summary_df}












# =============================================================================
# Coherent physical-noise state-vector simulator
# =============================================================================

@dataclass(frozen=True)
class CoherentNoiseParams:
    """Coherent physical Z-rotation errors for circuit locations."""

    epsilon_1q: float
    epsilon_2q: float
    epsilon_idle: float


def make_homogeneous_coherent_noise(
    strength: float,
    two_qubit_factor: float,
    idle_factor: float,
) -> CoherentNoiseParams:
    strength = float(strength)
    return CoherentNoiseParams(
        epsilon_1q=strength,
        epsilon_2q=float(two_qubit_factor) * strength,
        epsilon_idle=float(idle_factor) * strength,
    )


_coh_i2 = np.eye(2, dtype=np.complex128)
_coh_x2 = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_coh_y2 = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_coh_z2 = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_coh_h2 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / math.sqrt(2.0)
_coh_s2 = np.array([[1, 0], [0, 1j]], dtype=np.complex128)
_coh_sdg2 = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
_coh_data_qubits = tuple(range(7))
_coh_syndrome_qubit = 7
_coh_flag_qubit = 8
_coh_num_qubits = 9
_coh_logical_mask = (1 << 7) - 1


def _coh_rx(theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _coh_i2 - 1j * math.sin(theta / 2) * _coh_x2


def _coh_ry(theta: float) -> np.ndarray:
    return math.cos(theta / 2) * _coh_i2 - 1j * math.sin(theta / 2) * _coh_y2


def _coh_rz(theta: float) -> np.ndarray:
    return np.array(
        [[np.exp(-0.5j * theta), 0], [0, np.exp(+0.5j * theta)]],
        dtype=np.complex128,
    )


def _coh_normalize(state: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("State-vector norm is zero or non-finite.")
    return state / norm


def _coh_steane_rowspace_masks() -> List[int]:
    return [bits_to_int(bits) for bits in SteaneCode().rowspace]


_coh_steane_c2_masks = _coh_steane_rowspace_masks()


def encoded_logical_state(label: str) -> np.ndarray:
    """Return an encoded Steane logical state with two ancillas in |0>."""
    zero = np.zeros(2 ** _coh_num_qubits, dtype=np.complex128)
    one = np.zeros_like(zero)
    amp = 1.0 / math.sqrt(len(_coh_steane_c2_masks))
    for mask in _coh_steane_c2_masks:
        zero[mask] = amp
        one[mask ^ _coh_logical_mask] = amp

    if label == "0":
        state = zero
    elif label == "1":
        state = one
    elif label == "+":
        state = (zero + one) / math.sqrt(2.0)
    elif label == "-":
        state = (zero - one) / math.sqrt(2.0)
    elif label == "+i":
        state = (zero + 1j * one) / math.sqrt(2.0)
    elif label == "-i":
        state = (zero - 1j * one) / math.sqrt(2.0)
    else:
        raise ValueError(f"Unknown logical state {label!r}")
    return _coh_normalize(state)


class CoherentSteaneFTSimulator:
    """Nine-qubit state-vector trajectory simulator for coherent physical noise."""

    def __init__(
        self,
        decoder: VerifiedSteaneFlagDecoder,
        noise: CoherentNoiseParams,
        options: Mapping[str, bool],
        seed: Optional[int] = None,
    ):
        self.decoder = decoder
        self.model = decoder.model
        self.code = SteaneCode()
        self.noise = noise
        self.options = validate_circuit_options(options)
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(2 ** _coh_num_qubits, dtype=np.int64)
        self.state = encoded_logical_state("0")
        self.unverified_count = 0
        self.ec_cycles = 0
        self.ec_rounds = 0

    def _apply_1q_matrix(self, unitary: np.ndarray, q: int) -> None:
        bit = 1 << int(q)
        idx0 = self.indices[(self.indices & bit) == 0]
        idx1 = idx0 | bit
        a = self.state[idx0].copy()
        b = self.state[idx1].copy()
        self.state[idx0] = unitary[0, 0] * a + unitary[0, 1] * b
        self.state[idx1] = unitary[1, 0] * a + unitary[1, 1] * b

    def _apply_physical_1q_gate(
        self, unitary: np.ndarray, q: int, noisy: bool
    ) -> None:
        self._apply_1q_matrix(unitary, q)
        if noisy and abs(self.noise.epsilon_1q) > 0:
            self._apply_1q_matrix(_coh_rz(self.noise.epsilon_1q), q)

    def _apply_idle(self, q: int) -> None:
        if abs(self.noise.epsilon_idle) > 0:
            self._apply_1q_matrix(_coh_rz(self.noise.epsilon_idle), q)

    def _apply_ideal_cx(self, control: int, target: int) -> None:
        cb = 1 << int(control)
        tb = 1 << int(target)
        idx0 = self.indices[((self.indices & cb) != 0) & ((self.indices & tb) == 0)]
        idx1 = idx0 | tb
        tmp = self.state[idx0].copy()
        self.state[idx0] = self.state[idx1]
        self.state[idx1] = tmp

    def _apply_physical_cx(self, control: int, target: int, noisy: bool) -> None:
        self._apply_ideal_cx(control, target)
        if noisy and abs(self.noise.epsilon_2q) > 0:
            error = _coh_rz(self.noise.epsilon_2q)
            self._apply_1q_matrix(error, control)
            self._apply_1q_matrix(error, target)

    def _measure_z(self, q: int) -> int:
        bit = 1 << int(q)
        mask1 = (self.indices & bit) != 0
        p1 = float(np.sum(np.abs(self.state[mask1]) ** 2))
        p1 = min(max(p1, 0.0), 1.0)
        outcome = int(self.rng.random() < p1)
        keep = mask1 if outcome else ~mask1
        self.state[~keep] = 0.0
        self.state = _coh_normalize(self.state)
        return outcome

    def _reset_z(self, q: int) -> None:
        if self._measure_z(q):
            self._apply_1q_matrix(_coh_x2, q)

    def _reset_x(self, q: int) -> None:
        self._reset_z(q)
        self._apply_physical_1q_gate(_coh_h2, q, noisy=True)

    def _measure_x(self, q: int) -> int:
        self._apply_physical_1q_gate(_coh_h2, q, noisy=True)
        return self._measure_z(q)

    @staticmethod
    def _logical_gate_matrix(gate: str) -> np.ndarray:
        # These physical transversal representatives implement the named
        # logical operations for the Steane code conventions used here.
        if gate == "RZ90":
            return _coh_sdg2
        if gate == "RX90":
            return _coh_rx(-np.pi / 2)
        if gate == "RY90":
            return _coh_ry(np.pi / 2)
        if gate == "H":
            return _coh_h2
        if gate == "X":
            return _coh_x2
        if gate == "Z":
            return _coh_z2
        if gate == "Sdg":
            return _coh_s2
        raise ValueError(f"Unknown logical gate {gate!r}")

    def apply_logical_pauli(self, pauli: str) -> None:
        if pauli == "I":
            return
        if pauli not in ("X", "Y", "Z"):
            raise ValueError(pauli)
        for q in _coh_data_qubits:
            if pauli in ("X", "Y"):
                self._apply_1q_matrix(_coh_x2, q)
            if pauli in ("Z", "Y"):
                self._apply_1q_matrix(_coh_z2, q)

    def apply_logical_gate(
        self,
        gate: str,
        noisy: bool,
        apply_qec: bool,
        post_ec_logical_pauli: str = "I",
    ) -> None:
        unitary = self._logical_gate_matrix(gate)
        for q in _coh_data_qubits:
            self._apply_physical_1q_gate(unitary, q, noisy=noisy)
        if apply_qec:
            self.run_ft_ec(extra_logical_pauli=post_ec_logical_pauli)
        else:
            self.apply_logical_pauli(post_ec_logical_pauli)

    def prepare_initial(self, label: str) -> None:
        """Prepare a CSB input from an ideal encoded |0_L> reference state."""
        self.state = encoded_logical_state("0")
        self.unverified_count = 0
        self.ec_cycles = 0
        self.ec_rounds = 0
        prep_gates = {
            "0": [],
            "1": ["X"],
            "+": ["H"],
            "-": ["H", "Z"],
            "+i": ["H", "RZ90"],
            "-i": ["H", "Sdg"],
        }[label]
        for gate in prep_gates:
            self.apply_logical_gate(
                gate,
                noisy=bool(self.options["prep_gates_noisy"]),
                apply_qec=bool(self.options["prep_qec_after_gate"]),
            )

    def run_ft_ec(self, extra_logical_pauli: str = "I") -> Dict[str, Any]:
        x_syn = [0] * self.model.max_rounds
        z_syn = [0] * self.model.max_rounds
        x_flags = [0] * self.model.max_rounds
        z_flags = [0] * self.model.max_rounds
        completed = 0

        for op in self.model.ops:
            if op.name == "DATA_IDLE":
                for q in _coh_data_qubits:
                    self._apply_idle(q)
            elif op.name == "RESET_X":
                self._reset_x(int(op.args[0]))
            elif op.name == "RESET_Z":
                self._reset_z(int(op.args[0]))
            elif op.name == "CX":
                c, t = map(int, op.args)
                self._apply_physical_cx(c, t, noisy=True)
            elif op.name == "MEAS_Z":
                q = int(op.args[0])
                outcome = self._measure_z(q)
                r, basis, check, kind = op.meta
                if basis == "Z_CHECK" and kind == "syndrome":
                    z_syn[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag":
                    z_flags[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag":
                    x_flags[int(r)] ^= outcome << int(check)
            elif op.name == "MEAS_X":
                q = int(op.args[0])
                outcome = self._measure_x(q)
                r, basis, check, kind = op.meta
                if basis == "X_CHECK" and kind == "syndrome":
                    x_syn[int(r)] ^= outcome << int(check)
                elif basis == "X_CHECK" and kind == "flag":
                    x_flags[int(r)] ^= outcome << int(check)
                elif basis == "Z_CHECK" and kind == "flag":
                    z_flags[int(r)] ^= outcome << int(check)
                if basis == "Z_CHECK" and int(check) == 2 and kind == "flag":
                    completed = int(r) + 1
                    if self.model.terminal_reliable(
                        x_syn, z_syn, x_flags, z_flags, completed
                    ):
                        break
            else:
                raise ValueError(op.name)

        if completed == 0:
            completed = self.model.max_rounds
        key = (
            tuple(x_syn[:completed]), tuple(z_syn[:completed]),
            tuple(x_flags[:completed]), tuple(z_flags[:completed]),
        )

        unverified = False
        try:
            rx_bits, rz_bits = self.decoder.decode(key)
        except KeyError:
            if not self.options["fallback_on_unverified"]:
                raise
            unverified = True
            rx_bits = np.zeros(7, dtype=np.uint8)
            rz_bits = np.zeros(7, dtype=np.uint8)
            for q in self.code.ordinary_decode(z_syn[completed - 1]):
                rx_bits[q] = 1
            for q in self.code.ordinary_decode(x_syn[completed - 1]):
                rz_bits[q] = 1

        # Recovery and PEC Paulis are ideal frame updates, not new noisy gates.
        for q in _coh_data_qubits:
            if int(rx_bits[q]):
                self._apply_1q_matrix(_coh_x2, q)
            if int(rz_bits[q]):
                self._apply_1q_matrix(_coh_z2, q)
        self.apply_logical_pauli(extra_logical_pauli)

        self._reset_z(_coh_syndrome_qubit)
        self._reset_z(_coh_flag_qubit)
        self.ec_cycles += 1
        self.ec_rounds += completed
        self.unverified_count += int(unverified)
        return {"rounds": completed, "unverified": unverified,
                "extra_logical_pauli": extra_logical_pauli}

    def measure_logical_pauli(self, pauli: str) -> int:
        noisy_basis = bool(self.options["measurement_basis_gates_noisy"])
        if pauli == "Z":
            pass
        elif pauli == "X":
            for q in _coh_data_qubits:
                self._apply_physical_1q_gate(_coh_h2, q, noisy=noisy_basis)
        elif pauli == "Y":
            for q in _coh_data_qubits:
                self._apply_physical_1q_gate(_coh_s2, q, noisy=noisy_basis)
                self._apply_physical_1q_gate(_coh_h2, q, noisy=noisy_basis)
        else:
            raise ValueError(pauli)

        bits = np.zeros(7, dtype=np.uint8)
        for q in _coh_data_qubits:
            bits[q] = self._measure_z(q)
        syndrome = self.code.syndrome(bits)
        if syndrome != 0:
            bits[self.code.qubit_from_syndrome[syndrome]] ^= 1
        return +1 if int(np.sum(bits) % 2) == 0 else -1


def simulate_coherent_csb_point(
    decoder: VerifiedSteaneFlagDecoder,
    noise: CoherentNoiseParams,
    options: Mapping[str, bool],
    target_gate: str,
    branch_name: str,
    depth: int,
    shots: int,
    seed: int,
) -> Tuple[float, float, float]:
    branch = get_csb_branches(target_gate)[branch_name]
    rng = np.random.default_rng(seed)
    outcomes = np.empty(int(shots), dtype=float)
    unverified = 0
    ec_cycles = 0

    for shot in range(int(shots)):
        sim = CoherentSteaneFTSimulator(
            decoder, noise, options, seed=int(rng.integers(2**32 - 1))
        )
        sim.prepare_initial(branch["initial"])
        for _ in range(int(depth)):
            sim.apply_logical_gate(
                target_gate,
                noisy=True,
                apply_qec=bool(options["qec_after_each_logical_gate"]),
            )
        outcomes[shot] = int(branch.get("measure_sign", +1)) * sim.measure_logical_pauli(
            branch["measure"]
        )
        unverified += sim.unverified_count
        ec_cycles += sim.ec_cycles

    mean = float(np.mean(outcomes)) if shots > 0 else np.nan
    stderr = float(np.std(outcomes, ddof=1) / math.sqrt(shots)) if shots > 1 else 0.0
    return mean, stderr, float(unverified) / max(ec_cycles, 1)


def run_coherent_csb_spectrum(
    decoder: VerifiedSteaneFlagDecoder,
    noise: CoherentNoiseParams,
    options: Mapping[str, bool],
    steps: Sequence[int],
    rep: int,
    shots: int,
    seed: int,
    n_jobs: int,
    chunk_shots: int,
    target_gates: Sequence[str],
    cache_file: str | Path,
    min_rounds: int,
    max_rounds: int,
    verbose: bool,
) -> pd.DataFrame:
    """Run state-vector CSB under coherent physical gate errors."""
    steps = [int(s) for s in steps]
    rep = int(rep)
    shots = int(shots)
    chunk_shots = max(1, int(chunk_shots))
    rng = np.random.default_rng(seed)
    tasks = []
    point_id = 0

    for gate in target_gates:
        for branch_name, branch in get_csb_branches(gate).items():
            branch_total = max(1, shots // 2) if branch["kind"].startswith("equatorial") else shots
            for step in steps:
                depth = rep * step
                for chunk_id, n_chunk in enumerate(_chunk_sizes(branch_total, chunk_shots)):
                    tasks.append({
                        "point_id": point_id,
                        "chunk_id": chunk_id,
                        "target_gate": gate,
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
                    })
                point_id += 1

    cache_path = str(cache_file)
    if not Path(cache_path).exists() and hasattr(decoder, "save"):
        decoder.save(cache_path)

    def run_task(task):
        dec = get_worker_decoder(cache_path, min_rounds, max_rounds)
        mean, stderr, unv = simulate_coherent_csb_point(
            dec, noise, options, task["target_gate"], task["branch"],
            task["depth"], task["shots"], task["seed"],
        )
        return {**task, "mean": mean, "stderr": stderr, "unverified_rate": unv}

    if JOBLIB_AVAILABLE and n_jobs != 1 and len(tasks) > 1:
        chunk_rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(run_task)(task) for task in tasks
        )
    else:
        chunk_rows = [run_task(task) for task in tasks]

    chunk_df = pd.DataFrame(chunk_rows)
    rows = [
        _combine_csb_chunk_rows(sub)
        for _, sub in chunk_df.groupby("point_id", sort=True)
    ]
    result = pd.DataFrame(rows).sort_values(
        ["target_gate", "branch", "step"]
    ).reset_index(drop=True)

    if verbose:
        print("Coherent state-vector CSB")
        print("  target gates =", list(target_gates))
        print("  true depths  =", [rep * s for s in steps])
        print("  shots        =", shots)
        print("  chunk shots  =", chunk_shots)
        print("  n_jobs       =", n_jobs)
        print("  noise        =", noise)
    return result


def fit_single_gate_spectrum(csb_df: pd.DataFrame, gate: str) -> Dict[str, float]:
    """Fit one RZ90/RX90/RY90 CSB spectrum without requiring other gates."""
    sub = csb_df[csb_df.target_gate == gate]
    if len(sub) == 0:
        raise ValueError(f"No CSB rows for {gate}")
    qdf = merge_enhanced_quadratures(csb_df, gate)
    complex_fit = fit_complex_csb_eigenvalue(qdf, omega0=np.pi / 2)
    axis = sub[sub.branch == "axis"].sort_values("depth")
    axis_fit = fit_axis_decay_signed_continuous(
        axis.depth.to_numpy(dtype=float),
        axis["mean"].to_numpy(dtype=float),
        axis["stderr"].to_numpy(dtype=float),
    )
    fidelity = process_fidelity_from_complex_offdiag(
        float(complex_fit["g"]),
        float(complex_fit["delta_theta"]),
        float(axis_fit["f"]),
    )
    return {
        "target_gate": gate,
        "g_offdiag": float(complex_fit["g"]),
        "delta_theta": float(complex_fit["delta_theta"]),
        "g_axis": float(axis_fit["f"]),
        **fidelity,
    }


# =============================================================================
# Coherent phase-gate scaling analysis helpers
# =============================================================================


def summarize_coherent_scaling_repeats(
    raw_df: pd.DataFrame,
    summary_columns: Sequence[str],
) -> pd.DataFrame:
    """Summarize independent coherent-noise workflow repeats by physical angle."""
    rows = []
    for epsilon, sub in raw_df.groupby("coherent_angle", sort=True):
        row = {
            "coherent_angle": float(epsilon),
            "n_repeats": int(sub["repeat_id"].nunique()),
            "rep": int(sub["rep"].iloc[0]),
            "max_true_depth": int(sub["max_true_depth"].iloc[0]),
        }
        for column in summary_columns:
            values = sub[column].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            row[f"{column}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{column}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        if "g_offdiag_mean" in row:
            row["offdiag_loss_mean"] = 1.0 - row["g_offdiag_mean"]
            row["offdiag_loss_std"] = row.get("g_offdiag_std", np.nan)
        if "g_axis_mean" in row:
            row["axis_loss_mean"] = 1.0 - row["g_axis_mean"]
            row["axis_loss_std"] = row.get("g_axis_std", np.nan)
        if "delta_theta" in sub:
            values = np.abs(sub["delta_theta"].to_numpy(dtype=float))
            values = values[np.isfinite(values)]
            row["abs_delta_mean"] = float(np.mean(values)) if len(values) else np.nan
            row["abs_delta_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("coherent_angle").reset_index(drop=True)

def fit_fixed_power_small_noise(
    x: np.ndarray,
    y: np.ndarray,
    power: int,
    excluded_x: float,
    yerr: Optional[np.ndarray],
) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    yerr = np.full_like(y, np.nan) if yerr is None else np.asarray(yerr, dtype=float)
    fit_mask = (
        np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        & (x < excluded_x * (1.0 - 1e-12))
    )
    if np.count_nonzero(fit_mask) < 2:
        return {"prefactor": np.nan, "power": float(power), "fit_mask": fit_mask,
                "log_rms_residual": np.nan, "num_fit_points": int(np.count_nonzero(fit_mask))}
    x_fit = x[fit_mask]; y_fit = y[fit_mask]; yerr_fit = yerr[fit_mask]
    log_a_samples = np.log(y_fit) - float(power) * np.log(x_fit)
    relative_error = np.divide(yerr_fit, y_fit, out=np.full_like(y_fit, np.nan), where=y_fit > 0)
    weights = np.ones_like(y_fit)
    usable = np.isfinite(relative_error) & (relative_error > 0)
    weights[usable] = 1.0 / relative_error[usable] ** 2
    log_a = float(np.sum(weights * log_a_samples) / np.sum(weights))
    prefactor = float(np.exp(log_a))
    residual = np.log(y_fit) - np.log(prefactor * x_fit ** power)
    return {"prefactor": prefactor, "power": float(power), "fit_mask": fit_mask,
            "log_rms_residual": float(np.sqrt(np.mean(residual ** 2))),
            "num_fit_points": int(np.count_nonzero(fit_mask))}

def build_fixed_power_fits(
    summary_df: pd.DataFrame,
    specs: Mapping[str, Tuple[str, str, int]],
    excluded_x: float,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Fit notebook-specified fixed-power scaling laws without hardcoded schedules."""
    x = summary_df["coherent_angle"].to_numpy(dtype=float)
    results = {}
    rows = []
    for name, (mean_col, std_col, power) in specs.items():
        result = fit_fixed_power_small_noise(
            x, summary_df[mean_col].to_numpy(dtype=float), int(power), float(excluded_x),
            summary_df[std_col].to_numpy(dtype=float),
        )
        results[name] = result
        rows.append({"quantity": name, "fixed_power": int(power),
                     "prefactor": result["prefactor"], "excluded_epsilon": float(excluded_x),
                     "num_fit_points": result["num_fit_points"],
                     "log_rms_residual": result["log_rms_residual"]})
    return pd.DataFrame(rows), results
