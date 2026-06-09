"""PlainProof packaging via the pearl_mining PyO3 bindings.

The wire format (bincode-serialized PlainProof, base64) and the Merkle tree
semantics (keyed blake3, 1024-byte chunk leaves, multileaf proofs) come from
the Pearl repo. We never reimplement them: pearl_mining is built once from
pearl-research-labs/pearl/py-pearl-mining (``maturin build --release``) and
provides MerkleTree / MatrixMerkleProof / PlainProof / verify_plain_proof.

Mirrors zk-pow/src/ffi/mine.rs::build_matrix_proof.
"""

from __future__ import annotations

import numpy as np

from .algo import MiningConfig, pad_to_chunk_boundary
from .engines.base import Hit


def _pearl_mining():
    try:
        import pearl_mining
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "pearl_mining is required to package shares. Build it from the Pearl "
            "monorepo: cd pearl/py-pearl-mining && maturin build --release, then "
            "pip install the wheel on every rig."
        ) from exc
    return pearl_mining


def build_matrix_proof(matrix_i8: np.ndarray, job_key: bytes, row_indices: list[int]):
    """MatrixMerkleProof over the chunk-padded row-major matrix bytes."""
    pm = _pearl_mining()
    rows, cols = matrix_i8.shape
    padded = pad_to_chunk_boundary(matrix_i8.astype(np.uint8).tobytes())
    tree = pm.MerkleTree(padded, job_key)
    leaf_indices = pm.MerkleTree.compute_leaf_indices_from_rows(row_indices, (rows, cols))
    proof = tree.get_multileaf_proof(leaf_indices)
    return pm.MatrixMerkleProof(proof, list(row_indices))


def build_plain_proof_b64(hit: Hit, config: MiningConfig, job_key: bytes) -> str:
    """Assemble and serialize the share for mining.submit."""
    pm = _pearl_mining()
    m, k = hit.a_matrix.shape
    n, _ = hit.bt_matrix.shape
    a_proof = build_matrix_proof(hit.a_matrix, job_key, hit.a_rows)
    bt_proof = build_matrix_proof(hit.bt_matrix, job_key, hit.b_cols)
    proof = pm.PlainProof(m, n, k, config.rank, a_proof, bt_proof)
    return proof.to_base64()


def verify_share_locally(header: bytes, proof_b64: str, share_nbits: int) -> tuple[bool, str]:
    """Pre-submit sanity check, identical to pool-side grading.

    Catching a malformed share locally costs microseconds; a pool-side reject
    costs the share and (on some pools) ban points.
    """
    pm = _pearl_mining()
    verify = getattr(pm, "verify_plain_proof_with_nbits", None)
    hdr = pm.IncompleteBlockHeader.from_bytes(header)
    proof = pm.PlainProof.from_base64(proof_b64)
    if verify is None:
        ok, msg = pm.verify_plain_proof(hdr, proof)
        return bool(ok), f"(block-bound check only; nbits-override binding missing) {msg}"
    ok, msg = verify(hdr, proof, share_nbits)
    return bool(ok), str(msg)
