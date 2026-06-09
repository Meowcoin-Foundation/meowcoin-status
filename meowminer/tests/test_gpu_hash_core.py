"""Host-compiled validation of csrc/blake3_core.h against the reference blake3.

The CUDA kernels in csrc/gpu_hash.cu call exactly these functions; compiling
the shared header for the host and matching the reference wheel pins down the
cryptographic core without needing a GPU. Skipped when no C++ compiler is
available.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from blake3 import blake3

CSRC = Path(__file__).parent.parent / "csrc"


def pattern(n: int) -> bytes:
    return bytes((i * 31 + 7) & 0xFF for i in range(n))


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("no g++")
    out = tmp_path_factory.mktemp("b3") / "test_blake3_core"
    subprocess.run(
        ["g++", "-O2", "-std=c++17", str(CSRC / "test_blake3_core.cpp"), "-o", str(out)],
        check=True,
    )
    return out


def run(binary, *args) -> str:
    return subprocess.run([str(binary), *args], capture_output=True, text=True, check=True).stdout.strip()


def test_hash64_keyed_matches_reference(binary):
    expected = blake3(pattern(64), key=pattern(32)).hexdigest()
    assert run(binary, "hash64") == expected


@pytest.mark.parametrize("n_chunks", [1, 2, 4, 16, 64, 256])
def test_chunk_tree_root_matches_reference(binary, n_chunks):
    data = pattern(n_chunks * 1024)
    expected = blake3(data, key=pattern(32)).hexdigest()
    assert run(binary, "chunks", str(n_chunks)) == expected


def test_le256_compare(binary):
    assert run(binary, "le256") == "110"
