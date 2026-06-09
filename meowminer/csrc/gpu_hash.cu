// MeowMiner GPU hashing extension (v1): keyed BLAKE3 of the matrix
// commitments and jackpot grading on-device, removing the v0 CPU bottleneck.
//
// Exposed (via torch extension) as:
//   blake3_root(data_u8_cuda, key_u8x32) -> 32-byte tensor       (pow2 chunks)
//   jackpot_grade(jackpots_u32[N,16], key_u8x32, bound_le_u32x8)
//       -> (hits_idx_i64[H], hashes_u8[N,32])
//
// Built for sm_89 first (RTX 4070 Ti Super et al.); the code is generic CUDA
// and runs on any sm_70+.

#include <cuda_runtime.h>

#include "blake3_core.h"

// MEOWMINER_KERNELS_ONLY: compile just the __global__ kernels (used by CI to
// validate the device code for sm_89 without the torch host glue, whose
// headers are picky about host-compiler pairings).
#ifndef MEOWMINER_KERNELS_ONLY
#include <torch/extension.h>
#endif

#ifndef MEOWMINER_KERNELS_ONLY
#define CUDA_CHECK(expr)                                                        \
    do {                                                                        \
        cudaError_t _e = (expr);                                                \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
    } while (0)

#endif  // MEOWMINER_KERNELS_ONLY

// ── kernels ──────────────────────────────────────────────────────────────────

// One thread per 1024-byte chunk.
__global__ void k_chunk_cvs(const uint8_t* __restrict__ data, int64_t n_chunks,
                            const uint32_t* __restrict__ key, bool single_chunk_root,
                            uint32_t* __restrict__ out_cvs) {
    int64_t c = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (c >= n_chunks) return;
    uint32_t key_local[8];
#pragma unroll
    for (int i = 0; i < 8; i++) key_local[i] = key[i];
    b3::chunk_cv(data + c * b3::CHUNK_LEN, (uint64_t)c, key_local, single_chunk_root,
                 out_cvs + c * 8);
}

// One thread per parent at one tree level. n_parents = n_nodes / 2.
__global__ void k_parent_cvs(const uint32_t* __restrict__ in_cvs, int64_t n_parents,
                             const uint32_t* __restrict__ key, bool root_level,
                             uint32_t* __restrict__ out_cvs) {
    int64_t p = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (p >= n_parents) return;
    uint32_t key_local[8];
#pragma unroll
    for (int i = 0; i < 8; i++) key_local[i] = key[i];
    bool root = root_level && (n_parents == 1);
    b3::parent_cv(in_cvs + (2 * p) * 8, in_cvs + (2 * p + 1) * 8, key_local, root,
                  out_cvs + p * 8);
}

// One thread per jackpot: keyed blake3 of 64 bytes + LE-256 bound compare.
__global__ void k_jackpot(const uint32_t* __restrict__ jackpots, int64_t n,
                          const uint32_t* __restrict__ key,
                          const uint32_t* __restrict__ bound_le,
                          uint8_t* __restrict__ out_hashes, uint8_t* __restrict__ out_hits) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t key_local[8], bound_local[8], cv[8];
#pragma unroll
    for (int k = 0; k < 8; k++) key_local[k] = key[k];
#pragma unroll
    for (int k = 0; k < 8; k++) bound_local[k] = bound_le[k];
    // The 16xu32 jackpot is already little-endian words == its byte serialization.
    b3::hash64_keyed(reinterpret_cast<const uint8_t*>(jackpots + i * 16), key_local, cv);
    uint8_t* out = out_hashes + i * 32;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        out[4 * k + 0] = (uint8_t)(cv[k]);
        out[4 * k + 1] = (uint8_t)(cv[k] >> 8);
        out[4 * k + 2] = (uint8_t)(cv[k] >> 16);
        out[4 * k + 3] = (uint8_t)(cv[k] >> 24);
    }
    out_hits[i] = b3::le256_lte(cv, bound_local) ? 1 : 0;
}

#ifndef MEOWMINER_KERNELS_ONLY
// ── host wrappers ─────────────────────────────────────────────────────────────

// Reinterpret a 32-byte key as 8 little-endian u32 words on `dev`.
static torch::Tensor key_words_tensor(const torch::Tensor& key, const torch::Device& dev) {
    TORCH_CHECK(key.dtype() == torch::kUInt8 && key.numel() == 32, "key must be 32 uint8");
    return key.to(dev).contiguous().view(torch::kInt32).clone();
}

torch::Tensor blake3_root(torch::Tensor data, torch::Tensor key) {
    TORCH_CHECK(data.is_cuda() && data.dtype() == torch::kUInt8, "data must be uint8 CUDA");
    TORCH_CHECK(data.is_contiguous(), "data must be contiguous");
    int64_t n_bytes = data.numel();
    TORCH_CHECK(n_bytes > 0 && n_bytes % b3::CHUNK_LEN == 0, "data must be chunk-padded");
    int64_t n_chunks = n_bytes / b3::CHUNK_LEN;
    TORCH_CHECK((n_chunks & (n_chunks - 1)) == 0, "chunk count must be a power of two");

    auto dev = data.device();
    auto key_words = key_words_tensor(key, dev);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(dev);
    auto cvs = torch::empty({n_chunks * 8}, opts);

    const int threads = 256;
    int64_t blocks = (n_chunks + threads - 1) / threads;
    k_chunk_cvs<<<blocks, threads>>>(data.data_ptr<uint8_t>(), n_chunks,
                                     reinterpret_cast<uint32_t*>(key_words.data_ptr<int32_t>()),
                                     n_chunks == 1, reinterpret_cast<uint32_t*>(cvs.data_ptr<int32_t>()));
    CUDA_CHECK(cudaGetLastError());

    auto cur = cvs;
    int64_t nodes = n_chunks;
    while (nodes > 1) {
        int64_t parents = nodes / 2;
        auto nxt = torch::empty({parents * 8}, opts);
        int64_t pblocks = (parents + threads - 1) / threads;
        k_parent_cvs<<<pblocks, threads>>>(
            reinterpret_cast<uint32_t*>(cur.data_ptr<int32_t>()), parents,
            reinterpret_cast<uint32_t*>(key_words.data_ptr<int32_t>()),
            /*root_level=*/parents == 1,
            reinterpret_cast<uint32_t*>(nxt.data_ptr<int32_t>()));
        CUDA_CHECK(cudaGetLastError());
        cur = nxt;
        nodes = parents;
    }
    return cur.cpu().contiguous().view(torch::kUInt8).clone();  // 8 LE words -> 32 bytes
}

std::tuple<torch::Tensor, torch::Tensor> jackpot_grade(torch::Tensor jackpots, torch::Tensor key,
                                                       torch::Tensor bound_le_words) {
    TORCH_CHECK(jackpots.is_cuda() && jackpots.dtype() == torch::kInt32, "jackpots must be int32 CUDA");
    TORCH_CHECK(jackpots.dim() == 2 && jackpots.size(1) == 16, "jackpots must be [N,16]");
    TORCH_CHECK(jackpots.is_contiguous(), "jackpots must be contiguous");
    TORCH_CHECK(bound_le_words.dtype() == torch::kInt32 && bound_le_words.numel() == 8,
                "bound must be 8 int32 LE words");
    int64_t n = jackpots.size(0);
    auto dev = jackpots.device();
    auto key_words = key_words_tensor(key, dev);
    auto bound_dev = bound_le_words.to(dev).contiguous();
    auto hashes = torch::empty({n, 32}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    auto hits = torch::empty({n}, torch::TensorOptions().dtype(torch::kUInt8).device(dev));

    const int threads = 256;
    int64_t blocks = (n + threads - 1) / threads;
    k_jackpot<<<blocks, threads>>>(reinterpret_cast<uint32_t*>(jackpots.data_ptr<int32_t>()), n,
                                   reinterpret_cast<uint32_t*>(key_words.data_ptr<int32_t>()),
                                   reinterpret_cast<uint32_t*>(bound_dev.data_ptr<int32_t>()),
                                   hashes.data_ptr<uint8_t>(), hits.data_ptr<uint8_t>());
    CUDA_CHECK(cudaGetLastError());
    auto idx = torch::nonzero(hits).flatten();
    return {idx, hashes};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("blake3_root", &blake3_root, "Keyed BLAKE3 root of chunk-padded CUDA buffer");
    m.def("jackpot_grade", &jackpot_grade, "Keyed BLAKE3 + bound compare of [N,16] jackpots");
}
#endif  // MEOWMINER_KERNELS_ONLY
