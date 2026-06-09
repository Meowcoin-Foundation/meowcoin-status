// Portable BLAKE3 core shared by the CUDA kernels (gpu_hash.cu) and the host
// self-test (test_blake3_core.cpp). Every function is constexpr-free plain
// C++ compiled for both host and device, so the exact arithmetic that runs on
// the GPU is unit-tested on the CPU against the reference `blake3` wheel.
//
// Scope: keyed hashing of chunk-padded buffers whose chunk count is a power
// of two (meowminer pads matrices to the 1024-byte boundary and uses
// power-of-two dimensions, so the BLAKE3 left-leaning tree degenerates to a
// perfect binary tree), plus one-shot keyed hashing of 64-byte jackpots.
#pragma once

#include <stdint.h>

#if defined(__CUDACC__)
#define B3_HD __host__ __device__ __forceinline__
#else
#define B3_HD inline
#endif

namespace b3 {

// NOTE: kept function-local below — namespace-scope constexpr arrays are not
// indexable from CUDA device code without __device__ duplication.
B3_HD void iv_words(uint32_t out[4]) {
    out[0] = 0x6A09E667u;
    out[1] = 0xBB67AE85u;
    out[2] = 0x3C6EF372u;
    out[3] = 0xA54FF53Au;
}

constexpr uint32_t CHUNK_START = 1u << 0;
constexpr uint32_t CHUNK_END = 1u << 1;
constexpr uint32_t PARENT = 1u << 2;
constexpr uint32_t ROOT = 1u << 3;
constexpr uint32_t KEYED_HASH = 1u << 4;

constexpr int BLOCK_LEN = 64;
constexpr int CHUNK_LEN = 1024;
constexpr int BLOCKS_PER_CHUNK = CHUNK_LEN / BLOCK_LEN;



B3_HD uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

B3_HD void g(uint32_t* v, int a, int b, int c, int d, uint32_t mx, uint32_t my) {
    v[a] = v[a] + v[b] + mx;
    v[d] = rotr32(v[d] ^ v[a], 16);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 12);
    v[a] = v[a] + v[b] + my;
    v[d] = rotr32(v[d] ^ v[a], 8);
    v[c] = v[c] + v[d];
    v[b] = rotr32(v[b] ^ v[c], 7);
}

B3_HD void round_fn(uint32_t* v, const uint32_t* m) {
    g(v, 0, 4, 8, 12, m[0], m[1]);
    g(v, 1, 5, 9, 13, m[2], m[3]);
    g(v, 2, 6, 10, 14, m[4], m[5]);
    g(v, 3, 7, 11, 15, m[6], m[7]);
    g(v, 0, 5, 10, 15, m[8], m[9]);
    g(v, 1, 6, 11, 12, m[10], m[11]);
    g(v, 2, 7, 8, 13, m[12], m[13]);
    g(v, 3, 4, 9, 14, m[14], m[15]);
}

// One compression: cv (8 words) + 64-byte block (16 LE words) -> new cv.
B3_HD void compress(const uint32_t cv[8], const uint32_t block[16], uint64_t counter,
                    uint32_t block_len, uint32_t flags, uint32_t out_cv[8]) {
    uint32_t v[16];
    uint32_t m[16];
    for (int i = 0; i < 8; i++) v[i] = cv[i];
    uint32_t iv[4];
    iv_words(iv);
    for (int i = 0; i < 4; i++) v[8 + i] = iv[i];
    v[12] = (uint32_t)counter;
    v[13] = (uint32_t)(counter >> 32);
    v[14] = block_len;
    v[15] = flags;
    for (int i = 0; i < 16; i++) m[i] = block[i];

    // Message word schedule applied between rounds (function-local for CUDA).
    const uint8_t msg_perm[16] = {2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8};
    for (int r = 0; r < 7; r++) {
        round_fn(v, m);
        if (r < 6) {
            uint32_t p[16];
            for (int i = 0; i < 16; i++) p[i] = m[msg_perm[i]];
            for (int i = 0; i < 16; i++) m[i] = p[i];
        }
    }
    for (int i = 0; i < 8; i++) out_cv[i] = v[i] ^ v[i + 8];
}

B3_HD void load_words(const uint8_t* bytes, int n_words, uint32_t* out) {
    for (int i = 0; i < n_words; i++) {
        out[i] = (uint32_t)bytes[4 * i] | ((uint32_t)bytes[4 * i + 1] << 8) |
                 ((uint32_t)bytes[4 * i + 2] << 16) | ((uint32_t)bytes[4 * i + 3] << 24);
    }
}

// Hash one FULL 1024-byte chunk (16 x 64B blocks) with the given key.
// `root` must be true only for a single-chunk message.
B3_HD void chunk_cv(const uint8_t* chunk, uint64_t chunk_counter, const uint32_t key[8],
                    bool root, uint32_t out_cv[8]) {
    uint32_t cv[8];
    for (int i = 0; i < 8; i++) cv[i] = key[i];
    for (int b = 0; b < BLOCKS_PER_CHUNK; b++) {
        uint32_t block[16];
        load_words(chunk + b * BLOCK_LEN, 16, block);
        uint32_t flags = KEYED_HASH;
        if (b == 0) flags |= CHUNK_START;
        if (b == BLOCKS_PER_CHUNK - 1) {
            flags |= CHUNK_END;
            if (root) flags |= ROOT;
        }
        compress(cv, block, chunk_counter, BLOCK_LEN, flags, cv);
    }
    for (int i = 0; i < 8; i++) out_cv[i] = cv[i];
}

// Merge two child CVs into a parent CV.
B3_HD void parent_cv(const uint32_t left[8], const uint32_t right[8], const uint32_t key[8],
                     bool root, uint32_t out_cv[8]) {
    uint32_t block[16];
    for (int i = 0; i < 8; i++) block[i] = left[i];
    for (int i = 0; i < 8; i++) block[8 + i] = right[i];
    uint32_t flags = PARENT | KEYED_HASH | (root ? ROOT : 0);
    compress(key, block, 0, BLOCK_LEN, flags, out_cv);
}

// One-shot keyed hash of exactly 64 bytes (the 16xu32 jackpot transcript).
B3_HD void hash64_keyed(const uint8_t msg[64], const uint32_t key[8], uint32_t out_cv[8]) {
    uint32_t block[16];
    load_words(msg, 16, block);
    compress(key, block, 0, BLOCK_LEN, CHUNK_START | CHUNK_END | ROOT | KEYED_HASH, out_cv);
}

// hash_jackpot <= bound as 256-bit LITTLE-endian integers. Both arguments are
// 8 little-endian u32 words, least-significant word first.
B3_HD bool le256_lte(const uint32_t hash_words[8], const uint32_t bound_words[8]) {
    for (int i = 7; i >= 0; i--) {
        if (hash_words[i] < bound_words[i]) return true;
        if (hash_words[i] > bound_words[i]) return false;
    }
    return true;
}

}  // namespace b3
