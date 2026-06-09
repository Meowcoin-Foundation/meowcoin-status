// Host self-test for blake3_core.h: prints keyed hashes for fixed inputs as
// hex; tests/test_gpu_hash_core.py compares them to the reference `blake3`
// wheel. The exact same code paths run inside the CUDA kernels.
//
// Usage: test_blake3_core <mode> where mode is:
//   hash64     - keyed hash of the 64-byte pattern (jackpot path)
//   chunks N   - keyed root of N 1024-byte chunks (commitment path), N pow2
//   le256      - bound compare truth table
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "blake3_core.h"

static void fill_pattern(uint8_t* buf, size_t n) {
    for (size_t i = 0; i < n; i++) buf[i] = (uint8_t)((i * 31 + 7) & 0xFF);
}

static void print_hex(const uint32_t cv[8]) {
    for (int i = 0; i < 8; i++)
        printf("%02x%02x%02x%02x", cv[i] & 0xFF, (cv[i] >> 8) & 0xFF, (cv[i] >> 16) & 0xFF,
               (cv[i] >> 24) & 0xFF);
    printf("\n");
}

int main(int argc, char** argv) {
    if (argc < 2) return 2;
    uint8_t key_bytes[32];
    fill_pattern(key_bytes, 32);
    uint32_t key[8];
    b3::load_words(key_bytes, 8, key);

    if (!strcmp(argv[1], "hash64")) {
        uint8_t msg[64];
        fill_pattern(msg, 64);
        uint32_t cv[8];
        b3::hash64_keyed(msg, key, cv);
        print_hex(cv);
        return 0;
    }
    if (!strcmp(argv[1], "chunks")) {
        long n = atol(argv[2]);
        std::vector<uint8_t> data(n * b3::CHUNK_LEN);
        fill_pattern(data.data(), data.size());
        // chunk CVs
        std::vector<uint32_t> cvs(n * 8);
        for (long c = 0; c < n; c++)
            b3::chunk_cv(data.data() + c * b3::CHUNK_LEN, (uint64_t)c, key, n == 1, &cvs[c * 8]);
        // pairwise parent reduction (perfect tree: n is a power of two)
        long nodes = n;
        while (nodes > 1) {
            long parents = nodes / 2;
            std::vector<uint32_t> nxt(parents * 8);
            for (long p = 0; p < parents; p++)
                b3::parent_cv(&cvs[(2 * p) * 8], &cvs[(2 * p + 1) * 8], key, parents == 1,
                              &nxt[p * 8]);
            cvs = nxt;
            nodes = parents;
        }
        print_hex(cvs.data());
        return 0;
    }
    if (!strcmp(argv[1], "le256")) {
        uint32_t a[8] = {5, 0, 0, 0, 0, 0, 0, 1};
        uint32_t b_eq[8] = {5, 0, 0, 0, 0, 0, 0, 1};
        uint32_t b_hi[8] = {4, 0, 0, 0, 0, 0, 0, 2};   // bigger in the MS word
        uint32_t b_lo[8] = {9, 9, 9, 9, 9, 9, 9, 0};   // smaller in the MS word
        printf("%d%d%d\n", b3::le256_lte(a, b_eq), b3::le256_lte(a, b_hi), b3::le256_lte(a, b_lo));
        return 0;
    }
    return 2;
}
