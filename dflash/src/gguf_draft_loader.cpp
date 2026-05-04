// Loads a DFlash draft model from a GGUF file on disk into a ggml context
// on the CUDA backend.
//
// The draft graph builder (qwen3_dflash_graph.cpp) doesn't care about tensor
// storage types — ggml's ggml_mul_mat handles Q8_0 × F32 dequantization
// transparently.
//
// GGUF arch: "qwen35-dflash-draft" (from convert_dflash_to_gguf.py /
// quantize_draft_q8.py). Tensor naming convention:
//
//   dflash.fc.weight                        [5*hidden, hidden]  Q8_0 / F16
//   dflash.hidden_norm.weight               [hidden]            F32
//   output_norm.weight                      [hidden]            F32
//   blk.<i>.attn_norm.weight                [hidden]            F32
//   blk.<i>.ffn_norm.weight                 [hidden]            F32
//   blk.<i>.attn_q.weight                   [q_dim, hidden]     Q8_0 / F16
//   blk.<i>.attn_k.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_v.weight                   [kv_dim, hidden]    Q8_0 / F16
//   blk.<i>.attn_output.weight              [hidden, q_dim]     Q8_0 / F16
//   blk.<i>.attn_q_norm.weight              [head_dim]          F32
//   blk.<i>.attn_k_norm.weight              [head_dim]          F32
//   blk.<i>.ffn_gate.weight                 [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_up.weight                   [intermediate, hidden]  Q8_0 / F16
//   blk.<i>.ffn_down.weight                 [hidden, intermediate]  Q8_0 / F16

#include "internal.h"

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace dflash27b {

namespace {

struct Mmap {
    void *  addr = nullptr;
    size_t  len  = 0;
#if defined(_WIN32)
    HANDLE  hFile = INVALID_HANDLE_VALUE;
    HANDLE  hMap  = nullptr;
#else
    int     fd   = -1;
#endif

    bool open_ro(const std::string & path, std::string & err) {
#if defined(_WIN32)
        hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hFile == INVALID_HANDLE_VALUE) {
            err = "CreateFileA: " + path + ": error " + std::to_string(GetLastError());
            return false;
        }
        LARGE_INTEGER sz;
        if (!GetFileSizeEx(hFile, &sz)) {
            err = "GetFileSizeEx: error " + std::to_string(GetLastError());
            return false;
        }
        len = (size_t)sz.QuadPart;
        hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hMap) {
            err = "CreateFileMappingA: error " + std::to_string(GetLastError());
            return false;
        }
        addr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
        if (!addr) {
            err = "MapViewOfFile: error " + std::to_string(GetLastError());
            return false;
        }
#else
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { err = "open: " + path + ": " + std::strerror(errno); return false; }
        struct stat st;
        if (::fstat(fd, &st) < 0) { err = "fstat: " + std::string(std::strerror(errno)); return false; }
        len = (size_t)st.st_size;
        addr = ::mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
        if (addr == MAP_FAILED) { err = "mmap: " + std::string(std::strerror(errno)); addr = nullptr; return false; }
#endif
        return true;
    }
    ~Mmap() {
#if defined(_WIN32)
        if (addr)                        UnmapViewOfFile(addr);
        if (hMap)                        CloseHandle(hMap);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
        if (addr) ::munmap(addr, len);
        if (fd >= 0) ::close(fd);
#endif
    }
};

bool get_required_u32(const gguf_context * g, const char * key, uint32_t & out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;
    out = gguf_get_val_u32(g, id);
    return true;
}

bool get_bool_pattern(const gguf_context * g, const char * key,
                      uint32_t n_layer, std::vector<uint8_t> & out) {
    int64_t id = gguf_find_key(g, key);
    if (id < 0) return false;

    out.assign(n_layer, 0);
    const gguf_type type = gguf_get_kv_type(g, id);
    if (type == GGUF_TYPE_UINT32) {
        const uint32_t period = gguf_get_val_u32(g, id);
        if (period == 0) return true;
        for (uint32_t i = 0; i < n_layer; ++i) {
            out[i] = ((i + 1) % period) != 0;
        }
        return true;
    }
    if (type != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(g, id) != GGUF_TYPE_BOOL ||
        gguf_get_arr_n(g, id) < n_layer) {
        return false;
    }

    const int8_t * data = (const int8_t *)gguf_get_arr_data(g, id);
    for (uint32_t i = 0; i < n_layer; ++i) {
        out[i] = data[i] != 0;
    }
    return true;
}

} // namespace

bool load_draft_gguf(const std::string & path,
                     ggml_backend_t       backend,
                     DraftWeights &       out) {

    // ── 1. Parse metadata + create ggml_context with tensor descriptors ──
    ggml_context * meta_ctx = nullptr;
    gguf_init_params gip{};
    gip.no_alloc = true;
    gip.ctx      = &meta_ctx;
    gguf_context * gctx = gguf_init_from_file(path.c_str(), gip);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    // Validate arch
    {
        int64_t arch_id = gguf_find_key(gctx, "general.architecture");
        if (arch_id < 0) {
            set_last_error("missing general.architecture in draft GGUF");
            gguf_free(gctx);
            return false;
        }
        const char * arch = gguf_get_val_str(gctx, arch_id);
        if (std::string(arch) != "qwen35-dflash-draft") {
            set_last_error(std::string("unexpected draft arch: ") + arch +
                           " (expected qwen35-dflash-draft)");
            gguf_free(gctx);
            return false;
        }
    }

    // Read dimensions from GGUF metadata
    int64_t arch_id = gguf_find_key(gctx, "general.architecture");
    const char * A = gguf_get_val_str(gctx, arch_id);
    char key[128];

    auto read_u32 = [&](const char * suffix, uint32_t & out) -> bool {
        std::snprintf(key, sizeof(key), "%s.%s", A, suffix);
        return get_required_u32(gctx, key, out);
    };

    uint32_t n_embd = 0, n_layer = 0, n_ff = 0, n_head = 0;
    uint32_t n_head_kv = 0, head_dim = 0, block_sz = 0, n_tgt_lay = 0, n_swa = 0;
    if (!read_u32("embedding_length", n_embd) ||
        !read_u32("block_count", n_layer) ||
        !read_u32("feed_forward_length", n_ff) ||
        !read_u32("attention.head_count", n_head) ||
        !read_u32("attention.head_count_kv", n_head_kv) ||
        !read_u32("attention.key_length", head_dim) ||
        !read_u32("dflash.block_size", block_sz) ||
        !read_u32("dflash.n_target_layers", n_tgt_lay) ||
        !read_u32("attention.sliding_window", n_swa) ||
        n_embd == 0 || n_layer == 0 || n_ff == 0 || n_head == 0 ||
        n_head_kv == 0 || head_dim == 0 || n_swa == 0) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "draft GGUF: missing hparams: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u block_size=%u "
            "n_target_layers=%u sliding_window=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim,
            block_sz, n_tgt_lay, n_swa);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // The draft graph builder still hardcodes block_size and the number of
    // captured target layers (drives fc weight shape and capture_layer_ids
    // array length). Reject GGUFs whose metadata disagrees with the compiled
    // constants, otherwise we would silently mis-shape the graph.
    if (block_sz != (uint32_t)DFLASH27B_DRAFT_BLOCK_SIZE ||
        n_tgt_lay != (uint32_t)DFLASH27B_DRAFT_N_TARGET_LAYERS) {
        char buf[256];
        std::snprintf(buf, sizeof(buf),
            "draft GGUF: dflash.block_size=%u (expected %d), "
            "dflash.n_target_layers=%u (expected %d)",
            block_sz, DFLASH27B_DRAFT_BLOCK_SIZE,
            n_tgt_lay, DFLASH27B_DRAFT_N_TARGET_LAYERS);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // Upper bounds on hparams. Guards against malformed/hostile GGUFs that
    // would otherwise trigger huge allocations or signed-int overflow when
    // narrowed below. Limits chosen well above any plausible LLM config.
    constexpr uint32_t MAX_LAYERS  = 1024;
    constexpr uint32_t MAX_EMBD    = 1u << 17;   // 131072
    constexpr uint32_t MAX_FF      = 1u << 19;   // 524288
    constexpr uint32_t MAX_HEADS   = 1024;
    constexpr uint32_t MAX_HEADDIM = 1024;
    if (n_layer   > MAX_LAYERS  || n_embd    > MAX_EMBD  ||
        n_ff      > MAX_FF      || n_head    > MAX_HEADS ||
        n_head_kv > MAX_HEADS   || head_dim  > MAX_HEADDIM ||
        n_head_kv > n_head      || (n_head % n_head_kv) != 0) {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
            "draft GGUF: hparams out of range: n_embd=%u n_layer=%u n_ff=%u "
            "n_head=%u n_head_kv=%u head_dim=%u",
            n_embd, n_layer, n_ff, n_head, n_head_kv, head_dim);
        set_last_error(buf);
        gguf_free(gctx);
        return false;
    }

    // ── 2. Wire tensor pointers into DraftWeights ────────────────────────
    out.ctx     = meta_ctx;
    out.backend = backend;
    out.n_layer   = (int)n_layer;
    out.n_head    = (int)n_head;
    out.n_head_kv = (int)n_head_kv;
    out.head_dim  = (int)head_dim;
    out.n_embd    = (int)n_embd;
    out.n_ff      = (int)n_ff;
    out.layers.assign((size_t)n_layer, DraftLayer{});
    out.sliding_window = (int)n_swa;
    out.layer_is_swa.assign((size_t)n_layer, 0);
    std::snprintf(key, sizeof(key), "%s.%s", A, "attention.sliding_window_pattern");
    if (!get_bool_pattern(gctx, key, n_layer, out.layer_is_swa)) {
        set_last_error("draft GGUF: attention.sliding_window_pattern is missing or invalid");
        gguf_free(gctx);
        return false;
    }

    auto g = [&](const char * name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name);
    };

    out.fc          = g("dflash.fc.weight");
    out.hidden_norm = g("dflash.hidden_norm.weight");
    out.out_norm    = g("output_norm.weight");
    if (!out.fc || !out.hidden_norm || !out.out_norm) {
        set_last_error("draft GGUF: missing top-level tensors "
                       "(dflash.fc / dflash.hidden_norm / output_norm)");
        gguf_free(gctx);
        return false;
    }

    for (int il = 0; il < out.n_layer; il++) {
        char name[128];
        auto fnd = [&](const char * suffix) -> ggml_tensor * {
            std::snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return ggml_get_tensor(meta_ctx, name);
        };
        DraftLayer & L = out.layers[il];
        L.attn_norm = fnd("attn_norm.weight");
        L.ffn_norm  = fnd("ffn_norm.weight");
        L.wq        = fnd("attn_q.weight");
        L.wk        = fnd("attn_k.weight");
        L.wv        = fnd("attn_v.weight");
        L.wo        = fnd("attn_output.weight");
        L.q_norm    = fnd("attn_q_norm.weight");
        L.k_norm    = fnd("attn_k_norm.weight");
        L.w_gate    = fnd("ffn_gate.weight");
        L.w_up      = fnd("ffn_up.weight");
        L.w_down    = fnd("ffn_down.weight");
        if (!L.attn_norm || !L.ffn_norm || !L.wq || !L.wk || !L.wv || !L.wo ||
            !L.q_norm || !L.k_norm || !L.w_gate || !L.w_up || !L.w_down) {
            char b[128];
            std::snprintf(b, sizeof(b), "draft GGUF: layer %d missing tensors", il);
            set_last_error(b);
            gguf_free(gctx);
            return false;
        }
    }

    // ── 3. Allocate CUDA buffer for all tensors ──────────────────────────
    out.buf = ggml_backend_alloc_ctx_tensors(meta_ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed (draft GGUF)");
        gguf_free(gctx);
        return false;
    }

    // ── 4. mmap file and copy tensor bytes to CUDA ───────────────────────
    std::string err;
    Mmap mm;
    if (!mm.open_ro(path, err)) { set_last_error(err); gguf_free(gctx); return false; }
    const size_t data_start = gguf_get_data_offset(gctx);
    const int64_t n_tensors = gguf_get_n_tensors(gctx);

    size_t total = 0;
    for (int64_t tid = 0; tid < n_tensors; tid++) {
        const char * tname = gguf_get_tensor_name(gctx, tid);
        ggml_tensor * t = ggml_get_tensor(meta_ctx, tname);
        if (!t) continue;
        const size_t off = data_start + gguf_get_tensor_offset(gctx, tid);
        const size_t sz  = gguf_get_tensor_size(gctx, tid);
        if (off + sz > mm.len) {
            set_last_error(std::string("draft GGUF: tensor '") + tname + "' overflows file");
            gguf_free(gctx);
            return false;
        }
        ggml_backend_tensor_set(t, (const uint8_t *)mm.addr + off, 0, sz);
        total += sz;
    }

    gguf_free(gctx);

    char summary[192];
    std::snprintf(summary, sizeof(summary),
        "draft GGUF loaded: %" PRId64 " tensors, %.2f GiB on GPU",
        n_tensors, total / (1024.0 * 1024.0 * 1024.0));
    set_last_error(summary);

    return true;
}

bool load_draft_model(const std::string & path,
                      ggml_backend_t       backend,
                      DraftWeights &       out) {
    if (path.size() < 5 || path.substr(path.size() - 5) != ".gguf") {
        set_last_error("draft model must be a quantized DFlash GGUF (.gguf): " + path);
        return false;
    }
    return load_draft_gguf(path, backend, out);
}

void free_draft_weights(DraftWeights & w) {
    if (w.buf) {
        ggml_backend_buffer_free(w.buf);
        w.buf = nullptr;
    }
    if (w.ctx) {
        ggml_free(w.ctx);
        w.ctx = nullptr;
    }
    w.layers.clear();
    w.layer_is_swa.clear();
    w.fc = nullptr;
    w.hidden_norm = nullptr;
    w.out_norm = nullptr;
    w.n_layer = DFLASH27B_DRAFT_LAYERS;
    w.n_head = DFLASH27B_TARGET_N_HEADS;
    w.n_head_kv = DFLASH27B_TARGET_N_KV_HEADS;
    w.head_dim = DFLASH27B_TARGET_HEAD_DIM;
    w.n_embd = DFLASH27B_TARGET_HIDDEN;
    w.n_ff = DFLASH27B_TARGET_INTERMEDIATE;
    w.sliding_window = 0;
}

} // namespace dflash27b
