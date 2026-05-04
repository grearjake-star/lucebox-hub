#include "mega_pflash_native.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

extern "C" void launch_prefill_bf16_mega(
    const int *token_ids, int seq_len, int max_seq_len, int *output_token,
    const __nv_bfloat16 *embed_weight, const void *layers,
    const __nv_bfloat16 *final_norm_w, const __nv_bfloat16 *lm_head_w,
    __nv_bfloat16 *fa_k_cache, __nv_bfloat16 *fa_v_cache,
    float *dn_states, float *conv_bufs,
    __nv_bfloat16 *hidden, __nv_bfloat16 *residual, __nv_bfloat16 *normalized,
    __nv_bfloat16 *proj_buf, __nv_bfloat16 *proj_buf2,
    __nv_bfloat16 *attn_buf, __nv_bfloat16 *mlp_buf,
    __nv_bfloat16 *dn_out_buf,
    float *beta_buf, float *alpha_buf,
    __nv_bfloat16 *final_normed, __nv_bfloat16 *hidden_bf16_out,
    float *lm_bmv, int *lm_bmi,
    cudaStream_t stream);

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

} // namespace

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "usage: " << argv[0]
                  << " <qwen3.5-0.8b-safetensors-or-dir> <seq_len> [max_seq_len]\n";
        return 2;
    }

    const std::string model_path = argv[1];
    const int seq_len = std::max(32, std::atoi(argv[2]));
    const int max_seq_len = argc >= 4 ? std::max(seq_len, std::atoi(argv[3])) : seq_len;
    if ((seq_len % 32) != 0) {
        std::cerr << "smoke_prefill_megakernel requires seq_len multiple of 32\n";
        return 2;
    }

    setenv("DFLASH_MEGA_PFLASH_WITH_LM_HEAD", "1", 1);

    dflash27b::MegaPFlashContext ctx;
    if (!dflash27b::load_mega_pflash(model_path, max_seq_len, ctx)) {
        std::cerr << "load_mega_pflash failed\n";
        return 1;
    }

    std::vector<int32_t> ids(seq_len);
    for (int i = 0; i < seq_len; ++i) {
        ids[i] = 1000 + (i % 30000);
    }

    int *ids_dev = nullptr;
    if (cudaMalloc(&ids_dev, ids.size() * sizeof(int32_t)) != cudaSuccess) {
        std::cerr << "cudaMalloc ids failed\n";
        dflash27b::free_mega_pflash(ctx);
        return 1;
    }
    cudaMemcpy(ids_dev, ids.data(), ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemset(ctx.dn_states, 0, (size_t)18 * 16 * 128 * 128 * sizeof(float));
    cudaMemset(ctx.conv_bufs, 0, (size_t)18 * 6144 * 4 * sizeof(float));

    const auto t0 = std::chrono::steady_clock::now();
    launch_prefill_bf16_mega(
        ids_dev, seq_len, max_seq_len, static_cast<int *>(ctx.output_token),
        static_cast<const __nv_bfloat16 *>(ctx.embed_weight),
        ctx.layer_weights_dev,
        static_cast<const __nv_bfloat16 *>(ctx.final_norm_weight),
        static_cast<const __nv_bfloat16 *>(ctx.lm_head_weight),
        static_cast<__nv_bfloat16 *>(ctx.fa_k_cache),
        static_cast<__nv_bfloat16 *>(ctx.fa_v_cache),
        static_cast<float *>(ctx.dn_states),
        static_cast<float *>(ctx.conv_bufs),
        static_cast<__nv_bfloat16 *>(ctx.hidden),
        static_cast<__nv_bfloat16 *>(ctx.residual),
        static_cast<__nv_bfloat16 *>(ctx.normalized),
        static_cast<__nv_bfloat16 *>(ctx.proj_buf),
        static_cast<__nv_bfloat16 *>(ctx.proj_buf2),
        static_cast<__nv_bfloat16 *>(ctx.attn_buf),
        static_cast<__nv_bfloat16 *>(ctx.mlp_buf),
        static_cast<__nv_bfloat16 *>(ctx.dn_out_buf),
        static_cast<float *>(ctx.beta_buf),
        static_cast<float *>(ctx.alpha_buf),
        static_cast<__nv_bfloat16 *>(ctx.final_normed),
        static_cast<__nv_bfloat16 *>(ctx.hidden_bf16_out),
        static_cast<float *>(ctx.lm_bmv),
        static_cast<int *>(ctx.lm_bmi),
        0);
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "launch_prefill_bf16_mega failed: " << cudaGetErrorString(err) << "\n";
        cudaFree(ids_dev);
        dflash27b::free_mega_pflash(ctx);
        return 1;
    }

    int token = -1;
    cudaMemcpy(&token, ctx.output_token, sizeof(token), cudaMemcpyDeviceToHost);
    std::cout << "prefill megakernel smoke ok"
              << " seq_len=" << seq_len
              << " max_seq_len=" << max_seq_len
              << " token=" << token
              << " launch_ms=" << elapsed_ms(t0, t1)
              << "\n";

    cudaFree(ids_dev);
    dflash27b::free_mega_pflash(ctx);
    return 0;
}
