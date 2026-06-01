/****************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Hoai-Chau Tran and Huu-Chi Nguyen.
 * Simplified launcher for fused relative-position SAM inference.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>
#include <stdexcept>

#include "static_switch.h"
#include "hardware_info.h"
#include "flash_params.h"
#include "flash_fwd_kernel_fused_pos.h"

namespace FLASH_NAMESPACE {

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_kernel_fused_pos(Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_even_MN, Is_even_K>(params);
}

template<typename Kernel_traits>
void run_flash_fwd_fused_pos(Flash_fwd_params &params, cudaStream_t stream) {
    // constexpr size_t smem_size = Kernel_traits::kSmemSize + params.rel_h_last_dim * Kernel_traits::kBlockM * sizeof(typename Kernel_traits::Element);
    // constexpr size_t smem_size = Kernel_traits::kSmemSize + size(typename Kernel_traits::SmemLayoutRelH{}) * sizeof(typename Kernel_traits::Element);
    const size_t smem_size = Kernel_traits::kSmemSize
        + size(typename Kernel_traits::SmemLayoutRelH{}) * sizeof(typename Kernel_traits::Element)
        + size(typename Kernel_traits::SmemLayoutRelW{}) * sizeof(typename Kernel_traits::Element);
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    const bool is_even_MN = params.cu_seqlens_q == nullptr
                         && params.cu_seqlens_k == nullptr
                         && params.seqlen_k % Kernel_traits::kBlockN == 0
                         && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            auto kernel = &flash_fwd_kernel_fused_pos<
                Kernel_traits,
                IsEvenMNConst && IsEvenKConst,
                IsEvenKConst
            >;
            if (smem_size >= 48 * 1024) {
                C10_CUDA_CHECK(cudaFuncSetAttribute(
                    kernel,
                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                    smem_size
                ));
            }
            kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    });
}

// template<typename T>
// void run_mha_fwd_hdim64_fused_pos(Flash_fwd_params &params, cudaStream_t stream) {
//     constexpr static int Headdim = 64;
//     run_flash_fwd_fused_pos<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>>(params, stream);
// }

// }  // namespace FLASH_NAMESPACE

// try with different block size
template<typename T, int RelHLastDim, int RelWLastDim>
void run_mha_fwd_hdim64_fused_pos_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    run_flash_fwd_fused_pos<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T, RelHLastDim, RelWLastDim>>(params, stream);
}

template<typename T>
void run_mha_fwd_fused_pos(Flash_fwd_params &params, cudaStream_t stream) {

    if (params.rel_h_last_dim == 64 && params.rel_w_last_dim == 64) {
        run_mha_fwd_hdim64_fused_pos_dispatch<T, 64, 64>(params, stream);
    } else {
        throw std::runtime_error("Global-only fused-pos kernel expects (rel_h_last_dim, rel_w_last_dim) == (64, 64)");
    }
}

}