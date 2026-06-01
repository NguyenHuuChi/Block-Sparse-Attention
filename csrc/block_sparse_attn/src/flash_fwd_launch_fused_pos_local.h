/****************************************************************************
 * Local fused relative-position launcher.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>
#include <stdexcept>

#include "static_switch.h"
#include "hardware_info.h"
#include "flash_params.h"
#include "flash_fwd_kernel_fused_pos_local.h"

namespace FLASH_NAMESPACE {

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K>
__global__ void flash_fwd_kernel_fused_pos_local(Flash_fwd_params params) {
    FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_even_MN, Is_even_K>(params);
}

template<typename Kernel_traits>
void run_flash_fwd_fused_pos_local(Flash_fwd_params &params, cudaStream_t stream) {
    const size_t smem_size = Kernel_traits::kSmemSize
        + 2 * Kernel_traits::kBlockM * 14 * sizeof(typename Kernel_traits::Element);
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    const bool is_even_MN = params.cu_seqlens_q == nullptr
                         && params.cu_seqlens_k == nullptr
                         && params.seqlen_k % Kernel_traits::kBlockN == 0
                         && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;

    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        BOOL_SWITCH(is_even_K, IsEvenKConst, [&] {
            auto kernel = &flash_fwd_kernel_fused_pos_local<
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

template<typename T>
void run_mha_fwd_hdim64_fused_pos_local_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    run_flash_fwd_fused_pos_local<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T, 64, 64>>(params, stream);
}

template<typename T>
void run_mha_fwd_fused_pos_local(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_fused_pos_local_dispatch<T>(params, stream);
}

}  // namespace FLASH_NAMESPACE
