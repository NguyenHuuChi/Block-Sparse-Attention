/****************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Adapted by Hoai-Chau Tran and Huu-Chi Nguyen.
 * Vision transformer fused relative-position kernel instantiation for head_dim=64, fp16.
 ******************************************************************************/

#include "namespace_config.h"
#include "flash_fwd_launch_fused_pos.h"

namespace FLASH_NAMESPACE {

// void    (Flash_fwd_params &params, cudaStream_t stream) {
void run_mha_fwd_fused_pos(Flash_fwd_params &params, cudaStream_t stream) {
    // run_mha_fwd_hdim64_fused_pos<cutlass::half_t>(params, stream);
    FLASH_NAMESPACE::run_mha_fwd_fused_pos<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
