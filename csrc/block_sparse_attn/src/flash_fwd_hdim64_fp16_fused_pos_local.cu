#include "namespace_config.h"
#include "flash_fwd_launch_fused_pos_local.h"

namespace FLASH_NAMESPACE {

void run_mha_fwd_fused_pos_local(Flash_fwd_params &params, cudaStream_t stream) {
    FLASH_NAMESPACE::run_mha_fwd_fused_pos_local<cutlass::half_t>(params, stream);
}

}  // namespace FLASH_NAMESPACE
