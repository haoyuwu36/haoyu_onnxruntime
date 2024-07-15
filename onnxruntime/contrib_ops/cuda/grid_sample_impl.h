// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, bool IsNHWC>
void GridSampleImpl(
    cudaStream_t stream,
    const T* input_data,
    const T* grid_data,
    const int64_t mode,
    const int64_t padding_mode,
    const int64_t align_corners,
    const gsl::span<const int64_t>& dims_input,
    const gsl::span<const int64_t>& dims_grid,
    T* output_data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
