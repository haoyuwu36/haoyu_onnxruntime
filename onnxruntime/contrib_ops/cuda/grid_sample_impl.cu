// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "grid_sample_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
    namespace contrib {
        namespace cuda {
            template<typename T>
            __device__ T GsDenormalize(T n, int64_t length, bool align_corners) {
                T x = {};
                if (align_corners) {  // align_corners: true => [-1, 1] to [0, length - 1]
                    x = (n + static_cast<T>(1)) / static_cast<T>(2) * (length - 1);
                } else {  // align_corners: false => [-1, 1] to [-0.5, length - 0.5]
                    x = ((n + static_cast<T>(1)) * length - static_cast<T>(1)) / static_cast<T>(2);
                }
                return x;
            }


            template<typename T>
            __device__ T GsReflect(T fx, T x_min, T x_max) {
//  float fx = static_cast<float>(x);
                T dx = {};
                T range = x_max - x_min;
                if (fx < x_min) {
                    dx = x_min - fx;
                    int n = static_cast<int>(dx / range);
                    T r = dx - n * range;
                    if (n % 2 == 0) {
                        fx = x_min + r;
                    } else {
                        fx = x_max - r;
                    }
                } else if (fx > x_max) {
                    dx = fx - x_max;
                    int n = static_cast<int>(dx / range);
                    T r = dx - n * range;
                    if (n % 2 == 0) {
                        fx = x_max - r;
                    } else {
                        fx = x_min + r;
                    }
                }
                // else fallthrough
                return fx; //static_cast<T>(fx);
            }

            template<typename T, bool Layout>
            __device__ T PixelAtGrid(const T *input_data, int64_t *idx_input, const int64_t *dims_input,
                                     int64_t dim_size, int64_t padding_mode, T *border) {
                T pixel = 0.0f;

                using CH = Channels<Layout>;
                const int64_t spatial_size = dim_size - 2;
                int64_t *idx_input_spatial = &idx_input[CH::SPATIAL];
                const int64_t *dims_input_spatial = &dims_input[CH::SPATIAL];

                if (padding_mode == 0) {  // zeros
                    for (int i = 0; i < spatial_size; i++) {
                        if (idx_input_spatial[i] < 0 || idx_input_spatial[i] >= dims_input_spatial[i]) {
                            return pixel;
                        }
                    }
                } else if (padding_mode == 1) {  // border
                    for (int i = 0; i < spatial_size; i++) {
                        idx_input_spatial[i] = max((int64_t) 0, min((int64_t) dims_input_spatial[i] - 1,
                                                                    (int64_t) idx_input_spatial[i]));
                    }
                } else {  // Reflection
                    for (int i = 0; i < spatial_size; i++) {
                        idx_input_spatial[i] = (int64_t) GsReflect<T>(idx_input_spatial[i], border[2 * i],
                                                                      border[2 * i + 1]);
                    }
                }

                int64_t PixelOffset = indexNDTo1D(idx_input, dims_input, dim_size);
                return input_data[PixelOffset];
            }

            template<typename T>
            __device__ void GsGetCubicCoeffs(T x, T coeffs[4]) {
                float cubic_alpha = -0.75f;
                x = abs(x);
                coeffs[0] = (((cubic_alpha * (x + 1) - 5 * cubic_alpha) * (x + 1) + 8 * cubic_alpha) * (x + 1) -
                             4 * cubic_alpha);
                coeffs[1] = (((cubic_alpha + 2) * x - (cubic_alpha + 3)) * x * x + 1);
                coeffs[2] = (((cubic_alpha + 2) * (1 - x) - (cubic_alpha + 3)) * (1 - x) * (1 - x) + 1);
                coeffs[3] = (((cubic_alpha * (2 - x) - 5 * cubic_alpha) * (2 - x) + 8 * cubic_alpha) * (2 - x) -
                             4 * cubic_alpha);
            }

            template<typename T, int64_t spatial_size>
            __device__ T GsCubicInterpolate(T *cube, T *p) {
//                const int64_t dim_size = 4;
                int64_t dims_cube[spatial_size];
#pragma unroll
                for (int i = 0; i < spatial_size; i++) {
                    dims_cube[i] = 4;
                }

                T coeff_ND[spatial_size][4];
#pragma unroll
                for (int i = 0; i < spatial_size; i++) {
                    GsGetCubicCoeffs<T>(p[i], coeff_ND[i]);
                }

                T output = 0.f;
                int64_t cube_idx_count = 1 << (spatial_size * 2);
                int64_t idx_ND[spatial_size];
                for (int idx = 0; idx < cube_idx_count; idx++) {
                    index1DtoND(idx, dims_cube, spatial_size, idx_ND);
                    T weight = p[idx];
                    for (int j = 0; j < spatial_size; j++) {
                        weight *= coeff_ND[j][idx_ND[j]];
                    }
                    output += weight;
                }
                return output;
            }

            template<bool Layout, int64_t dim_size>
            __device__ int64_t outputIndexToGridIdx(
                    const int64_t idx,
                    const int64_t *dims_input,
                    const int64_t *dims_grid,
                    const int64_t *idx_output_ND
                    ) {
                using CH = Channels<Layout>;
                int spatial_size = dim_size - 2;

                //We need batch size from dims_input and spatial dims from dims_grid
                //E.g 4D calculation: int grid_idx = BIdx * H_out * W_out + yIdx * W_out + xIdx
                int64_t idx_output_excl_channel[dim_size - 1];
                idx_output_excl_channel[0] = idx_output_ND[CH::N];
                memcpy(idx_output_excl_channel + 1, idx_output_ND + CH::SPATIAL, spatial_size);

                return indexNDTo1D(idx_output_excl_channel, dims_grid, dim_size - 1);;
            }

            template<typename T, bool Layout, int64_t dim_size>
            __global__ void _GridSampleKernel(
                    const T *input_data,
                    const T *grid_data,
                    const int64_t mode,
                    const int64_t padding_mode,
                    const int64_t align_corners,
                    const TArray<int64_t, dim_size> dims_input_array,
                    const TArray<int64_t, dim_size> dims_grid_array,
                    T *output_data) {
                using CH = Channels<Layout>;
                const int64_t *dims_input = dims_input_array.Data();
                const int64_t *dims_grid = dims_grid_array.Data();
//                const int64_t dim_size = 4;
                const int64_t spatial_size = dim_size - 2;
//                const int64_t *dims_input_spatial = dims_input + CH::SPATIAL;

                //Check if thread idx is bigger than output vector size. If so, end early.
                int64_t total_thread_count = dims_input[CH::N] * dims_input[CH::C];
                for (size_t i = 1; i < dim_size - 1; i++) {
                    total_thread_count *= dims_grid[i];
                }
                CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, total_thread_count);


                // extract N-D index and dimensions for the output vector.
                int64_t dims_output[dim_size];
                int64_t idx_output_ND[dim_size];
                {
                    dims_output[CH::N] = dims_input[CH::N];
                    dims_output[CH::C] = dims_input[CH::C];
                    memcpy(dims_output + CH::SPATIAL, dims_grid, (spatial_size) * sizeof(int64_t));
                }
                index1DtoND(id, dims_output, dim_size, idx_output_ND);

                // Apply grid filter to current thread pixel
                T grid_imageSpace[spatial_size];
                {
                    int grid_idx = outputIndexToGridIdx<Layout, dim_size>(id, dims_input, dims_grid, idx_output_ND);
//                    const T *grid_vals;
                    //note that this ordering of spatial dimension is the reverse of the tensor dimension ordering, e.g. WHD vs DHW for 3D
                    const T *grid_vals = &grid_data[grid_idx * spatial_size];
#pragma unroll
                    for (int i = 0; i < spatial_size; i++) {
                        grid_imageSpace[i] = GsDenormalize(grid_vals[spatial_size - 1 - i], dims_input[CH::SPATIAL+i],
                                                           align_corners == 1);
                    }
                }

                //Preprocess for interpolation: Handle pixel mapped outside input spatial dimensions
                T border[spatial_size * 2];
                {
                    if (align_corners == 1) {
                        for (int i = 0; i < spatial_size; i++) {
                            border[i * 2] = 0;
                            border[i * 2 + 1] = dims_input[CH::SPATIAL+i] - 1;
                        }
                    } else {
                        for (int i = 0; i < spatial_size; i++) {
                            border[i * 2] = -0.5;
                            border[i * 2 + 1] = dims_input[CH::SPATIAL+i] - 0.5;
                        }
                    }
                    for (int i = 0; i < spatial_size; i++) {
                        if (grid_imageSpace[i] < border[2 * i] || grid_imageSpace[i] > border[2 * i + 1]) {
                            if (padding_mode == 1) {  // border
                                // Clamping must not be done here, see #10607
                            } else if (padding_mode == 2) {  // reflection
                                grid_imageSpace[i] = GsReflect(grid_imageSpace[i], border[2 * i], border[2 * i + 1]);
                            }
                        }
                    }
                }

                //Perform interpolation on the pixel
                int64_t idx_input_ND[dim_size];
                idx_input_ND[CH::N] = idx_output_ND[CH::N];
                idx_input_ND[CH::C] = idx_output_ND[CH::C];
                if (mode == 0) {  // linear
                    int64_t grid_imageSpace_roundup[spatial_size];
                    int64_t grid_imageSpace_rounddown[spatial_size];
                    for (int i = 0; i < spatial_size; i++) {
                        grid_imageSpace_rounddown[i] = floor(grid_imageSpace[i]);
                        grid_imageSpace_roundup[i] = grid_imageSpace_rounddown[i] + 1;
                    }

                    //need to calculate contribution for each integer pixel in the bounding box, there are 2^spatial pixels
                    T val_interpolated;
                    for (int i = 0; i < 1 << spatial_size; i++) {
                        T weight_multiplier = 1;
#pragma unroll
                        for (int j = 0; j < spatial_size; j++) {
                            if (i & (1 << j)) { //if jth dimension is 1
                                weight_multiplier *= (grid_imageSpace[j] - grid_imageSpace_rounddown[j]);
                                idx_input_ND[CH::SPATIAL + j] = grid_imageSpace_roundup[j];
                            } else {
                                weight_multiplier *= (grid_imageSpace_roundup[j] - grid_imageSpace[j]);
                                idx_input_ND[CH::SPATIAL + j] = grid_imageSpace_rounddown[j];
                            }
                        }
                        T pixel_val = PixelAtGrid<T, Layout>(input_data, idx_input_ND, dims_input, dim_size,
                                                             padding_mode, border);
                        val_interpolated += weight_multiplier * pixel_val;
                    }

                    output_data[id] = val_interpolated;
                    return;
                }
                if (mode == 1) {  // nearest
                    for (int i = 0; i < spatial_size; i++) {
                        idx_input_ND[CH::SPATIAL + i] = nearbyint(grid_imageSpace[i]);
                    }
                    output_data[id] =
                            PixelAtGrid<T, Layout>(input_data, idx_input_ND, dims_input, dim_size, padding_mode,
                                                   border);
                    return;
                }
                if (mode == 2) {  // cubic
                    int64_t p[spatial_size]; // top-left corner of the bbox
                    for (int i = 0; i < spatial_size; i++) {
                        p[i] = static_cast<int64_t>(std::floor(grid_imageSpace[i])) - 1;
                    }

                    const int64_t cube_idx_count = 1 << (spatial_size * 2);
                    T cube[cube_idx_count];
                    int64_t dims_cube[spatial_size];
                    for (int i = 0; i < spatial_size; i++) {
                        dims_cube[i] = 4;
                    }

                    for (int64_t idx = 0; idx < cube_idx_count; idx++) {
                        index1DtoND(idx, dims_cube, spatial_size,
                                    &idx_input_ND[CH::SPATIAL]); //p_cube stores the offset from top-left corner
                        for (int j = 0; j < spatial_size; j++) {
                            idx_input_ND[CH::SPATIAL + j] += p[j]; //apply corner idx to get cube idx;
                        }
                        cube[idx] =
                                PixelAtGrid<T, Layout>(input_data, idx_input_ND, dims_input, dim_size, padding_mode,
                                                       border);
                    }

                    T dp[spatial_size]; // top-left corner of the bbox
                    for (int i = 0; i < spatial_size; i++) {
                        dp[i] = grid_imageSpace[i] - p[i] - 1;
                    }
                    output_data[id] = GsCubicInterpolate<T, spatial_size>(cube, dp);
                    return;
                }
            }
            template<typename T, bool IsNHWC>
            void GridSampleImpl(
                    cudaStream_t stream,
                    const T *input_data,
                    const T *grid_data,
                    const int64_t mode,
                    const int64_t padding_mode,
                    const int64_t align_corners,
                    const gsl::span<const int64_t>& dims_input,
                    const gsl::span<const int64_t>& dims_grid,
//                    const int64_t dim_size1,
                    T *output_data) {
                using CH = Channels<IsNHWC>;
                const int64_t dim_size = dims_input.size();
                int64_t total_thread_count = dims_input[CH::N] * dims_input[CH::C];
                for (int64_t i = 1; i < dim_size - 1; i++) {
                    total_thread_count *= dims_grid[i];
                }

                int blocksPerGrid = static_cast<int>(
                        ceil(static_cast<T>(total_thread_count) / GridDim::maxThreadsPerBlock));

                switch (dim_size){
                    case 4:
                        _GridSampleKernel < T, IsNHWC, 4 ><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
                                input_data, grid_data, mode, padding_mode, align_corners,
                                dims_input, dims_grid, output_data);
                        break;

                    case 5:
                        _GridSampleKernel < T, IsNHWC, 5 ><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
                                        input_data, grid_data, mode, padding_mode, align_corners,
                                        dims_input, dims_grid, output_data);
                        break;
                }

            }

#define SPECIALIZED_IMPL(T, IsNHWC)                                                                                    \
  template void GridSampleImpl<T, IsNHWC>(cudaStream_t stream, const T* input_data, const T* grid_data,                \
                                          const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                          const gsl::span<const int64_t>& dims_input, const gsl::span<const int64_t>& dims_grid, T* output_data);

            SPECIALIZED_IMPL(float, false)  // NCHW
            SPECIALIZED_IMPL(float, true)   // NHWC


        }  // namespace cuda
    }  // namespace contrib
}  // namespace onnxruntime