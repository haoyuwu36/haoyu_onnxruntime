// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "grid_sample_impl.h"
#include <stdio.h>

#define DEBUG_SIZE 1// && id > 2

using namespace onnxruntime::cuda;

namespace onnxruntime {
    namespace contrib {
        namespace cuda {
            template <typename T>
            __device__ __inline__ int64_t roundDown2ll(T value);
//            {
//                static_assert(false, "not implemented");
//            };

            template <>
            __device__ __inline__ int64_t roundDown2ll<float>(float value) {
                return std::floor(value);
            }

            template <>
            __device__ __inline__ int64_t roundDown2ll<double>(double value) {
                return std::floor(value);
            }

            template <>
            __device__ __inline__ int64_t roundDown2ll<int>(int value) {
                return std::floor(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t roundDown2ll<__half>(__half value) {
                return __half2ll_rd(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t roundDown2ll<__nv_bfloat16>(__nv_bfloat16 value) {
                return __bfloat162ll_rd(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t roundDown2ll<BFloat16>(BFloat16 value) {
                return __bfloat162ll_rd(static_cast<__nv_bfloat16>(value));  // floor of an int is itself
            }

            template <typename T>
            __device__ __inline__ int64_t near2ll(T value);
//            {
//                static_assert(false, "not implemented");
//            };

            template <>
            __device__ __inline__ int64_t near2ll<float>(float value) {
                return std::nearbyint(value);
            }

            template <>
            __device__ __inline__ int64_t near2ll<double>(double value) {
                return std::nearbyint(value);
            }

            template <>
            __device__ __inline__ int64_t near2ll<int>(int value) {
                return std::nearbyint(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t near2ll<__half>(__half value) {
                return __half2ll_rn(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t near2ll<__nv_bfloat16>(__nv_bfloat16 value) {
                return __bfloat162ll_rn(value);  // floor of an int is itself
            }

            template <>
            __device__ __inline__ int64_t near2ll<BFloat16>(BFloat16 value) {
                return __bfloat162ll_rn(static_cast<__nv_bfloat16>(value));  // floor of an int is itself
            }



            template<typename T>
            __device__ T GsDenormalize(T n, int64_t length, bool align_corners) {
                T x = {};
                if (align_corners) {  // align_corners: true => [-1, 1] to [0, length - 1]
                    x = (n + static_cast<T>(1.f)) / static_cast<T>(2.f) * static_cast<T>(length - 1);
                } else {  // align_corners: false => [-1, 1] to [-0.5, length - 0.5]
                    x = ((n + static_cast<T>(1.f)) * static_cast<T>(length) - static_cast<T>(1.f)) / static_cast<T>(2.f);
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
                    T r = dx - static_cast<T>(n) * range;
                    if (n % 2 == 0) {
                        fx = x_min + r;
                    } else {
                        fx = x_max - r;
                    }
                } else if (fx > x_max) {
                    dx = fx - x_max;
                    int n = static_cast<int>(dx / range);
                    T r = dx - static_cast<T>(n) * range;
                    if (n % 2 == 0) {
                        fx = x_max - r;
                    } else {
                        fx = x_min + r;
                    }
                }
                // else fallthrough
                return fx; //static_cast<T>(fx);
            }

            template<typename T, bool Layout, int64_t dim_size>
            __device__ T PixelAtGrid(const T *input_data, int64_t *idx_input, const int64_t *dims_input,
                                      int64_t padding_mode, T *border) {
                T pixel = static_cast<T>(0.f);

                using CH = Channels<Layout, static_cast<int>(dim_size)>;
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
            __device__ void GsGetCubicCoeffs(T c, T coeffs[4]) {
                float cubic_alpha = -0.75f;
                float x = abs(static_cast<float>(c));
                coeffs[0] = static_cast<float>((((cubic_alpha * (x + 1.f) - 5.f * cubic_alpha) * (x + 1.f) + 8.f * cubic_alpha) * (x + 1.f) -
                             4.f * cubic_alpha));
                coeffs[1] = static_cast<float>((((cubic_alpha + 2.f) * x - (cubic_alpha + 3.f)) * x * x + 1.f));
                coeffs[2] = static_cast<float>((((cubic_alpha + 2.f) * (1.f - x) - (cubic_alpha + 3.f)) * (1.f - x) * (1.f - x) + 1.f));
                coeffs[3] = static_cast<float>((((cubic_alpha * (2.f - x) - 5.f * cubic_alpha) * (2.f - x) + 8.f * cubic_alpha) * (2.f - x) -
                             4.f * cubic_alpha));
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
                    if (threadIdx.x < DEBUG_SIZE){
                        //printf("p[%d]: %f\n", i, p[i]);
                        //printf("coeff_ND[%d][0]: %f\n", i, coeff_ND[i][0]);
                        //printf("coeff_ND[%d][1]: %f\n", i, coeff_ND[i][1]);
                        //printf("coeff_ND[%d][2]: %f\n", i, coeff_ND[i][2]);
                        //printf("coeff_ND[%d][3]: %f\n", i, coeff_ND[i][3]);
                    }
                }

                if (threadIdx.x < DEBUG_SIZE) {
                    //printf("cube[0]: %f\n", cube[0]);
                }
                T output = 0.f;
                int64_t cube_idx_count = 1 << (spatial_size * 2);
                int64_t idx_ND[spatial_size];
                for (int idx = 0; idx < cube_idx_count; idx++) {
                    index1DtoND(idx, dims_cube, spatial_size, idx_ND);
                    T weight = cube[idx];
                    for (int j = 0; j < spatial_size; j++) {
                        weight *= coeff_ND[j][idx_ND[j]];
                        if (threadIdx.x < DEBUG_SIZE){
                            //printf("weight: %f\n", weight);
                            //printf("output: %f\n", output);
                        }
                    }
                    output += weight;

                }
                return output;
            }

            template<bool Layout, int64_t dim_size>
            __device__ int64_t outputIndexToGridIdx(
                    const int64_t id,
                    const int64_t *dims_input,
                    const int64_t *dims_grid,
                    const int64_t *idx_output_ND
                    ) {
                using CH = Channels<Layout, static_cast<int>(dim_size)>;
                int spatial_size = dim_size - 2;

                //We need batch size from dims_input and spatial dims from dims_grid
                //E.g 4D calculation: int grid_idx = BIdx * H_out * W_out + yIdx * W_out + xIdx
                int64_t idx_output_excl_channel[dim_size - 1];
                idx_output_excl_channel[0] = idx_output_ND[CH::N];
                memcpy(idx_output_excl_channel + 1, idx_output_ND + CH::SPATIAL, sizeof(int64_t)*(spatial_size));

                if (id < DEBUG_SIZE) {
                    for (int i=0; i<dim_size-1; i++){
                        //printf("id: %ld, idx_output_excl[%d]: %ld\n", id, i, idx_output_excl_channel[i]);
                    }
                }
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
                using CH = Channels<Layout, static_cast<int>(dim_size)>;
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
                    // dims_grid has spatial dimension started at index 1
                    memcpy(dims_output + CH::SPATIAL, &dims_grid[1], (spatial_size) * sizeof(int64_t));
                }
                index1DtoND(id, dims_output, dim_size, idx_output_ND);
                if (id < DEBUG_SIZE){
                    //printf("id: %d, Layout: %s\n", id, Layout ? "NHWC" : "NCHW");
                    for (int i=0; i<dim_size; i++){
                        //printf("dims_output[%d]: %ld\n", i, dims_output[i]);
                    }
                    for (int i=0; i<dim_size; i++){
                        //printf("idx_output_ND[%d]: %ld\n", i, idx_output_ND[i]);
                    }
                }
                // Apply grid filter to current thread pixel
                T grid_imageSpace[spatial_size];
                {
                    int grid_idx = outputIndexToGridIdx<Layout, dim_size>(id, dims_input, dims_grid, idx_output_ND);
//                    const T *grid_vals;
                    //note that this ordering of spatial dimension is the reverse of the tensor dimension ordering, e.g. WHD vs DHW for 3D
                    const T *grid_vals = &grid_data[grid_idx * spatial_size];
#pragma unroll
                    for (int i = 0; i < spatial_size; i++) {
                        grid_imageSpace[i] = GsDenormalize(grid_vals[spatial_size -1 -i], dims_input[CH::SPATIAL + i],
                                                           align_corners == 1);
                    }

                    if (id < DEBUG_SIZE){
                        //printf("grid_idx: %d\n", grid_idx);
                        //printf("grid_vals: %f, %f", grid_vals[0], grid_vals[1]);
                        for (int i=0; i<spatial_size; i++){
                            //printf("idx: %d, grid_imageSpace[%d]: %f\n", id, i, grid_imageSpace[i]);
                        }
                    }
                }


                //Preprocess for interpolation: Handle pixel mapped outside input spatial dimensions
                T border[spatial_size * 2];
                {
                    if (align_corners == 1) {
                        for (int i = 0; i < spatial_size; i++) {
                            border[i * 2] = static_cast<T>(0.f);
                            border[i * 2 + 1] = static_cast<T>(dims_input[CH::SPATIAL+i]) - static_cast<T>(1.f);
                        }
                    } else {
                        for (int i = 0; i < spatial_size; i++) {
                            border[i * 2] = static_cast<T>(-0.5f);
                            border[i * 2 + 1] = static_cast<T>(dims_input[CH::SPATIAL+i]) - static_cast<T>(0.5f);
                        }
                    }
                    for (int i = 0; i < spatial_size; i++) {
                        if (grid_imageSpace[i] < border[2 * i] || grid_imageSpace[i] > border[2 * i + 1]) {
                            if (padding_mode == 1) {  // border
                                // Clamping must not be done here, see #10607
                            } else if (padding_mode == 2) {  // reflection
                                grid_imageSpace[i] = GsReflect<T>(grid_imageSpace[i], border[2 * i], border[2 * i + 1]);
                            }
                        }
                    }
                }
                if (id < DEBUG_SIZE){
                    for (int i=0; i<spatial_size*2; i++){
                        //printf("idx: %d, after padding border[%d]: %f\n", id, i, border[i]);
                    }
                    for (int i=0; i<spatial_size; i++){
                        //printf("idx: %d, after padding grid_imageSpace[%d]: %f\n", id, i, grid_imageSpace[i]);
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
                        grid_imageSpace_rounddown[i] = roundDown2ll<T>(grid_imageSpace[i]);
                        grid_imageSpace_roundup[i] = grid_imageSpace_rounddown[i] + 1;
                    }

                    //need to calculate contribution for each integer pixel in the bounding box, there are 2^spatial pixels
                    T val_interpolated;
                    for (int i = 0; i < 1 << spatial_size; i++) {
                        T weight_multiplier = static_cast<T>(1.f);
#pragma unroll
                        for (int j = 0; j < spatial_size; j++) {
                            if (i & (1 << j)) { //if jth dimension is 1
                                weight_multiplier *= (grid_imageSpace[j] - static_cast<T>(grid_imageSpace_rounddown[j]));
                                idx_input_ND[CH::SPATIAL + j] = grid_imageSpace_roundup[j];
                            } else {
                                weight_multiplier *= (static_cast<T>(grid_imageSpace_roundup[j]) - grid_imageSpace[j]);
                                idx_input_ND[CH::SPATIAL + j] = grid_imageSpace_rounddown[j];
                            }
                        }
                        T pixel_val = PixelAtGrid<T, Layout, dim_size>(input_data, idx_input_ND, dims_input,
                                                             padding_mode, border);
                        val_interpolated += weight_multiplier * pixel_val;
                    }

                    output_data[id] = val_interpolated;
                    return;
                }
                if (mode == 1) {  // nearest
                    for (int i = 0; i < spatial_size; i++) {
                        idx_input_ND[CH::SPATIAL + i] = near2ll<T>(grid_imageSpace[i]);
                    }
                    output_data[id] =
                            PixelAtGrid<T, Layout, dim_size>(input_data, idx_input_ND, dims_input, padding_mode,
                                                   border);
//                    if (id < DEBUG_SIZE){
//                        for (int i = 0; i < spatial_size; i++) {
//                            //printf("idx: %d, idx_input_ND[%d]: %ld\n", id, i, idx_input_ND[CH::SPATIAL + i]);
//                        }
//                        //printf("idx: %d, output: %f\n", id, output_data[id]);
//                    }
                    return;
                }
                if (mode == 2) {  // cubic
                    int64_t p[spatial_size]; // top-left corner of the bbox
                    for (int i = 0; i < spatial_size; i++) {
                        p[i] = static_cast<int64_t>(roundDown2ll<T>(grid_imageSpace[i])) - 1;
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
                                PixelAtGrid<T, Layout, dim_size>(input_data, idx_input_ND, dims_input, padding_mode,
                                                       border);
                        if (id < DEBUG_SIZE){
                            //printf("cube[%ld]: %f\n", idx, cube[idx]);
                        }
                    }

                    T dp[spatial_size]; // top-left corner of the bbox
                    for (int i = 0; i < spatial_size; i++) {
                        dp[i] = grid_imageSpace[i] - static_cast<T>(p[i] + 1.f);
                    }
                    output_data[id] = GsCubicInterpolate<T, spatial_size>(cube, dp);
                    if (id < DEBUG_SIZE){
                        for (int i=0; i<spatial_size; i++){
                            //printf("p[i]: %ld\n", p[i]);
                        }
                        for (int i=0; i<spatial_size; i++){
                            //printf("cube[i]: %f\n", cube[i]);
                        }
                        for (int i=0; i<spatial_size; i++){
                            //printf("dp[i]: %f\n", dp[i]);
                        }
                        //printf("output_data[id]: %f", output_data[id]);
                    }
                    return;
                }
            }

            // TODO change to lambda function after C++20?
            template<bool IsNHWC, int64_t dim_size>
            int64_t getBlocksCount(
                    const gsl::span<const int64_t>& dims_input,
                    const gsl::span<const int64_t>& dims_grid
                    ) {
                using CH = Channels<IsNHWC, static_cast<int>(dim_size)>;
                int64_t total_thread_count = dims_input[CH::N] * dims_input[CH::C];
                for (int64_t i = 1; i < dim_size - 1; i++) {
                    total_thread_count *= dims_grid[i];
                }
                int64_t blocksPerGrid = CeilDiv(total_thread_count, GridDim::maxThreadsPerBlock);
                return blocksPerGrid;
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
                    T *output_data) {
                const int dim_size = dims_input.size();
                typedef typename ToCudaType<T>::MappedType CudaT;
                const CudaT* input_data_cuda = reinterpret_cast<const CudaT*>(input_data);
                const CudaT* grid_data_cuda = reinterpret_cast<const CudaT*>(grid_data);
                CudaT* output_data_cuda = reinterpret_cast<CudaT*>(output_data);

                int64_t blocksPerGrid;
                switch (dim_size){
                    case 4:
                        blocksPerGrid = getBlocksCount<IsNHWC, 4>(dims_input, dims_grid);
                        _GridSampleKernel < CudaT, IsNHWC, 4 ><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
                                input_data_cuda, grid_data_cuda, mode, padding_mode, align_corners,
                                dims_input, dims_grid, output_data_cuda);
                        break;

                    case 5:
                        blocksPerGrid = getBlocksCount<IsNHWC, 5>(dims_input, dims_grid);
                        _GridSampleKernel < CudaT, IsNHWC, 5 ><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
                                input_data_cuda, grid_data_cuda, mode, padding_mode, align_corners,
                                dims_input, dims_grid, output_data_cuda);
                        break;
                }
            }

#define SPECIALIZED_IMPL(T, IsNHWC)                                                                                    \
  template void GridSampleImpl<T, IsNHWC>(cudaStream_t stream, const T* input_data, const T* grid_data,                \
                                          const int64_t mode, const int64_t padding_mode, const int64_t align_corners, \
                                          const gsl::span<const int64_t>& dims_input, const gsl::span<const int64_t>& dims_grid, T* output_data);

            SPECIALIZED_IMPL(float, false)  // NCHW
            SPECIALIZED_IMPL(float, true)   // NHWC
            SPECIALIZED_IMPL(MLFloat16, false)
            SPECIALIZED_IMPL(MLFloat16, true)
            SPECIALIZED_IMPL(BFloat16, false)
            SPECIALIZED_IMPL(BFloat16, true)
        }  // namespace cuda
    }  // namespace contrib
}  // namespace onnxruntime