// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ T WarpUniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = WARP_SHFL((unsigned)p.asInt, 0);
  return p.value;
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, half scale, uint8_t zp, const half* a, half* sums) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __ushort2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  // Extract 8 uint8_t values from the 64-bit input.
  uint8_t q[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    q[i] = (values_quant >> (i * 8)) & 0xFF;
  }

  // Convert pairs to half2 (0,4 1,5 2,6 3,7 interleaved order).
  half2 element04 = __halves2half2(__ushort2half_rn(q[0]), __ushort2half_rn(q[4]));
  half2 element15 = __halves2half2(__ushort2half_rn(q[1]), __ushort2half_rn(q[5]));
  half2 element26 = __halves2half2(__ushort2half_rn(q[2]), __ushort2half_rn(q[6]));
  half2 element37 = __halves2half2(__ushort2half_rn(q[3]), __ushort2half_rn(q[7]));

  half2 v0 = element04 * scale_half2 + zp_adjust2;
  half2 v1 = element15 * scale_half2 + zp_adjust2;
  half2 v2 = element26 * scale_half2 + zp_adjust2;
  half2 v3 = element37 * scale_half2 + zp_adjust2;

  half2* sums_half2 = reinterpret_cast<half2*>(sums);
  sums_half2[0] = sums_half2[0] + v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  sums_half2[1] = sums_half2[1] + v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  sums_half2[2] = sums_half2[2] + v2 * (*(reinterpret_cast<half2*>(&(vec_a.z))));
  sums_half2[3] = sums_half2[3] + v3 * (*(reinterpret_cast<half2*>(&(vec_a.w))));
}

__device__ __forceinline__ void AccumulateEightElements8b(uint64_t values_quant, float scale, uint8_t zp, const float* a, float* sums) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xFF) * scale + zp_adjust;
  float v1 = float((values_quant >> 8) & 0xFF) * scale + zp_adjust;
  float v2 = float((values_quant >> 16) & 0xFF) * scale + zp_adjust;
  float v3 = float((values_quant >> 24) & 0xFF) * scale + zp_adjust;
  float v4 = float((values_quant >> 32) & 0xFF) * scale + zp_adjust;
  float v5 = float((values_quant >> 40) & 0xFF) * scale + zp_adjust;
  float v6 = float((values_quant >> 48) & 0xFF) * scale + zp_adjust;
  float v7 = float((values_quant >> 56) & 0xFF) * scale + zp_adjust;

  sums[0] += v0 * a_vec_0.x;
  sums[1] += v1 * a_vec_0.y;
  sums[2] += v2 * a_vec_0.z;
  sums[3] += v3 * a_vec_0.w;
  sums[4] += v4 * a_vec_1.x;
  sums[5] += v5 * a_vec_1.y;
  sums[6] += v6 * a_vec_1.z;
  sums[7] += v7 * a_vec_1.w;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kElementsPerThreadPerIteration = 8;
constexpr int kWarpSize = GPU_WARP_SIZE;

// kernel for 8bits quantized gemv, i.e., computing A(1, K) x B(K, N)
// B(K, N) is quantized blockwise with 8bits and stored as [N, (K + block_size - 1)/block_size, block_size]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N / kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, block_size],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size, bool has_zero_point>
__global__ void __launch_bounds__(kWarpSize* kColsPerThreadBlock) MatMulFloat8bKernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  const int n_block_id = blockIdx.x;  // Each block handles kColsPerThreadBlock columns of N
  const int m_id = blockIdx.y;        // Should always be 0 based on TryMatMul8Bits check
  const int lane_id = threadIdx.x;
  const int n_id = n_block_id * kColsPerThreadBlock + threadIdx.y;

  constexpr bool print_debug = false;
  if constexpr (print_debug) {
    printf("MatMulFloat8bKernel: blockIdx.x=%d, blockIdx.y=%d, threadIdx.x=%d, threadIdx.y=%d, n_id=%d\n",
           blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, n_id);
  }

  // Ensure n_id does not go out of bounds if n is not perfectly divisible by kColsPerThreadBlock
  // Although TryMatMul8Bits checks for n % kColsPerThreadBlock == 0, it's safer.
  if (n_id >= n) return;

  constexpr int k_per_iter = kWarpSize * kElementsPerThreadPerIteration;  // 32 * 8 = 256

  extern __shared__ char shared_buffer[];
  // load scale to shared buffer
  T* b_scale_vec_shared = (T*)shared_buffer;
  // Calculate offset into global scales/zp data for the current N-column-block
  int scale_zp_offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  // Load scales for the columns processed by this thread block
  // Each thread loads its portion of the scales/zp for its assigned columns
  for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    // Check bounds for safety, although dimensions should ensure this is okay
    if (scale_zp_offset + i < n * blocks_per_K) {
      b_scale_vec_shared[i] = scales_data[scale_zp_offset + i];
    }
  }

  uint8_t* b_zp_vec_thread = nullptr;  // Pointer for the current thread's warp ZPs
  uint8_t* b_zp_vec_shared = nullptr;  // Pointer for the shared memory ZPs
  if constexpr (has_zero_point) {
    b_zp_vec_shared = reinterpret_cast<uint8_t*>(b_scale_vec_shared + kColsPerThreadBlock * blocks_per_K);
    for (int i = threadIdx.y * kWarpSize + threadIdx.x; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
      // Check bounds for safety
      if (scale_zp_offset + i < n * blocks_per_K) {
        b_zp_vec_shared[i] = zero_points[scale_zp_offset + i];
      }
    }
    // Point to the start of the zero points for the warp's assigned column
    b_zp_vec_thread = b_zp_vec_shared + threadIdx.y * blocks_per_K;
  }
  __syncthreads();  // Ensure scales and ZPs are loaded

  if constexpr (print_debug) {
    for (int i = 0; i < kColsPerThreadBlock * blocks_per_K; ++i) {
      if (scale_zp_offset + i < n * blocks_per_K) {
        printf("MatMulFloat8bKernel: b_scale_vec_shared[%d <= %d] = %.2f\n", i, scale_zp_offset + i, static_cast<float>(b_scale_vec_shared[i]));
      }
    }
    if constexpr (has_zero_point) {
      for (int i = 0; i < kColsPerThreadBlock * blocks_per_K; ++i) {
        if (scale_zp_offset + i < n * blocks_per_K) {
          printf("MatMulFloat8bKernel: b_zp_vec_shared[%d <= %d] = %d\n", i, scale_zp_offset + i, static_cast<int>(b_zp_vec_shared[i]));
        }
      }
    }
  }

  // Point a_data to the start of the row (m_id is 0) and offset by lane's portion
  a_data += (lane_id << 3);  // lane_id * 8 elements offset

  // Point to the start of the scales for the warp's assigned column
  const T* b_scale_vec_thread = b_scale_vec_shared + threadIdx.y * blocks_per_K;

  T sums[kElementsPerThreadPerIteration] = {static_cast<T>(0.0f)};  // Initialize sums to zero

  int k_id = 0;
  // Adjust b_data_quant pointer for the specific N column and the thread's lane
  // Layout: [N, blocks_per_K, block_size]
  const uint8_t* b_data_quant_thread = b_data_quant + (n_id * blocks_per_K * block_size) + (lane_id * sizeof(uint64_t));

  constexpr bool use_unroll = true;  // Unroll the loop for better performance

  if constexpr (use_unroll) {
    // Calculate the block index within the K dimension for the start of this thread's work
    // Each thread handles 8 elements, corresponding to lane_id * 8
    // t_meta_k is the index into the scale/zp arrays (which are per block_size)
    int t_meta_k = (lane_id * kElementsPerThreadPerIteration) / block_size;

#define UnRollReduction(unroll_size)                                                                                    \
  do {                                                                                                                  \
    constexpr int kUnroll = unroll_size;                                                                                \
    constexpr int kElementsPerUnrollIter = k_per_iter * kUnroll; /* Elements processed per outer loop iter */           \
    for (; k_id + kElementsPerUnrollIter <= k; k_id += kElementsPerUnrollIter) {                                        \
      _Pragma("unroll") for (int i = 0; i < kUnroll; ++i) {                                                             \
        /* Calculate pointer for this inner iteration's quantized data */                                               \
        /* Base pointer `b_data_quant_thread` is already offset by n_id and lane_id */                                  \
        /* Offset further by k_id progression and inner loop step 'i' */                                                \
        /* Each full warp iteration consumes k_per_iter = 256 elements = 256 bytes */                                   \
        const uint8_t* current_b_ptr = b_data_quant_thread + k_id * sizeof(uint8_t) + i * k_per_iter * sizeof(uint8_t); \
                                                                                                                        \
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);                                       \
        uint32_t val_low = ptr32[0];                                                                                    \
        uint32_t val_high = ptr32[1];                                                                                   \
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;                                             \
                                                                                                                        \
        /* Calculate index into scale/zp for this inner iteration */                                                    \
        int current_meta_k = t_meta_k + (k_id / block_size) + i * (k_per_iter / block_size);                            \
        T scale = b_scale_vec_thread[current_meta_k];                                                                   \
        uint8_t zp = 0; /* Default ZP if not used */                                                                    \
        if constexpr (has_zero_point) {                                                                                 \
          zp = b_zp_vec_thread[current_meta_k];                                                                         \
        }                                                                                                               \
        AccumulateEightElements8b(value, scale, zp, a_data + k_id + i * k_per_iter, sums);                              \
      }                                                                                                                 \
      /* Advance base pointers/indices for next outer loop iteration */                                                 \
      /* b_data_quant_thread does not need global advance here, handled by k_id */                                      \
      /* t_meta_k does not need global advance here, handled by current_meta_k calc */                                  \
    }                                                                                                                   \
  } while (false)

    UnRollReduction(16);
    UnRollReduction(4);
    UnRollReduction(1);

#undef UnRollReduction
  } else {
    // Simpler loop structure (replace UnRollReduction)
    for (; k_id + k_per_iter <= k; k_id += k_per_iter) {
      const uint8_t* current_b_ptr = b_data_quant_thread + k_id;  // Offset by k_id bytes (since each thread reads 8 bytes = 8 elements)

      const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
      uint32_t val_low = ptr32[0];
      uint32_t val_high = ptr32[1];
      uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

      int current_meta_k = (lane_id * kElementsPerThreadPerIteration + k_id) / block_size;  // Recalculate meta index based on absolute K pos
      T scale = b_scale_vec_thread[current_meta_k];
      uint8_t zp = 128;
      if constexpr (has_zero_point) {
        zp = b_zp_vec_thread[current_meta_k];
      }

      if constexpr (print_debug) {
        const uint8_t* q = reinterpret_cast<const uint8_t*>(&value);
        printf("MatMulFloat8bKernel: k_id=%d, current_meta_k=%d, scale=%.2f, zp=%d, %d, %d, %d, %d, %d, %d, %d, %d\n",
               k_id, current_meta_k, static_cast<float>(scale), static_cast<int>(zp),
               q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
      }

      AccumulateEightElements8b(value, scale, zp, a_data + k_id, sums);
    }
  }

  if constexpr (print_debug) {
    for (int i = 0; i < 8; ++i) {
      printf("MatMulFloat8bKernel: sums[%d] = %.2f\n", i, static_cast<float>(sums[i]));
    }
  }

  // Handle remainder elements for K (processing one k_per_iter block at a time was too coarse)
  // We need to handle the remainder elements *within* the last block_size segment if K is not a multiple of block_size
  // And also handle if K is not a multiple of k_per_iter.
  // Each thread `lane_id` is responsible for elements `k` such that `k % kWarpSize == lane_id`.
  // More accurately, thread `lane_id` handles `a_data` indices `lane_id*8 + 0..7`, `lane_id*8 + k_per_iter + 0..7`, etc.

  // b_data_quant layout: [N, blocks_per_K, block_size]
  const uint8_t* b_base_ptr_n = b_data_quant + n_id * blocks_per_K * block_size;

  k_id = 0;  // Reset k_id for clarity
  for (int block_k_idx = 0; block_k_idx < blocks_per_K; ++block_k_idx) {
    int k_start_block = block_k_idx * block_size;
    int k_end_block = k_start_block + block_size;

    // Get scale/zp for this block
    T scale = b_scale_vec_thread[block_k_idx];
    uint8_t zp = 128;
    if constexpr (has_zero_point) {
      zp = b_zp_vec_thread[block_k_idx];
    }

    // Iterate within the block, assigning 8 elements per thread per step
    // Each thread `lane_id` handles elements k_start_block + lane_id*8, k_start_block + lane_id*8 + 1, ..., k_start_block + lane_id*8 + 7
    // We only proceed if these elements are within the valid K range and within the current block.
    int current_k_base = k_start_block + lane_id * kElementsPerThreadPerIteration;  // Base K index for this thread in this block

    if (current_k_base < k_end_block && current_k_base < k) {
      int elements_to_process = min(kElementsPerThreadPerIteration, k - current_k_base);
      elements_to_process = min(elements_to_process, k_end_block - current_k_base);

      // Check if we have a full 8 elements (uint64_t) to load
      if (elements_to_process == kElementsPerThreadPerIteration) {
        const uint8_t* current_b_ptr = b_base_ptr_n + current_k_base;

        // Load uint64_t safely ---`
        const uint32_t* ptr32 = reinterpret_cast<const uint32_t*>(current_b_ptr);
        uint32_t val_low = ptr32[0];
        uint32_t val_high = ptr32[1];
        uint64_t value = (static_cast<uint64_t>(val_high) << 32) | val_low;

        if constexpr (print_debug) {
          const uint8_t* q = reinterpret_cast<const uint8_t*>(&value);
          printf("MatMulFloat8bKernel: block_k_idx=%d, current_k_base=%d, scale=%.2f, zp=%d, %d, %d, %d, %d, %d, %d, %d, %d\n",
                 block_k_idx, current_k_base, static_cast<float>(scale), static_cast<int>(zp),
                 q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
        }

        AccumulateEightElements8b(value, scale, zp, a_data + current_k_base - (lane_id << 3), sums);  // Adjust a_data pointer back to thread's base

      } else {
        // Handle partial tail elements (less than 8) is not needed since TryMatMul8Bits ensures k is a multiple of 8.
      }
    }
  }

  if constexpr (print_debug) {
    for (int i = 0; i < 8; ++i) {
      printf("MatMulFloat8bKernel: sums[%d] = %.2f\n", i, static_cast<float>(sums[i]));
    }
  }

  constexpr bool use_warp_reduction = false;
  if constexpr (use_warp_reduction) {
    float sum = (float)(sums[0] + sums[1] + sums[2] + sums[3] + sums[4] + sums[5] + sums[6] + sums[7]);
    // warp reduction
    for (int i = kWarpSize / 2; i > 0; i = i / 2) {
      sum += WARP_SHFL_DOWN(sum, i);
    }

    if (lane_id == 0) {
      output[m_id * n + n_id] = sum;
    }
  } else {
    // The reduction needs to sum the 8 partial sums within each thread first.
    T total_sum_thread = static_cast<T>(0.0f);
#pragma unroll
    for (int i = 0; i < kElementsPerThreadPerIteration; ++i) {
      total_sum_thread += sums[i];
    }

    // Use CUB for efficient warp reduction
    using BlockReduce = cub::WarpReduce<T>;
    __shared__ typename BlockReduce::TempStorage temp_storage[kColsPerThreadBlock];
    total_sum_thread = BlockReduce(temp_storage[threadIdx.y]).Sum(total_sum_thread);

    if (lane_id == 0) {
      output[m_id * n + n_id] = total_sum_thread;
    }
  }
}

template <class T>
bool TryMatMul8Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  printf("TryMatMul8Bits: m=%d, n=%d, k=%d, block_size=%d, shared_mem_per_block=%d\n", m, n, k, block_size, shared_mem_per_block);

  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    printf("TryMatMul8Bits: n %% kColsPerThreadBlock != 0 or k %% 8 != 0 or m > 1\n");
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(GPU_WARP_SIZE_HOST, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int shared_mem_size = (sizeof(T) + (zero_points != nullptr ? sizeof(uint8_t) : 0)) * blocks_per_K * kColsPerThreadBlock;
  if (shared_mem_size > shared_mem_per_block) {
    printf("TryMatMul8Bits: shared_mem_size %d > shared_mem_per_block %d\n", shared_mem_size, shared_mem_per_block);
    return false;
  }

#define MatMulFloat8bKernelDispatch(block_size)                                              \
  if (nullptr != zero_points) {                                                              \
    MatMulFloat8bKernel<T, block_size, true><<<blocks, threads, shared_mem_size, stream>>>(  \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);      \
  } else {                                                                                   \
    MatMulFloat8bKernel<T, block_size, false><<<blocks, threads, shared_mem_size, stream>>>( \
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);      \
  }

  if (16 == block_size) {
    MatMulFloat8bKernelDispatch(16);
  } else if (32 == block_size) {
    MatMulFloat8bKernelDispatch(32);
  } else if (64 == block_size) {
    MatMulFloat8bKernelDispatch(64);
  } else if (128 == block_size) {
    MatMulFloat8bKernelDispatch(128);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

#undef MatMulFloat8bKernelDispatch

  return true;
}

template bool TryMatMul8Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul8Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
