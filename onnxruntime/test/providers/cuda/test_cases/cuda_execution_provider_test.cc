// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test is built only under DEBUG mode because it requires
// extra code in the core of CUDA EP and that code may
//  1. slow down performance critical applications and
//  2. increase binary size of ORT.

#include "gtest/gtest.h"
#include <iostream>

#include "core/framework/run_options.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime {
namespace cuda {
namespace test {
// TODO: Since the "DeferredRelease" has been migrated to CudaStream class,
// we should migrate this test from CudaEP unit test to CudaStream unit test.
TEST(TestDeferredRelease, WithArena) {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];

  onnxruntime::RunOptions run_opts;
  run_opts.run_tag = "log1";
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  AllocatorPtr cpu_pinned_alloc = ep.CreatePreferredAllocators()[1];
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_allocator->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr, info);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart(run_opts));
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = IAllocator::MakeUniquePtr<void>(cpu_pinned_alloc, n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  // Memory stats
  AllocatorStats stats;
  cpu_pinned_alloc->GetStats(&stats);
  ASSERT_EQ(stats.num_allocs, n_allocs);
  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true, run_opts));
}

TEST(TestDeferredRelease, WithoutArena) {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);

  onnxruntime::RunOptions run_opts;
  run_opts.run_tag = "log1";

  OrtDevice pinned_device{OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                          info.device_id};
  // Create allocator without BFCArena
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
      },
      pinned_device.Id(),
      false /* no arena */);
  auto cuda_pinned_alloc = CreateAllocator(pinned_memory_info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_allocator->Info().device, cuda_pinned_alloc, false, true, nullptr, nullptr, info);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart(run_opts));
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = IAllocator::MakeUniquePtr<void>(cuda_pinned_alloc, n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true, run_opts));
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
