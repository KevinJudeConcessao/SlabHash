/*
 * Copyright 2019 Saman Ashkiani
 * Copyright 2021 [TODO: Assign copyright]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <memory>

template <typename KeyT, typename ValueT, typename AllocPolicy>
void GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::buildBulk(
    KeyT* d_key,
    ValueT* d_value,
    uint32_t num_keys) {
  const uint32_t num_blocks = (num_keys + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  // calling the kernel for bulk build:
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
  cset::build_table_kernel<KeyT, AllocPolicy>
      <<<num_blocks, BLOCKSIZE_>>>(d_key, num_keys, gpu_context_);
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
void GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::
    searchIndividual(KeyT* d_query, ValueT* d_result, uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
  const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  cset::search_table<KeyT, AllocPolicy>
      <<<num_blocks, BLOCKSIZE_>>>(d_query, d_result, num_queries, gpu_context_);
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
void GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::
    deleteIndividual(KeyT* d_query, uint32_t num_queries) {
  CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
  const uint32_t num_blocks = (num_queries + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  cset::delete_table_keys<KeyT, AllocPolicy>
      <<<num_blocks, BLOCKSIZE_>>>(d_query, num_queries, gpu_context_);
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
std::string
GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::to_string() {
  std::string result;
  result += " ==== GpuSlabHash: \n";
  result += "\t Running on device \t\t " + std::to_string(device_idx_) + "\n";
  result += "\t SlabHashType:     \t\t " + gpu_context_.getSlabHashTypeName() + "\n";
  result += "\t Number of buckets:\t\t " + std::to_string(num_buckets_) + "\n";
  result += "\t d_table_ address: \t\t " +
            std::to_string(reinterpret_cast<uint64_t>(static_cast<void*>(d_table_))) +
            "\n";
  result += "\t hash function = \t\t (" + std::to_string(hf_.x) + ", " +
            std::to_string(hf_.y) + ")\n";
  return result;
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
double
GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::computeLoadFactor(
    int = 0) {
  auto FreeDeviceMem = [](void* Ptr) -> void { CHECK_ERROR(cudaFree(Ptr)); };

  std::unique_ptr<uint32_t> BucketKeyCount{new uint32_t[num_buckets_]};
  std::unique_ptr<uint32_t> BucketSlabCount{new uint32_t[num_buckets_]};

  uint32_t* _BucketKeyCountDev;
  uint32_t* _BucketSlabCountDev;

  CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&_BucketKeyCountDev),
                         sizeof(uint32_t) * num_buckets_));
  CHECK_ERROR(cudaMemset(_BucketKeyCountDev, 0, sizeof(uint32_t) * num_buckets_));

  CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&_BucketSlabCountDev),
                         sizeof(uint32_t) * num_buckets_));
  CHECK_ERROR(cudaMemset(_BucketSlabCountDev, 0, sizeof(uint32_t) * num_buckets_));

  std::unique_ptr<uint32_t, decltype(FreeDeviceMem)> BucketKeyCountDev{_BucketKeyCountDev,
                                                                       FreeDeviceMem};
  std::unique_ptr<uint32_t, decltype(FreeDeviceMem)> BucketSlabCountDev{
      _BucketSlabCountDev, FreeDeviceMem};

  uint32_t NumberOfThreadBlocks =
      (num_buckets_ * WARP_WIDTH_ + BLOCKSIZE_ - 1) / BLOCKSIZE_;
  bucket_count_kernel<KeyT, ValueT, AllocPolicy><<<NumberOfThreadBlocks, BLOCKSIZE_>>>(
      gpu_context_, BucketKeyCountDev.get(), BucketSlabCountDev.get(), num_buckets_);

  CHECK_ERROR(cudaMemcpy(BucketKeyCount.get(),
                         BucketKeyCountDev.get(),
                         sizeof(uint32_t) * num_buckets_,
                         cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaMemcpy(BucketSlabCount.get(),
                         BucketSlabCountDev.get(),
                         sizeof(uint32_t) * num_buckets_,
                         cudaMemcpyDeviceToHost));

  int NumberOfElements =
      std::accumulate(BucketKeyCount.get(), BucketKeyCount.get() + num_buckets_, 0);
  int NumberOfSlabs =
      std::accumulate(BucketSlabCount.get(), BucketSlabCount.get() + num_buckets_, 0);

  double LoadFactor = static_cast<double>(NumberOfElements * sizeof(KeyT)) /
                      static_cast<double>(NumberOfSlabs * WARP_WIDTH_ * sizeof(uint32_t));

  return LoadFactor;
}