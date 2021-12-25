/*
 * Copyright 2021 [TODO: Assign Copyright]
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

#ifndef PCMAP_IMPLEMENTATION_H_
#define PCMAP_IMPLEMENTATION_H_

#include <algorithm>
#include <functional>
#include <memory>

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::buildBulk(
    KeyT* KeysDevPtr,
    ValueT* ValuesDevPtr,
    uint32_t NumberOfKeys) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  build_table_kernel<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::
    buildBulkWithUniqueKeys(KeyT* KeysDevPtr,
                            ValueT* ValuesDevPtr,
                            uint32_t NumberOfKeys) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  build_table_with_unique_keys_kernel<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::searchIndividual(
    KeyT* KeysDevPtr,
    ValueT* ValuesDevPtr,
    uint32_t NumberOfQueries) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  search_table<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::searchBulk(
    KeyT* KeysDevPtr,
    ValueT* ValuesDevPtr,
    uint32_t NumberOfQueries) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  search_table_bulk<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::deleteIndividual(
    KeyT* KeysDevPtr,
    uint32_t NumberOfKeys) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  delete_table_keys<KeyT, ValueT>
      <<<NumberOfBlocks, BLOCKSIZE_>>>(KeysDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::batchedOperation(
    KeyT* KeysDevPtr,
    ValueT* ResultsDevPtr,
    uint32_t NumberOfOperations) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  /* TODO: Complete Implementation */
  __builtin_unreachable();
}

template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::countIndividual(
    KeyT* KeysDevPtr,
    uint32_t* CountDevPtr,
    uint32_t NumberOfQueries) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  count_key<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, CountDevPtr, NumberOfQueries, SlabHashCtxt);
}

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::updateBulk(
    KeyT* KeysDevPtr,
    ValueT* ValuesDevPtr,
    uint32_t NumberOfKeys) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  update_keys<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
void GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::upsertBulk(
    KeyT* KeysDevPtr,
    ValueT* ValuesDevPtr,
    uint32_t NumberOfKeys) {
  uint32_t NumberOfBlocks = (NumberOfKeys, BLOCKSIZE_ - 1) / BLOCKSIZE_;

  CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));
  upsert_keys<KeyT, ValueT><<<NumberOfBlocks, BLOCKSIZE_>>>(
      KeysDevPtr, ValuesDevPtr, NumberOfKeys, SlabHashCtxt);
}

template <typename KeyT, typename ValueT>
std::string GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::to_string() {
  std::string Result;

  /* TODO: Populate stringified fields */

  return Result;
}

template <typename KeyT, typename ValueT>
double GpuSlabHash<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::computeLoadFactor(
    int = 0) {
  auto FreeDeviceMem = [](const void* Ptr) -> void { CHECK_ERROR(cudaFree(Ptr)); }

  std::unique_ptr<uint32_t>
      BucketKeyCount{new uint32_t[NumberOfBuckets]};
  std::unique_ptr<uint32_t> BucketSlabCount{new uint32_t[NumberOfBuckets]};

  uint32_t* _BucketKeyCountDev;
  uint32_t* _BucketSlabCountDev;

  CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&_BucketKeyCountDev),
                         sizeof(uint32_t) * NumberOfBuckets));
  CHECK_ERROR(cudaMemset(_BucketKeyCountDev, 0, sizeof(uint32_t) * NumberOfBuckets));

  CHECK_ERROR(cudaMalloc(reinterpret_cast<void**>(&_BucketSlabCountDev),
                         sizeof(uint32_t) * NumberOfBuckets));
  CHECK_ERROR(cudaMemset(_BucketSlabCountDev, 0, sizeof(uint32_t) * NumberOfBuckets));

  std::unique_ptr<uint32_t, decltype(FreeDeviceMem)> BucketKeyCountDev{_BucketKeyCountDev,
                                                                       FreeDeviceMem};
  std::unique_ptr<uint32_t, decltype(FreeDeviceMem)> BucketSlabCountDev{
      _BucketSlabCountDev, FreeDeviceMem};

  uint32_t NumberOfThreadBlocks =
      (NumberOfBuckets * WarpWidth + BlockSize - 1) / BlockSize;
  bucket_count_kernel<KeyT, ValueT><<<NumberOfThreadBlocks, BlockSize>>>(
      SlabHashCtxt, BucketKeyCountDev.get(), BucketSlabCountDev.get(), NumberOfBuckets);

  CHECK_ERROR(cudaMemcpy(BucketKeyCount.get(),
                         BucketKeyCountDev.get(),
                         sizeof(uint32_t) * NumberOfBuckets,
                         cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaMemcpy(BucketSlabCount.get(),
                         BucketSlabCountDev.get(),
                         sizeof(uint32_t) * NumberOfBuckets,
                         cudaMemcpyDeviceToHost));

  int NumberOfElements =
      std::accumulate(BucketKeyCount.get(), BucketKeyCount.get() + NumberOfBuckets, 0);
  int NumberOfSlabs =
      std::accumulate(BucketSlabCount.get(), BucketSlabCount.get() + NumberOfBuckets, 0);

  double LoadFactor = static_cast<double>(NumberOfElements * sizeof(KeyT)) /
                      static_cast<double>(NumberOfSlabs * WARP_WIDTH_ * sizeof(uint32_t));

  return LoadFactor;
}

#endif  // PCMAP_IMPLEMENTATION_H_
