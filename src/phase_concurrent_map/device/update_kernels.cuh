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

#ifndef PCMAP_UPDATE_KERNELS_H_
#define PCMAP_UPDATE_KERNELS_H_

template <typename KeyT, typename ValueT, typename AllocPolicy, typename FilterMapTy>
__global__ void update_keys(
    KeyT* TheKeys,
    ValueT* TheValues,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
        SlabHashCtxt,
    FilterMapTy* FilterMaps = nullptr) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfKeys)
    return;

  KeyT TheKey{};
  ValueT TheValue{};
  uint32_t TheBucket = 0;
  bool ToUpdate = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheValue = TheValues[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToUpdate = true;
  }

  SlabHashCtxt.updatePair(
      ToUpdate,
      LaneID,
      TheKey,
      TheValue,
      TheBucket,
      (FilterMaps != nullptr) ? (FilterMaps + ThreadID) : nullptr);
}

template <typename KeyT, typename ValueT, typename AllocPolicy, typename FilterTy>
__global__ void upsert_keys(
    KeyT* TheKeys,
    ValueT* TheValues,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
        SlabHashCtxt,
    FilterTy* Filters = nullptr) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfKeys)
    return;

  typename AllocPolicy::AllocatorContextT TheAllocatorContext(
      SlabHashCtxt.getAllocatorContext());
  TheAllocatorContext.initAllocator(ThreadID, LaneID);

  KeyT TheKey{};
  ValueT TheValue{};
  uint32_t TheBucket = 0;
  bool ToUpsert = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheValue = TheValues[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToUpsert = true;
  }

  SlabHashCtxt.upsertPair(
      ToUpsert,
      LaneID,
      TheKey,
      TheValue,
      TheBucket,
      TheAllocatorContext,
      (Filters != nullptr) ? (Filters + ThreadID) : nullptr);
}

#endif  // PCMAP_UPDATE_KERNELS_H_
