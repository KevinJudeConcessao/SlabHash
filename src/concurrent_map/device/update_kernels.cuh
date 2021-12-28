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

#pragma once

template <typename KeyT,
          typename ValueT,
          typename AllocPolicy,
          typename FilterTy,
          typename MapTy>
__global__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    update_keys(
        KeyT* TheKeys,
        ValueT* TheValues,
        uint32_t NumberOfKeys,
        GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentMap>
            SlabHashCtxt) {
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
  bool ToUpdate = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheValue = TheValues[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToUpdate = true;
  }

  SlabHashCtxt.updatePair<FilterTy, MapTy>(
      ToUpdate, LaneID, TheKey, TheValue, TheBucket, TheAllocatorContext);
}

template <typename KeyT,
          typename ValueT,
          typename AllocPolicy,
          typename FilterTy,
          typename MapTy>
__global__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    upsert_keys(
        KeyT* TheKeys,
        ValueT* TheValues,
        uint32_t NumberOfKeys,
        GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentMap>
            SlabHashCtxt) {
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

  SlabHashCtxt.upsertPair<FilterTy, MapTy>(
      ToUpdate, LaneID, TheKey, TheValue, TheBucket, TheAllocatorContext);
}
