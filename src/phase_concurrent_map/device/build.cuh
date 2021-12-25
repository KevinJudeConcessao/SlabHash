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

#ifndef PCMAP_BUILD_H_
#define PCMAP_BUILD_H_

template <typename KeyT, typename ValueT>
__global__ void build_table_kernel(
    KeyT* TheKeys,
    ValueT* TheValues,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap> SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfKeys)
    return;

  AllocatorContextT TheAllocatorContext(SlabHashCtxt.getAllocatorContext());
  TheAllocatorContext.initAllocator(ThreadID, LaneID);

  KeyT TheKey{};
  ValueT TheValue{};
  uint32_t TheBucket = 0;
  bool ToInsert = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheValue = TheValues[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToInsert = true;
  }

  SlabHashCtxt.insertPair(
      ToInsert, LaneID, TheKey, TheValue, TheBucket, TheAllocatorContext);
}

template <typename KeyT, typename ValueT>
__global__ void build_table_with_unique_keys_kernel(
    KeyT* TheKeys,
    ValueT* TheValues,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap> SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfKeys)
    return;

  AllocatorContextT TheAllocatorContext(SlabHashCtxt.getAllocatorContext());
  TheAllocatorContext.initAllocator(ThreadID, LaneID);

  KeyT TheKey{};
  ValueT TheValue{};
  uint32_t TheBucket = 0;
  bool ToInsert = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheValue = TheValues[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToInsert = true;
  }

  SlabHashCtxt.insertPairUnique(
      ToInsert, LaneID, TheKey, TheValue, TheBucket, TheAllocatorContext);
}

#endif  // PCMAP_BUILD_H_
