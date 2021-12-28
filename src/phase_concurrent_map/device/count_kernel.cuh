/*
 * Copyright 2018 Saman Ashkiani
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

#ifndef PCMAP_COUNT_KERNEL_H_
#define PCMAP_COUNT_KERNEL_H_

template <typename KeyT, typename ValueT, typename AllocPolicy>
__global__ void count_key(
    KeyT* TheKeys,
    uint32_t* KeyCounts,
    uint32_t NumberOfQueries,
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
        SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfQueries)
    return;

  KeyT TheKey{};
  uint32_t TheCount = 0;
  uint32_t TheBucket = 0;
  bool ToCount = false;

  if (ThreadID < NumberOfQueries) {
    TheKey = TheKeys[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToCount = true;
  }

  SlabHashCtxt.countKey(ToCount, LaneID, TheKey, TheCount, TheBucket);

  if (ThreadID < NumberOfQueries)
    KeyCounts[ThreadID] = TheCount;
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
__global__ void bucket_count_kernel(
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
        SlabHashCtxt,
    uint32_t* KeysCount,
    uint32_t* SlabsCount,
    uint32_t NumberOfBuckets) {
  /* Assign one warp for each bucket */

  uint32_t GlobalThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t BlockWarpID = GlobalThreadID >> 5;  // Also the BucketID
  uint32_t LaneID = threadIdx.x & 0x1F;

  using SlabHashTy = PhaseConcurrentMapT<KeyT, ValueT>;

  if (BlockWarpID >= NumberOfBuckets)
    return;

  uint32_t KeyCount = 0;
  uint32_t SlabCount = 1;
  uint32_t CurrentSlabPtr = SlabHashTy::A_INDEX_POINTER;

  uint32_t LaneKey = *getPointerFromBucket(BlockWarpID, LaneID);
  KeyCount = __popc(__ballot_sync(0xFFFFFFFF, LaneKey != EMPTY_KEY) &
                    SlabHashTy::REGULAR_NODE_KEY_MASK);
  CurrentSlabPtr = __shfl_sync(0xFFFFFFFF, LaneKey, SlabHashTy::NEXT_PTR_LANE, 32);

  while (CurrentSlabPtr != SlabHashTy::EMPTY_INDEX_POINTER) {
    LaneKey = *getPointerFromSlab(CurrentSlabPtr, LaneID);

    KeyCount += __popc(__ballot_sync(0xFFFFFFFF, LaneKey != EMPTY_KEY) &
                       SlabHashTy::REGULAR_NODE_KEY_MASK);
    SlabCount += 1;

    CurrentSlabPtr = __shfl_sync(0xFFFFFFFF, LaneKey, SlabHashTy::NEXT_PTR_LANE, 32);
  }

  if (LaneID == 0) {
    KeyCounts[BlockWarpID] = KeyCount;
    SlabsCount[BlockWarpID] = SlabCount;
  }
}

#endif  // PCMAP_COUNT_KERNEL_H_
