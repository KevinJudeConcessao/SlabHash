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

#ifndef PCMAP_HASH_CTXT_COUNT_H_
#define PCMAP_HASH_CTXT_COUNT_H_

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>::
    countKey(bool& ToBeSearched,
             const uint32_t& LaneID,
             const KeyT& TheKey,
             uint32_t& TheCount,
             const uint32_t BucketID) {
  using SlabHashTypeT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeSearched)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t RequiredKey =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)),
                    SourceLane,
                    32);
    uint32_t Data = (CurrentSlabPtr == A_INDEX_POINTER)
                        ? *getPointerFromBucket(SourceBucket, LaneID)
                        : *getPointerFromSlab(CurrentSlabPtr, LaneID);

    uint32_t KeyCountInWarp = __popc(__ballot_sync(0xFFFFFFFF, Data == RequiredKey) &
                                     SlabHashT::REGULAR_NODE_KEY_MASK);

    if (LaneID == SourceLane)
      TheCount = TheCount + KeyCountInWarp;

    uint32_t NextPtr = __shfl_sync(0xFFFFFFFF, Data, SlabHashTypeT::NEXT_PTR_LANE, 32);
    if (NextPtr == SlabHashTypeT::EMPTY_INDEX_POINTER) {
      if (LaneID == SourceLane)
        ToBeSearched = false;
    } else {
      CurrentSlabPtr = NextPtr;
      IsHeadSlab = false;
    }
  }
}

#endif  // PCMAP_HASH_CTXT_COUNT_H_
