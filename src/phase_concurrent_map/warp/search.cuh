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

#ifndef PCMAP_HASH_CTXT_SEARCH_H_
#define PCMAP_HASH_CTXT_SEARCH_H_

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::searchKey(
    bool& ToBeSearched,
    const uint32_t& LaneID,
    const KeyT& TheKey,
    const ValueT& TheValue,
    const uint32_t BucketID) {
  using SlabHashT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeSearched)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __ballot_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t SearchKey =
        __shfl_sync(0xFFFFFFFF,
                    *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey))),
                    SourceLane,
                    32);
    uint32_t FoundKey = IsHeadSlab ? *getPointerFromBucket(SourceBucket, LaneID)
                                   : *getPointerFromSlab(CurrentSlabPtr, LaneID);
    uint32_t FoundLane = __ffs(__ballot_sync(0xFFFFFFFF, SearchKey == FoundKey) &
                               SlabHashT::REGULAR_NODE_KEY_MASK) -
                         1;

    if (FoundLane < 0) {
      uint32_t NextPtr = __shfl_sync(0xFFFFFFFF, FoundKey, SlabHashT::NEXT_PTR_LANE, 32);

      if (NextPtrLane == SlabHashT::EMPTY_INDEX_POINTER) {
        if (LaneID == SourceLane) {
          TheValue = static_cast<ValueT>(SEARCH_NOT_FOUND);
          ToBeSearched = false;
        }
      } else {
        CurrentSlabPtr = NextPtr;
        IsHeadSlab = false;
      }
    } else {
      uint32_t ValuesPtr =
          __shfl_sync(0xFFFFFFFF, FoundKey, SlabHashT::VALUES_PTR_LANE, 32);

      /* Avoid bank conflict by having all the warp threads request value
       * in their lanes
       */
      uint32_t FoundValue =
          __shfl_sync(0xFFFFFFFF, *getPointerFromSlab(ValuesPtr, LaneID), FoundLane, 32);

      if (LaneID == SourceLane) {
        TheValue = *reinterpret_cast<const ValueT*>(
            reinterpret_cast<const uint8_t*>(&FoundValue));
        ToBeSearched = false;
      }
    }
  }
}

template <typename KeyT, typename ValueT>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::searchKeyBulk(
    const uint32_t& LaneID,
    const KeyT& TheKey,
    const ValueT& TheValue,
    const uint32_t BucketID) {
  /* TODO: Complete implementation */ 
  __builtin_unreachable();
}

#endif  // PCMAP_HASH_CTXT_SEARCH_H_
