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

#ifndef PCMAP_HASH_CTXT_DELETE_H_
#define PCMAP_HASH_CTXT_DELETE_H_

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>::
    deleteKey(bool& ToBeDeleted,
              const uint32_t& LaneID,
              const KeyT& TheKey,
              const uint32_t BucketID) {
  using SlabHashT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;
  bool DeletionStatus = false;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeDeleted)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t FoundKey = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                         ? *getPointerFromBucket(SourceBucket, LaneID)
                         : *getPointerFromSlab(CurrentSlabPtr, LaneID);
    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)),
                    SourceLane,
                    32);

    uint32_t IsFound =
        __ballot_sync(0xFFFFFFFF, FoundKey == ReqKey) & SlabHashT::REGULAR_NODE_KEY_MASK;

    if (IsFound) {
      int CandidateLane = __ffs(IsFound) - 1;
      uint32_t ValuesSlabVAddr =
          __shfl_sync(0xFFFFFFFF, FoundKey, SlabHashT::VALUES_PTR_LANE, 32);
      uint32_t MutexSlabVAddr =
          __shfl_sync(0xFFFFFFFF, FoundKey, SlabHashT::MUTEX_PTR_LANE, 32);

      if (LaneID == SourceLane) {
        uint32_t* DataPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                                ? getPointerFromBucket(SourceBucket, CandidateLane)
                                : getPointerFromSlab(CurrentSlabPtr, CandidateLane);
        uint32_t* ValuePtr = getPointerFromSlab(ValuesSlabVAddr, CandidateLane);
        uint32_t* MutexPtr = getPointerFromSlab(MutexSlabVAddr, CandidateLane);

        if (atomicCAS(MutexPtr, EMPTY_KEY, 0) == EMPTY_KEY) {
          uint32_t Key = *DataPtr;
          if (Key == *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey))) {
            *DataPtr = TOMBSTONE_KEY;
            *ValuePtr = TOMBSTONE_VALUE;
            DeletionStatus = true;
            ToBeDeleted = false;
          }

          atomicExch(MutexPtr, EMPTY_KEY);
        }
      }
    } else {
      uint32_t NextSlabPtr = __shfl_sync(0xFFFFFFFF, FoundKey, SlabHashT::NEXT_PTR_LANE, 32);

      if (NextSlabPtr == SlabHashT::EMPTY_INDEX_POINTER) {
        if (SourceLane == LaneID) {
          DeletionStatus = false;
          ToBeDeleted = false;
        }
      } else {
        CurrentSlabPtr = NextSlabPtr;
        IsHeadSlab = false;
      }
    }
  }

  return DeletionStatus;
}

#endif  // PCMAP_HASH_CTXT_DELETE_H_
