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

#ifndef PCMAP_HASH_CTXT_UPDATE_H_
#define PCMAP_HASH_CTXT_UPDATE_H_

template <typename FilterMapTy>
template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ void
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>::
    updatePair(bool& ToBeUpdated,
               const uint32_t& LaneID,
               const KeyT& TheKey,
               const ValueT& TheValue,
               const uint32_t BucketID,
               FilterMapTy* FilterMap) {
  using SlabHashT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeUpdated)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey))),
                    SourceLane,
                    32);

    uint32_t* PtrData = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                            ? getPointerFromBucket(SourceBucket, LaneID)
                            : getPointerFromSlab(CurrentSlabPtr, LaneID);
    uint32_t Data = *PtrData;

    int FoundLane =
        (__ballot_sync(0xFFFFFFFF, Data == ReqKey) & SlabHashT::REGULAR_NODE_KEY_MASK) -
        1;

    if (FoundLane < 0) {
      uint32_t NextSlabPtr = __shfl_sync(0xFFFFFFFF, Data, SlabHashT::NEXT_PTR_LANE, 32);
      if (NextSlabPtr == SlabHashT::EMPTY_INDEX_POINTER) {
        if (LaneID == SourceLane)
          ToBeUpdated = false;
      } else {
        CurrentSlabPtr = NextSlabPtr;
        IsHeadSlab = false;
      }
    } else {
      const uint32_t UpdateLane = static_cast<uint32_t>(FoundLane);
      uint32_t ValueSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::VALUES_PTR_LANE, 32);
      uint32_t MutexSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::MUTEX_PTR_LANE, 32);

      if (LaneID == SourceLane) {
        uint32_t* KeyPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                               ? getPointerFromBucket(SourceBucket, UpdateLane)
                               : getPointerFromSlab(CurrentSlabPtr, UpdateLane);
        uint32_t* ValuePtr = getPointerFromSlab(ValueSlabVAddr, UpdateLane);
        uint32_t* MutexPtr = getPointerFromSlab(MutexSlabVAddr, UpdateLane);

        if (atomicCAS(MutexPtr, EMPTY_KEY, 0) == EMPTY_KEY) {
          uint32_t Key = *KeyPtr;
          uint32_t CurrentSlabValue = *ValuePtr;

          if (Key ==
              *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)))) {
            ValueT NewValue = (FilterMap != nullptr)
                                  ? (*FilterMap)(CurrentSlabValue, TheValue)
                                  : FilterMapTy()(CurrentSlabValue, TheValue);

            if (NewValue != CurrentSlabValue)
              *ValuePtr =
                  *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&NewValue)));

            ToBeUpdated = false;
          }
          atomicExch(MutexPtr, EMPTY_KEY);
        }
      }
    }
  }
}

template <typename FilterTy>
template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ UpsertStatusKind
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>::
    upsertPair(bool& ToBeUpserted,
               const uint32_t& LaneID,
               const KeyT& TheKey,
               const ValueT& TheValue,
               const uint32_t BucketID,
               typename AllocPolicy::AllocatorContext& TheAllocatorContext,
               FilterTy* Filter) {
  using SlabHashT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;
  UpsertStatusKind UpsertStatus = USK_FAIL;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeUpserted)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey))),
                    SourceLane,
                    32);

    uint32_t* PtrData = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                            ? getPointerFromBucket(SourceBucket, LaneID)
                            : getPointerFromSlab(CurrentSlabPtr, LaneID);
    uint32_t Data = *PtrData;

    int FoundLane = __ffs(__ballot_sync(0xFFFFFFFF, Data == ReqKey) &
                          SlabHashT::REGULAR_NODE_KEY_MASK) -
                    1;
    uint32_t EmptyLanes =
        __ballot_sync(0xFFFFFFFF, Data == EMPTY_KEY) & SlabHashT::REGULAR_NODE_KEY_MASK;

    if (FoundLane < 0) {
      if (EmptyLanes == 0) {
        uint32_t NextSlabPtr =
            __shfl_sync(0xFFFFFFFF, Data, SlabHashT::NEXT_PTR_LANE, 32);
        if (NextSlabPtr == SlabHashT::EMPTY_INDEX_POINTER) {
          uint32_t NewKeySlabVAddr = allocateSlab(TheAllocatorContext, LaneID);
          uint32_t NewValueSlabVAddr = allocateSlab(TheAllocatorContext, LaneID);
          uint32_t NewMutexSlabVAddr = allocateSlab(TheAllocatorContext, LaneID);

          if (LaneID == SlabHashT::MUTEX_PTR_LANE) {
            uint32_t* MutexLanePtr =
                getPointerFromSlab(NewKeySlabVAddr, SlabHashT::MUTEX_PTR_LANE);
            *MutexPtrLane = NewMutexSlabVAddr;
          }

          if (LaneID == SlabHashT::VALUES_PTR_LANE) {
            uint32_t* ValuesLanePtr =
                getPointerFromSlab(NewKeySlabVAddr, SlabHashT::VALUES_PTR_LANE);
            *ValuesPtrLane = NewValueSlabVAddr;
          }

          if (LaneID == SlabHashT::NEXT_PTR_LANE) {
            uint32_t* NextLanePtr =
                IsHeadSlab ? getPointerFromBucket(SourceBucket, SlabHashT::NEXT_PTR_LANE)
                           : getPointerFromSlab(CurrentSlabPtr, SlabHashT::NEXT_PTR_LANE);

            uint32_t OldVAddr =
                atomicCAS(NextLanePtr, SlabHashT::EMPTY_INDEX_POINTER, NewKeySlabVAddr);
            if (OldVAddr != SlabHashT::EMPTY_INDEX_POINTER) {
              freeSlab(NewKeySlabVAddr);
              freeSlab(NewValueSlabVAddr);
              freeSlab(NewMutexSlabVAddr);
            }
          }
        } else {
          CurrentSlabPtr = NextSlabPtr;
          IsHeadSlab = false;
        }
      } else {
        uint32_t InsertLane = __ffs(EmptyLanes & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
        uint32_t ValueSlabVAddr =
            __shfl_sync(0xFFFFFFFF, Data, SlabHashT::VALUES_PTR_LANE, 32);
        uint32_t MutexSlabVAddr =
            __shfl_sync(0xFFFFFFFF, Data, SlabHashT::MUTEX_PTR_LANE, 32);

        if (SourceLane == LaneID) {
          uint32_t* KeyPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                                 ? getPointerFromBucket(SourceBucket, InsertLane)
                                 : getPointerFromSlab(CurrentSlabPtr, InsertLane);
          uint32_t* ValuePtr = getPointerFromSlab(ValueSlabVAddr, InsertLane);
          uint32_t* MutexPtr = getPointerFromSlab(MutexSlabVAddr, InsertLane);

          if (atomicCAS(MutexPtr, EMPTY_KEY, 0) == EMPTY_KEY) {
            if ((static_cast<uint64_t>(*ValuePtr) << 32 | *KeyPtr) == EMPTY_PAIR_64) {
              *KeyPtr = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey));
              *ValuePtr =
                  *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheValue));
              ToBeUpserted = false;
              UpsertStatus = USK_INSERT;
            }

            atomicExch(MutexPtr, EMPTY_KEY);
          }
        }
      }
    } else {
      const uint32_t UpdateLane = static_cast<uint32_t>(FoundLane);
      uint32_t ValueSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::VALUES_PTR_LANE, 32);
      uint32_t MutexSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::MUTEX_PTR_LANE, 32);

      if (LaneID == SourceLane) {
        uint32_t* KeyPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                               ? getPointerFromBucket(SourceBucket, UpdateLane)
                               : getPointerFromSlab(CurrentSlabPtr, UpdateLane);
        uint32_t* ValuePtr = getPointerFromSlab(ValueSlabVAddr, UpdateLane);
        uint32_t* MutexPtr = getPointerFromSlab(MutexSlabVAddr, UpdateLane);

        if (atomicCAS(MutexPtr, EMPTY_KEY, 0) == EMPTY_KEY) {
          uint32_t Key = *KeyPtr;
          uint32_t CurrentSlabValue = *ValuePtr;

          if (Key ==
              *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)))) {
            bool FilterStatus = (Filter != nullptr)
                                    ? (*Filter)(CurrentSlabValue, TheValue)
                                    : FilterTy()(CurrentSlabValue, TheValue);

            if (FilterStatus) {
              *ValuePtr =
                  *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheValue)));
              UpsertStatus = USK_UPDATE;
            } else {
              UpsertStatus = USK_FAIL;
            }

            ToBeUpserted = false;
          }
          atomicExch(MutexPtr, EMPTY_KEY);
        }
      }
    }
  }

  return UpsertStatus;
}

#endif  // PCMAP_HASH_CTXT_UPDATE_H_
