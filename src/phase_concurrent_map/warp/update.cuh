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

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
__device__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::updatePair(
        bool& ToBeUpdated,
        const uint32_t& LaneID,
        const KeyT& TheKey,
        const ValueT& TheValue,
        const uint32_t BucketID,
        AllocatorContext& TheAllocatorContext) {
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

    uint32_t* DataPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                            ? getPointerFromBucket(SourceBucket, LaneID)
                            : getPointerFromSlab(CurrentSlabPtr, LaneID);
    uint32_t Data = *DataPtr;

    int FoundLane = __ffs(__ballot_sync(0xFFFFFFFF, Data == ReqKey) &
                          SlabHashT::REGULAR_NODE_KEY_MASK) -
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
      uint32_t UpdateLane = __ffs(Found & SlabHashT::REGULAR_NODE_KEY_MASK) - 1;
      uint32_t ValueSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::VALUES_PTR_LANE, 32);
      uint32_t MutexSlabVAddr =
          __shfl_sync(0xFFFFFFFF, Data, SlabHashT::MUTEX_PTR_LANE, 32);

      if (LaneID == SourceLane) {
        uint32_t* ValuePtr = getPointerFromSlab(ValueSlabVAddr, UpdateLane);
        uint32_t* MutexPtr = getPointerFromSlab(MutexSlabVAddr, UpdateLane);

        if (atomicCAS(MutexPtr, EMPTY_KEY, 0) == EMPTY_KEY) {
          uint32_t Key = *DataPtr;
          uint32_t Value = *ValuePtr;

          if (Key ==
              *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)))) {
            FilterTy F{};
            MapTy M{Value};

            if (F(Value))
              *ValuePtr =
                  *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheValue)));

            ToBeUpdated = false;
          }
          atomicExch(MutexPtr, EMPTY_KEY);
        }
      }
    }
  }
}

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
__device__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap>::upsertPair(
        bool& ToBeUpdated,
        const uint32_t& LaneID,
        const KeyT& TheKey,
        const ValueT& TheValue,
        const uint32_t BucketID,
        AllocatorContext& TheAllocatorContext) {
  using SlabHashT = PhaseConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;

  while (WorkQueue = __ballot_sync)
}

#endif  // PCMAP_HASH_CTXT_UPDATE_H_
