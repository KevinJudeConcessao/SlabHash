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

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
__device__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>::updatePair(
        bool& ToBeUpdated,
        const uint32_t& LaneID,
        const KeyT& TheKey,
        const ValueT& TheValue,
        const uint32_t BucketID,
        AllocatorContextT& LocalAllocatorCtxt) {
  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;
  uint64_t OldKeyValuePair = 0;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeUpdated)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t MyKey = *(reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)));
    uint32_t ReqKey = __shfl_sync(0xFFFFFFFF, MyKey, SourceLane, 32);
    uint32_t SourceData = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                              ? *(getPointerFromBucket(SourceBucket, LaneID))
                              : *(getPointerFromSlab(CurrentSlabPtr, LaneID));

    int FoundLane = __ffs(__ballot_sync(0xFFFFFFFF, SourceData == ReqKey) &
                          SlabHashT::REGULAR_NODE_KEY_MASK) -
                    1;

    if (FoundLane < 0) {
      uint32_t NextPtr = __shfl_sync(0xFFFFFFFF, SourceData, 31, 32);
      if (NextPtr == SlabHashT::EMPTY_INDEX_POINTER) {
        if (LaneID == SourceLane)
          ToBeUpdated = false;
      } else  
        CurrentSlabPtr = NextPtr;
        IsHeadSlab = false;
      }
    } else {
      if (LaneID == SourceLane) {
        uint64_t* UpdatePtr = reinterpret_cast<uint64_t*>(
            CurrentSlabPtr == SlabHashT::A_INDEX_POINTER
                ? getPointerFromBucket(SourceBucket, FoundLane)
                : getPointerFromSlab(CurrentPtr, FoundLane));
        uint64_t CurrentKeyValuePair = *UpdatePtr;
        uint32_t CurrentValueInSlab = static_cast<uint32_t>(CurrentKeyValuePair >> 16);
        FilterTy F{};
        MapTy M{CurrentValueInSlab};

        if (F(CurrentValueInSlab)) {
          uint32_t NewValueForSlab = M(TheValue);
          OldKeyValuePair = atomicCAS(UpdatePtr,
                                      CurrentKeyValuePair,
                                      (static_cast<uint64_t>(*reinterpret_cast<uint32_t*>(
                                           reinterpret_cast<uint8_t*>(&NewValueForSlab)))
                                       << 32) |
                                          MyKey);

          if (OldKeyValuePair == CurrentKeyValuePair)
            ToBeUpdated = false;
        } else {
          ToBeUpdated = false;
        }
      }
    }
  }
}

template <typename FilterTy, typename MapTy>
template <typename KeyT, typename ValueT>
__device__ __forceinline__
    typename std::enable_if<FilterCheck<FilterTy>::value && MapCheck<MapTy>::value>::type
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::ConcurrentMap>::upsertPair(
        bool& ToBeUpserted,
        const uint32_t& LaneID,
        const KeyT& TheKey,
        const ValueT& TheValue,
        const uint32_t BucketID,
        AllocatorContextT& LocalAllocatorCtxt) {
  using SlabHashT = ConcurrentMapT<KeyT, ValueT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;
  uint64_t OldKeyValuePair = 0;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeUpserted)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);

    /* Implementation is incomplete.
     * TODO: Finish the upsert device kernel
     */
  }

  __builtin_unreachable();
}
