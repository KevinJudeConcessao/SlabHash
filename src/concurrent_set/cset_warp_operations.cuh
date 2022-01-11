/*
 * Copyright 2019 Saman Ashkiani
 * Copyright 2021 [TODO: Assign copyright]
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

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::insertKey(
    bool& to_be_inserted,
    const uint32_t& laneId,
    const KeyT& myKey,
    const uint32_t bucket_id,
    typename AllocPolicy::AllocatorContextT& local_allocator_ctx) {
  using SlabHashT = ConcurrentSetT<KeyT>;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = 0;
  uint32_t next = SlabHashT::A_INDEX_POINTER;
  bool new_insertion = false;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_inserted))) {
    // to know whether it is a base node, or a regular node
    next = (last_work_queue != work_queue) ? SlabHashT::A_INDEX_POINTER
                                           : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    KeyT src_key = __shfl_sync(0xFFFFFFFF, myKey, src_lane, 32);
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);

    uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                 ? *getPointerFromBucket(src_bucket, laneId)
                                 : *getPointerFromSlab(next, laneId);

    uint32_t old_key = 0;

    // looking for the same key (if it exists), or an empty spot:
    int32_t dest_lane = SlabHash_NS::findKeyOrEmptyPerWarp<KeyT, ConcurrentSetT<KeyT>>(
        src_key, src_unit_data);

    if (dest_lane == -1) {  // key not found and/or no empty slot available:
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {
        // allocate a new node:
        uint32_t new_node_ptr = allocateSlab(local_allocator_ctx, laneId);

        if (laneId == 31) {
          uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                            ? getPointerFromBucket(src_bucket, 31)
                            : getPointerFromSlab(next, 31);

          uint32_t temp =
              atomicCAS((unsigned int*)p, SlabHashT::EMPTY_INDEX_POINTER, new_node_ptr);
          // check whether it was successful, and
          // free the allocated memory otherwise
          if (temp != SlabHashT::EMPTY_INDEX_POINTER)
            freeSlab(new_node_ptr);
        }
      } else {
        next = next_ptr;
      }
    } else {  // either the key is found, or there is an empty slot available
      if (laneId == src_lane) {
        const uint32_t* p = (next == SlabHashT::A_INDEX_POINTER)
                                ? getPointerFromBucket(src_bucket, dest_lane)
                                : getPointerFromSlab(next, dest_lane);

        old_key = atomicCAS((unsigned int*)p,
                            EMPTY_KEY,
                            *reinterpret_cast<const uint32_t*>(
                                reinterpret_cast<const unsigned char*>(&myKey)));
        new_insertion = (old_key == EMPTY_KEY);
        if (new_insertion || (old_key == src_key)) {
          to_be_inserted = false;  // succesful insertion
        }
      }
    }
    last_work_queue = work_queue;
  }
  return new_insertion;
}

// ========
template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::searchKey(
    bool& to_be_searched,
    const uint32_t& laneId,
    const KeyT& myKey,
    const uint32_t bucket_id) {
  bool myResult = false;
  using SlabHashT = ConcurrentSetT<KeyT>;
  uint32_t work_queue = 0;
  uint32_t last_work_queue = work_queue;
  uint32_t next = SlabHashT::A_INDEX_POINTER;

  while ((work_queue = __ballot_sync(0xFFFFFFFF, to_be_searched))) {
    next = (last_work_queue != work_queue) ? SlabHashT::A_INDEX_POINTER
                                           : next;  // a successfull insertion in the warp
    uint32_t src_lane = __ffs(work_queue) - 1;
    uint32_t src_bucket = __shfl_sync(0xFFFFFFFF, bucket_id, src_lane, 32);
    KeyT wanted_key = __shfl_sync(0xFFFFFFFF, myKey, src_lane, 32);

    const uint32_t src_unit_data = (next == SlabHashT::A_INDEX_POINTER)
                                       ? *getPointerFromBucket(src_bucket, laneId)
                                       : *getPointerFromSlab(next, laneId);

    int32_t found_lane = SlabHash_NS::findKeyPerWarp<KeyT, ConcurrentSetT<KeyT>>(
        wanted_key, src_unit_data);

    if (found_lane < 0) {  // not found
      uint32_t next_ptr = __shfl_sync(0xFFFFFFFF, src_unit_data, 31, 32);
      if (next_ptr == SlabHashT::EMPTY_INDEX_POINTER) {  // not found
        if (laneId == src_lane) {
          to_be_searched = false;
        }
      } else {
        next = next_ptr;
      }
    } else {  // found the key:
      if (laneId == src_lane) {
        to_be_searched = false;
        myResult = true;
      }
    }
    last_work_queue = work_queue;
  }
  return myResult;
}

// ========
template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::
    searchKeyBulk(const uint32_t& laneId, const KeyT& myKey, const uint32_t bucket_id) {}

// ========
template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ bool
GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::deleteKey(
    bool& ToBeDeleted,
    const uint32_t& LaneID,
    KeyT& TheKey,
    const uint32_t BucketID) {
  using SlabHashT = ConcurrentSetT<KeyT>;

  uint32_t WorkQueue = 0;
  uint32_t CurrentSlabPtr = SlabHashT::A_INDEX_POINTER;
  bool IsHeadSlab = true;
  bool DeletionStatus = false;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToBeDeleted)) != 0) {
    CurrentSlabPtr = IsHeadSlab ? SlabHashT::A_INDEX_POINTER : CurrentSlabPtr;

    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t SourceBucket = __shfl_sync(0xFFFFFFFF, BucketID, SourceLane, 32);
    uint32_t Data = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                        ? *getPointerFromBucket(SourceBucket, LaneID)
                        : *getPointerFromSlab(CurrentSlabPtr, LaneID);

    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&TheKey)),
                    SourceLane,
                    32);

    uint32_t IsFound =
        __ballot_sync(0xFFFFFFFF, Data == ReqKey) & SlabHashT::REGULAR_NODE_KEY_MASK;

    if (IsFound) {
      int CandidateLane = __ffs(IsFound) - 1;

      if (LaneID == SourceLane) {
        uint32_t* DestPtr = (CurrentSlabPtr == SlabHashT::A_INDEX_POINTER)
                                ? getPointerFromBucket(SourceBucket, CandidateLane)
                                : getPointerFromSlab(CurrentSlabPtr, CandidateLane);
        uint32_t OldKey = atomicExch(DestPtr, TOMBSTONE_KEY);

        DeletionStatus = (OldKey == TheKey);
        ToBeDeleted = false;
      }
    } else {
      uint32_t NextSlabPtr = __shfl_sync(0xFFFFFFFF, Data, SlabHashT::NEXT_PTR_LANE, 32);

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

template <typename KeyT, typename ValueT, typename AllocPolicy>
using ResultT =
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::ResultT;

template <typename KeyT, typename ValueT, typename AllocPolicy>
using IteratorT =
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>::
        BucketIterator;

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ ResultT<KeyT, ValueT, AllocPolicy> insertKey(
    bool& to_be_inserted,
    const uint32_t& laneId,
    const KeyT& myKey,
    IteratorT<KeyT, ValueT, AllocPolicy>& Iterator,
    typename AllocPolicy::AllocatorContextT& local_allocator_context) {

  using SlabHashT = ConcurrentSet<KeyT>;
  using BucketIterator = Iterator<KeyT, ValueT, AllocPolicy>;
  uint32_t WorkQueue = 0;

  BucketIterator TheIterator{EndAt(CurrentIterator.GetBucketId())};
  uint32_t FoundLane = ResultT<KeyT, ValueT, AllocPolicy>::InvalidLane;
  
  /*
   * TODO: Implement iterator based insert operation
   */

  return {FoundLane, TheIterator};
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ ResultT<KeyT, ValueT, AllocPolicy> searchKey(
    bool& to_be_searched,
    const uint32_t& laneId,
    const KeyT& myKey,
    IteratorT<KeyT, ValueT, AllocPolicy>& Iterator) {
  using SlabHashT = ConcurrentSet<KeyT>;
  using BucketIterator = Iterator<KeyT, ValueT, AllocPolicy>;
  uint32_t WorkQueue = 0;

  BucketIterator TheIterator{EndAt(CurrentIterator.GetBucketId())};
  uint32_t FoundLane = ResultT<KeyT, ValueT, AllocPolicy>::InvalidLane;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, to_be_searched)) != 0) {
    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&myKey)),
                    SourceLane,
                    32);

    BucketIterator CurrentIterator =
        Iterator.BucketShuffleSync(0xFFFFFFFF, SourceLane, 32);
    BucketIterator End = EndAt(CurrentIterator.GetBucketId());
    bool Found = false;

    while (!Found && CurrentIterator != End) {
      KeyT TheKey = *CurrentIterator.GetPointer(LaneID);
      int Lane = __ffs(__ballot_sync(0xFFFFFFFF, TheKey == ReqKey) &
                       SlabHashT::REGULAR_NODE_KEY_MASK) -
                 1;
      Found = (Lane >= 0);
      if (Found) {
        if (SourceLane == laneId) {
          TheIterator = CurrentIterator;
          FoundLane = Lane;
          to_be_searched = false;
        }
      } else {
        ++CurrentIterator;
      }
    }
  }

  return {FoundLane, TheIterator};
}

template <typename KeyT, typename ValueT, typename AllocPolicy>
__device__ __forceinline__ ResultT<KeyT, ValueT, AllocPolicy> deleteKey(
    bool& ToDelete,
    const uint32_t& LaneID,
    KeyT& TheKey,
    IteratorT<KeyT, ValueT, AllocPolicy>& Iterator) {
  using SlabHashT = ConcurrentSet<KeyT>;
  using BucketIterator = Iterator<KeyT, ValueT, AllocPolicy>;
  uint32_t WorkQueue = 0;

  BucketIterator TheIterator{EndAt(CurrentIterator.GetBucketId())};
  uint32_t FoundLane = ResultT<KeyT, ValueT, AllocPolicy>::InvalidLane;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToDelete)) != 0) {
    uint32_t SourceLane = __ffs(WorkQueue) - 1;
    uint32_t ReqKey =
        __shfl_sync(0xFFFFFFFF,
                    *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(&myKey)),
                    SourceLane,
                    32);

    BucketIterator CurrentIterator =
        Iterator.BucketShuffleSync(0xFFFFFFFF, SourceLane, 32);
    KeyT CurrentLaneKey = *CurrentIterator.GetPointer(LaneID);
    int Lane = __ffs(__ballot_sync(0xFFFFFFFF, CurrentLaneKey == ReqKey) &
                     SlabHashT::REGULAR_NODE_KEY_MASK) -
               1;

    bool Found = (Lane >= 0);
    if (Found) {
      if (LaneID == SourceLane) {
        uint32_t* DestPtr = TheIterator.GetPointer(Lane);
        if (TheKey == atomicExch(DestPtr, TOMBSTONE_KEY)) {
          TheIterator = CurrentIterator;
          FoundLane = Lane;
        }
      }
    }

    if (LaneID == SourceLane)
      ToBeDeleted = false;
  }

  return {FoundLane, TheIterator};
}
