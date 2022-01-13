/*
 * Copyright 2019 Saman Ashkiani
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

namespace iterator {
template <typename ContainerPolicyT, typename BucketIteratorT>
class SlabHashIterator;

template <typename ContainerPolicyT>
class BucketIterator {
 protected:
  typename ContainerPolicyT::SlabHashContextT& SlabHashCtxt;
  uint32_t BucketId;
  uint32_t AllocatorAddr;
  uint32_t* PtrPrevSlabNextLane;

 public:
  using SlabInfoT = typename ContainerPolicyT::SlabInfoT;

 public:
  __device__ __forceinline__ BucketIterator& operator++() {
    uint32_t LaneId = threadIdx.x & 0x1F;
    if (AllocatorAddr != SlabInfoT::EMPTY_INDEX_POINTER) {
      if (LaneId == SlabInfoT::NEXT_PTR_LANE) {
        PtrPrevSlabNextLane =
            (AllocatorAddr == SlabInfoT::A_INDEX_POINTER)
                ? SlabHashCtxt.getPointerFromBucket(BucketId, LaneId)
                : SlabHashCtxt.getPointerFromSlab(AllocatorAddr, LaneId);
        AllocatorAddr = *PtrPrevSlabNextLane;
      }

      PtrPrevSlabNextLane = reinterpret_cast<uint32_t*>(
          __shfl_sync(0xFFFFFFFF,
                      reinterpret_cast<uint64_t>(PtrPrevSlabNextLane),
                      SlabInfoT::NEXT_PTR_LANE,
                      32));
      AllocatorAddr =
          __shfl_sync(0xFFFFFFFF, AllocatorAddr, SlabInfoT::NEXT_PTR_LANE, 32);
    }
    
    return (*this);
  }

  __device__ __forceinline__ BucketIterator<ContainerPolicyT> operator++(int) {
    BucketIterator<ContainerPolicyT> OldIterator{*this};
    ++(*this);
    return OldIterator;
  }

  __device__ __forceinline__ uint32_t* GetPtrPrevSlabNextLane() {
    return PtrPrevSlabNextLane;
  }

  __device__ __forceinline__ typename ContainerPolicyT::KeyT* GetPointer(
      uint32_t LaneID = 0) {
    uint32_t* Ptr = (AllocatorAddr == SlabInfoT::A_INDEX_POINTER)
                        ? SlabHashCtxt.getPointerFromBucket(BucketId, LaneID)
                        : SlabHashCtxt.getPointerFromSlab(AllocatorAddr, LaneID);
    return reinterpret_cast<typename ContainerPolicyT::KeyT*>(Ptr);
  }

  __device__ __forceinline__ bool operator==(
      const BucketIterator<ContainerPolicyT>& Other) {
    return (BucketId == Other.BucketId) && (AllocatorAddr == Other.AllocatorAddr);
  }

  __device__ __forceinline__ bool operator!=(
      const BucketIterator<ContainerPolicyT>& Other) {
    return !(*this == Other);
  }

  __device__ __forceinline__ BucketIterator<ContainerPolicyT>
  BucketShuffleSync(unsigned Mask, int SourceLane, int Width = 32) {
    uint64_t SlabAddr = reinterpret_cast<uint64_t>(&SlabHashCtxt);
    uint64_t TheSlabAddr = __shfl_sync(Mask, SlabAddr, SourceLane, Width);
    uint32_t TheBucketId = __shfl_sync(Mask, BucketId, SourceLane, Width);
    uint32_t TheAllocatorAddr = __shfl_sync(Mask, AllocatorAddr, SourceLane, Width);
    uint64_t ThePtrPrevSlabNextLane = __shfl_sync(
        Mask, reinterpret_cast<uint64_t>(PtrPrevSlabNextLane), SourceLane, Width);

    return {*reinterpret_cast<typename ContainerPolicyT::SlabHashContextT*>(TheSlabAddr),
            TheBucketId,
            TheAllocatorAddr,
            ThePtrPrevSlabNextLane};
  }

  __device__ BucketIterator(typename ContainerPolicyT::SlabHashContextT& TheSlabHashCtxt,
                            uint32_t TheBucketId,
                            uint32_t TheAllocatorAddr = SlabInfoT::A_INDEX_POINTER,
                            uint32_t* ThePrevSlabNextLanePtr = nullptr)
      : SlabHashCtxt{TheSlabHashCtxt}
      , BucketId{TheBucketId}
      , AllocatorAddr{TheAllocatorAddr}
      , PtrPrevSlabNextLane{ThePrevSlabNextLanePtr} {}

  __device__ BucketIterator(const BucketIterator<ContainerPolicyT>& Other)
      : SlabHashCtxt{Other.SlabHashCtxt}
      , BucketId{Other.BucketId}
      , AllocatorAddr{Other.AllocatorAddr}
      , PtrPrevSlabNextLane{Other.PtrPrevSlabNextLane} {}

  __device__ __forceinline__ typename ContainerPolicyT::SlabHashContextT&
  GetSlabHashCtxt() {
    return SlabHashCtxt;
  }

  __device__ __forceinline__ uint32_t GetBucketId() { return BucketId; }
  __device__ __forceinline__ uint32_t GetAllocatorAddr() { return AllocatorAddr; }
};

template <typename BucketIteratorT>
struct ResultT {
  static constexpr uint32_t InvalidLane = 32;
  uint32_t LaneID;
  BucketIteratorT TheIterator;
};

template <typename ContainerPolicyT, typename BucketIteratorT>
class SlabHashIterator {
 public:
  using SlabInfoT = typename ContainerPolicyT::SlabInfoT;

 protected:
  typename ContainerPolicyT::SlabHashContextT& SlabHashCtxt;
  BucketIterator<ContainerPolicyT> TheBucketIterator;

 public:
  __device__ __forceinline__ SlabHashIterator<ContainerPolicyT, BucketIteratorT>&
  operator++() {
    uint32_t BucketId = TheBucketIterator.GetBucketId();
    if (BucketId < SlabHashCtxt.getNumBuckets()) {
      ++TheBucketIterator;
      if (TheBucketIterator.GetAllocatorAddr() == SlabInfoT::EMPTY_INDEX_POINTER) {
        TheBucketIterator = BucketIterator<ContainerPolicyT>(
            SlabHashCtxt, BucketId + 1, SlabInfoT::A_INDEX_POINTER);
      }
    }

    return (*this);
  }

  __device__ __forceinline__ SlabHashIterator<ContainerPolicyT, BucketIteratorT>
  operator++(int) {
    SlabHashIterator<ContainerPolicyT, BucketIteratorT> OldIterator{*this};
    ++(*this);
    return OldIterator;
  }

  __device__ __forceinline__ typename ContainerPolicyT::KeyT* GetPointer(
      uint32_t LaneID = 0) {
    return reinterpret_cast<typename ContainerPolicyT::KeyT*>(
        TheBucketIterator.GetPointer(LaneID));
  }

  __device__ __forceinline__ bool operator==(
      const SlabHashIterator<ContainerPolicyT, BucketIteratorT>& Other) {
    return TheBucketIterator == Other.TheBucketIterator;
  }

  __device__ __forceinline__ bool operator!=(
      const SlabHashIterator<ContainerPolicyT, BucketIteratorT>& Other) {
    return TheBucketIterator != Other.TheBucketIterator;
  }

  __device__ __forceinline__ typename ContainerPolicyT::SlabHashContextT&
  GetSlabHashCtxt() {
    return SlabHashCtxt;
  }

  __device__ __forceinline__ BucketIterator<ContainerPolicyT>& GetBucketIterator() {
    return TheBucketIterator;
  }

  __device__ SlabHashIterator(
      typename ContainerPolicyT::SlabHashContextT& TheSlabHashCtxt,
      uint32_t TheBucketId,
      uint32_t TheAllocatorAddr = SlabInfoT::A_INDEX_POINTER)
      : SlabHashCtxt{TheSlabHashCtxt}
      , TheBucketIterator{TheSlabHashCtxt, TheBucketId, TheAllocatorAddr} {}

  __device__ SlabHashIterator(
      const SlabHashIterator<ContainerPolicyT, BucketIteratorT>& Other)
      : SlabHashCtxt{Other.SlabHashCtxt}, TheBucketIterator{Other.TheBucketIterator} {}

  __device__ SlabHashIterator(const BucketIterator<ContainerPolicyT>& Other)
      : SlabHashCtxt{Other.GetSlabHashCtxt()}, TheBucketIterator{Other} {}
};
}  // end namespace iterator

// a forward iterator for the slab hash data structure:
// currently just specialized for concurrent set
// TODO implement for other types
template <typename KeyT, typename AllocPolicy>
class SlabIterator {
 public:
  using SlabHashT = ConcurrentSetT<KeyT>;

  GpuSlabHashContext<KeyT, KeyT, AllocPolicy, SlabHashTypeT::ConcurrentSet>& slab_hash_;

  // current position of the iterator
  KeyT* cur_ptr_;
  uint32_t cur_size_;    // keep track of current level's size (in units of
                         // sizeof(KeyT))
  uint32_t cur_bucket_;  // keeping track of the current bucket
  SlabAddressT cur_slab_address_;
  // initialize the iterator with the first bucket's pointer address of the slab
  // hash
  __host__ __device__
  SlabIterator(GpuSlabHashContext<KeyT, KeyT, AllocPolicy, SlabHashTypeT::ConcurrentSet>&
                   slab_hash)
      : slab_hash_(slab_hash)
      , cur_ptr_(reinterpret_cast<KeyT*>(slab_hash_.getDeviceTablePointer()))
      , cur_size_(slab_hash_.getNumBuckets() * SlabHashT::BASE_UNIT_SIZE)
      , cur_bucket_(0)
      , cur_slab_address_(*slab_hash.getPointerFromBucket(0, SlabHashT::NEXT_PTR_LANE)) {}

  __device__ __forceinline__ KeyT* getPointer() const { return cur_ptr_; }
  __device__ __forceinline__ uint32_t getSize() const { return cur_size_; }

  // returns true, if there's a valid next element, else returns false
  // this function is being run by only one thread, so it is wrong to assume all
  // threads within a warp have access to the caller's iterator state
  __device__ __forceinline__ bool next() {
    if (cur_bucket_ == slab_hash_.getNumBuckets()) {
      return false;
    }

    while (cur_slab_address_ == SlabHashT::EMPTY_INDEX_POINTER) {
      cur_bucket_++;
      if (cur_bucket_ == slab_hash_.getNumBuckets()) {
        return false;
      }
      cur_slab_address_ =
          *slab_hash_.getPointerFromBucket(cur_bucket_, SlabHashT::NEXT_PTR_LANE);
    }

    cur_ptr_ = slab_hash_.getPointerFromSlab(cur_slab_address_, 0);
    cur_slab_address_ =
        *slab_hash_.getPointerFromSlab(cur_slab_address_, SlabHashT::NEXT_PTR_LANE);
    cur_size_ = SlabHashT::BASE_UNIT_SIZE;
    return true;
  }
};