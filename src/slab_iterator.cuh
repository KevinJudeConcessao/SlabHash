/*
 * Copyright 2019 Saman Ashkiani
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

// a forward iterator for the slab hash data structure:
// currently just specialized for concurrent set
// TODO implement for other types

template <typename ContainerPolicyT>
class SlabHashIterator;

template <typename ContainerPolicyT>
class BucketIterator {
  typename ContainerPolicyT::SlabHashContextT& SlabHashCtxt;
  uint32_t BucketId;
  uint32_t AllocatorAddr;
  using SlabInfoT = typename ContainerPolicyT::SlabInfoT;

 public:
  __device__ BucketIterator& operator++() {
    uint32_t LaneId = threadIdx.x & 0x1F;
    if (AllocatorAddr == SlabInfoT::EMPTY_INDEX_POINTER)
      return;

    if (LaneId == SlabInfoT::NEXT_PTR_LANE) {
      AllocatorAddr = (AllocatorAddr == SlabInfoT::A_INDEX_POINTER)
                          ? *SlabHashCtxt.getPointerFromBucket(BucketId, LaneID)
                          : *SlabHashCtxt.getPointerFromSlab(AllocatorAddr, LaneID);
    }

    AllocatorAddr = __shfl_sync(0xFFFFFFFF, AllocatorAddr, SlabInfoT::NEXT_PTR_LANE, 32);
  }

  __device__ BucketIterator operator++(int) {
    BucketIterator OldIterator{*this};
    ++(*this);
    return OldIterator;
  }

  __device__ typename ContainerPolicyT::KeyT* GetPointer(uint32_t LaneID = 0) {
    uint32_t* Ptr = (AllocatorAddr == SlabInfoT::A_INDEX_POINTER)
                        ? SlabHashCtxt.getPointerFromBucket(BucketId, LaneID)
                        : SlabHashCtxt.getPointerFromSlab(AllocatorAddr, LaneID);
    return reinterpret_cast<typename ContainerPolicyT::KeyT*>(Ptr);
  }

  __device__ bool operator==(const BucketIterator& Other) {
    return (BucketId == Other.BucketId) && (AllocatorAddr == Other.AllocatorAddr);
  }

  __device__ bool operator!=(const BucketIterator& Other) { return !(*this == Other); }

  __device__ BucketIterator(typename ContainerPolicyT::SlabHashContextT& TheSlabHashCtxt,
                            uint32_t TheBucketId,
                            uint32_t TheAllocatorAddr = SlabInfoT::A_INDEX_POINTER)
      : SlabHashCtxt{TheSlabHashCtxt}
      , BucketId{TheBucketId}
      , AllocatorAddr{TheAllocatorAddr} {}

  __device__ BucketIterator(const BucketIterator& Other)
      : SlabHashCtxt{Other.TheSlabHashCtxt}
      , BucketId{Other.TheBucketId}
      , AllocatorAddr{Other.TheAllocatorAddr} {}

  __device__ ContainerPolicyT::SlabHashContextT& GetSlabHashCtxt() {
    return SlabHashCtxt;
  }

  __device__ uint32_t GetBucketId() { return BucketId; }
  __device__ uint32_t GetAllocatorAddr() { return AllocatorAddr; }

 private:
  friend class SlabHashIterator<ContainerPolicyT>;
};

template <typename ContainerPolicyT>
class SlabHashIterator {
  typename ContainerPolicyT::SlabHashContextT& SlabHashCtxt;
  BucketIterator TheBucketIterator;

 public:
  __device__ SlabHashIterator& operator++() {
    /*
     * TODO: Expand SlabHashIterator ++ operator
     */
  }

  __device__ SlabHashIterator operator++(int) {
    SlabHashIterator OldIterator{*this};
    ++(*this);
    return OldIterator;
  }

  __device__ typename ContainerPolicyT::KeyT* getPointer(uint32_t LaneID = 0) {
    return reinterpret_cast<typename ContainerPolicyT::KeyT*>(
        TheBucketIterator.getPointer(LaneID));
  }

  __device__ bool operator==(const SlabHashIterator& Other) {
    return TheBucketIterator == Other.TheBucketIterator;
  }

  __device__ bool operator!=(const SlabHashIterator& Other) {
    return TheBucketIterator != Other.TheBucketIterator;
  }

  __device__ ContainerPolicyT::SlabHashContextT& GetSlabHashCtxt() {
    return SlabHashCtxt;
  }

  __device__ BucketIterator GetBucketIterator() { return TheBucketIterator; }

  __device__ SlabHashIterator(
      typename ContainerPolicyT::SlabHashContextT& TheSlabHashCtxt,
      uint32_t TheBucketId,
      uint32_t TheAllocatorAddr = SlabInfoT::A_INDEX_POINTER)
      : SlabHashCtxt{TheSlabHashCtxt}
      , TheBucketIterator{TheSlabHashCtxt, TheBucketId, TheallocatorAddr} {}

  __device__ SlabHashIterator(const SlabHashIterator& Other)
      : SlabHashCtxt{Other.TheSlabHashCtxt}, TheBucketIterator{Other.TheBucketIterator} {}

  __device__ SlabHashIterator(const BucketIterator& Other)
      : SlabHashCtxt{Other.GetSlabHashCtxt()}, TheBucketIterator{Other} {}
};

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