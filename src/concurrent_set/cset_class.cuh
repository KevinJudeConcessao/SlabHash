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

/*
 * This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the gpu
 * (i.e., d_table_)
 */
template <typename KeyT, typename ValueT, typename AllocPolicy>
class GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet> {
 public:
  // fixed known parameters:
  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;
  static constexpr uint32_t WARP_WIDTH_ = 32;

#pragma hd_warning_disable
  __device__ __host__ GpuSlabHashContext()
      : num_buckets_(0), hash_x_(0), hash_y_(0), d_table_(nullptr) {
    // a single slab on a ConcurrentSet should be 128 bytes
  }

#pragma hd_warning_disable
  __host__ __device__ GpuSlabHashContext(
      const GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>&
          rhs) {
    num_buckets_ = rhs.num_buckets_;
    hash_x_ = rhs.hash_x_;
    hash_y_ = rhs.hash_y_;
    d_table_ = rhs.d_table_;
    global_allocator_ctx_ = rhs.global_allocator_ctx_;
  }
#pragma hd_warning_disable
  __host__ __device__ ~GpuSlabHashContext() {}

  static size_t getSlabUnitSize() {
    return sizeof(typename ConcurrentSetT<KeyT>::SlabTypeT);
  }

  static std::string getSlabHashTypeName() { return ConcurrentSetT<KeyT>::getTypeName(); }

  __host__ void initParameters(const uint32_t num_buckets,
                               const uint32_t hash_x,
                               const uint32_t hash_y,
                               int8_t* d_table,
                               typename AllocPolicy::AllocatorContextT* allocator_ctx) {
    num_buckets_ = num_buckets;
    hash_x_ = hash_x;
    hash_y_ = hash_y;
    d_table_ = reinterpret_cast<typename ConcurrentSetT<KeyT>::SlabTypeT*>(d_table);
    global_allocator_ctx_ = *allocator_ctx;
  }

  __device__ __host__ __forceinline__ typename AllocPolicy::AllocatorContextT&
  getAllocatorContext() {
    return global_allocator_ctx_;
  }

  __device__ __host__ __forceinline__ typename ConcurrentSetT<KeyT>::SlabTypeT*
  getDeviceTablePointer() {
    return d_table_;
  }

  __device__ __host__ __forceinline__ uint32_t getNumBuckets() { return num_buckets_; }
  __device__ __host__ __forceinline__ uint32_t getHashX() { return hash_x_; }
  __device__ __host__ __forceinline__ uint32_t getHashY() { return hash_y_; }

  __device__ __host__ __forceinline__ uint32_t computeBucket(const KeyT& key) const {
    return (((hash_x_ ^ key) + hash_y_) % PRIME_DIVISOR_) % num_buckets_;
  }

  // threads in a warp cooperate with each other to insert keys
  // into the slab hash set
  __device__ __forceinline__ bool insertKey(
      bool& to_be_inserted,
      const uint32_t& laneId,
      const KeyT& myKey,
      const uint32_t bucket_id,
      typename AllocPolicy::AllocatorContextT& local_allocator_context);

  // threads in a warp cooeparte with each other to search for keys
  // if found, it returns the true, else false
  __device__ __forceinline__ bool searchKey(bool& to_be_searched,
                                            const uint32_t& laneId,
                                            const KeyT& myKey,
                                            const uint32_t bucket_id);

  // threads in a warp cooperate with each other to search for keys.
  // the main difference with above function is that it is assumed all
  // threads have something to search for (no to_be_searched argument)
  __device__ __forceinline__ bool searchKeyBulk(const uint32_t& laneId,
                                                const KeyT& myKey,
                                                const uint32_t bucket_id);

  __device__ __forceinline__ bool deleteKey(bool& ToDelete,
                                            const uint32_t& LaneID,
                                            KeyT& TheKey,
                                            const uint32_t BucketID);

  __device__ __forceinline__ uint32_t* getPointerFromSlab(
      const SlabAddressT& slab_address,
      const uint32_t laneId) {
    return global_allocator_ctx_.getPointerFromSlab(slab_address, laneId);
  }

  __device__ __forceinline__ uint32_t* getPointerFromBucket(const uint32_t bucket_id,
                                                            const uint32_t laneId) {
    return reinterpret_cast<uint32_t*>(d_table_) +
           bucket_id * ConcurrentSetT<KeyT>::BASE_UNIT_SIZE + laneId;
  }

  using BucketIteratorBase =
      iterator::BucketIterator<ConcurrentSetPolicy<KeyT, AllocPolicy>>;
  template <typename BucketIteratorT>
  using IteratorBase =
      iterator::SlabHashIterator<ConcurrentSetPolicy<KeyT, AllocPolicy>, BucketIteratorT>;

  struct BucketIterator : public BucketIteratorBase {
    __device__ BucketIterator(
        GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>*
            TheSlabHashCtxt,
        uint32_t TheBucketId,
        uint32_t TheAllocatorAddr = ConcurrentSetT<KeyT>::A_INDEX_POINTER,
        uint32_t* ThePrevSlabNextLanePtr = nullptr)
        : BucketIteratorBase{TheSlabHashCtxt,
                             TheBucketId,
                             TheAllocatorAddr,
                             ThePrevSlabNextLanePtr} {}
  };

  struct Iterator : public IteratorBase<BucketIterator> {
    __device__ Iterator(
        GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>*
            TheSlabHashCtxt,
        uint32_t TheBucketId,
        uint32_t TheAllocatorAddr = ConcurrentSetT<KeyT>::A_INDEX_POINTER,
        uint32_t* ThePrevSlabNextLanePtr = nullptr)
        : IteratorBase<BucketIterator>{TheSlabHashCtxt,
                                       TheBucketId,
                                       TheAllocatorAddr,
                                       ThePrevSlabNextLanePtr} {}
  };

  using ResultT = iterator::ResultT<BucketIterator>;

  __device__ Iterator Begin() {
    return Iterator(this, 0, ConcurrentSetT<KeyT>::A_INDEX_POINTER);
  }

  __device__ Iterator End() {
    return Iterator(this, num_buckets_, ConcurrentSetT<KeyT>::A_INDEX_POINTER);
  }

  __device__ BucketIterator BeginAt(uint32_t BucketId) {
    return BucketIterator(this, BucketId, ConcurrentSetT<KeyT>::A_INDEX_POINTER);
  }

  __device__ BucketIterator EndAt(uint32_t BucketId) {
    return BucketIterator(this, BucketId, ConcurrentSetT<KeyT>::EMPTY_INDEX_POINTER);
  }

  /* Semantics of the iterator based operations
   * ~ SEARCH:
   *   - If found, return iterator to the slab where the key is found.
   *   - If not found, return the iterator to the last slab if there are
   *     free lanes available. Otherwise, return end().
   *
   * ~ INSERT:
   *   - Accepts an iterator as hint for possible insertion in the bucket.
   *   - If successful, returns an iterator for the slab where the insertion occured.
   *   - If not successful (key was found), return the iterator where the key was found,
   *     in the slab list, since the hint iterator's original position.
   *
   * ~ DELETE:
   *   - Accepts an iterator as hint for possible deletion in the bucket.
   *   - Returns the same iterator if the key was found; returns end() otherwise.
   */

  __device__ __forceinline__ ResultT
  insertKey(bool& to_be_inserted,
            const uint32_t& laneId,
            const KeyT& myKey,
            BucketIterator& Iterator,
            typename AllocPolicy::AllocatorContextT& local_allocator_context);

  __device__ __forceinline__ ResultT searchKey(bool& to_be_searched,
                                               const uint32_t& laneId,
                                               const KeyT& myKey,
                                               BucketIterator& Iterator);

  __device__ __forceinline__ ResultT deleteKey(bool& ToDelete,
                                               const uint32_t& LaneID,
                                               KeyT& TheKey,
                                               BucketIterator& Iterator);

 private:
  // this function should be operated in a warp-wide fashion
  // TODO: add required asserts to make sure this is true in tests/debugs
  __device__ __forceinline__ SlabAllocAddressT allocateSlab(const uint32_t& laneId) {
    return global_allocator_ctx_.warpAllocate(laneId);
  }

  __device__ __forceinline__ SlabAllocAddressT
  allocateSlab(typename AllocPolicy::AllocatorContextT& local_allocator_ctx,
               const uint32_t& laneId) {
    return local_allocator_ctx.warpAllocate(laneId);
  }

  // a thread-wide function to free the slab that was just allocated
  __device__ __forceinline__ void freeSlab(const SlabAllocAddressT slab_ptr) {
    global_allocator_ctx_.freeUntouched(slab_ptr);
  }

  // === members:
  uint32_t num_buckets_;
  uint32_t hash_x_;
  uint32_t hash_y_;
  typename ConcurrentSetT<KeyT>::SlabTypeT* d_table_;
  // a copy of dynamic allocator's context to be used on the GPU
  typename AllocPolicy::AllocatorContextT global_allocator_ctx_;
};

/*
 * This class owns the allocated memory for the hash table
 */
template <typename KeyT, typename ValueT, typename AllocPolicy>
class GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet> {
 private:
  // fixed known parameters:
  static constexpr uint32_t BLOCKSIZE_ = 128;
  static constexpr uint32_t WARP_WIDTH_ = 32;
  static constexpr uint32_t PRIME_DIVISOR_ = 4294967291u;

  struct hash_function {
    uint32_t x;
    uint32_t y;
  } hf_;

  // total number of buckets (slabs) for this hash table
  uint32_t num_buckets_;

  // a raw pointer to the initial allocated memory for all buckets
  int8_t* d_table_;
  size_t slab_unit_size_;  // size of each slab unit in bytes (might differ
                           // based on the type)

  // slab hash context, contains everything that a GPU application needs to be
  // able to use this data structure
  GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>
      gpu_context_;

  // const pointer to an allocator that all instances of slab hash are going to
  // use. The allocator itself is not owned by this class
  typename AllocPolicy::DynamicAllocatorT* dynamic_allocator_;
  uint32_t device_idx_;

 public:
  GpuSlabHash(const uint32_t num_buckets,
              typename AllocPolicy::DynamicAllocatorT* dynamic_allocator,
              uint32_t device_idx,
              const time_t seed = 0,
              const bool identity_hash = false)
      : num_buckets_(num_buckets)
      , d_table_(nullptr)
      , slab_unit_size_(0)
      , dynamic_allocator_(dynamic_allocator)
      , device_idx_(device_idx) {
    assert(dynamic_allocator && "No proper dynamic allocator attached to the slab hash.");
    assert(sizeof(typename ConcurrentSetT<KeyT>::SlabTypeT) ==
               (WARP_WIDTH_ * sizeof(uint32_t)) &&
           "A single slab on a ConcurrentMap should be 128 bytes");
    int32_t devCount = 0;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&devCount));
    assert(device_idx_ < devCount);

    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));

    slab_unit_size_ =
        GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentMap>::
            getSlabUnitSize();

    // allocating initial buckets:
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_table_, slab_unit_size_ * num_buckets_));

    CHECK_CUDA_ERROR(cudaMemset(d_table_, 0xFF, slab_unit_size_ * num_buckets_));

    // creating a random number generator:
    if (!identity_hash) {
      std::mt19937 rng(seed ? seed : time(0));
      hf_.x = rng() % PRIME_DIVISOR_;
      if (hf_.x < 1)
        hf_.x = 1;
      hf_.y = rng() % PRIME_DIVISOR_;
    } else {
      hf_ = {0u, 0u};
    }

    // initializing the gpu_context_:
    gpu_context_.initParameters(
        num_buckets_, hf_.x, hf_.y, d_table_, dynamic_allocator_->getContextPtr());
  }

  ~GpuSlabHash() {
  // TODO: Inspect CUDA Error Invalid Argument
  #if 0
    CHECK_CUDA_ERROR(cudaSetDevice(device_idx_));
    CHECK_CUDA_ERROR(cudaFree(d_table_));
  #endif
  }

  // returns some debug information about the slab hash
  std::string to_string();

  double computeLoadFactor(int flag);

  GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentSet>&
  getSlabHashContext() {
    return gpu_context_;
  }

  void buildBulk(KeyT* d_key, ValueT* d_value, uint32_t num_keys);
  void searchIndividual(KeyT* d_query, ValueT* d_result, uint32_t num_queries);
  void searchBulk(KeyT* d_query, ValueT* d_result, uint32_t num_queries) {}
  void deleteIndividual(KeyT* d_key, uint32_t num_keys);
};
