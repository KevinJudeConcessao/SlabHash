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

#ifndef PCMAP_CLASS_CUH_
#define PCMAP_CLASS_CUH_

#include <cassert>
#include <ctime>
#include <limits>
#include <random>

template <typename KeyT, typename ValueT, typename AllocPolicy>
class GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>;

/* This is the main class that will be shallowly copied into the device to be
 * used at runtime. This class does not own the allocated memory on the GPU
 */

template <typename KeyT, typename ValueT, typename AllocPolicy>
class GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap> {
 public:
  static constexpr uint32_t PrimeDivisor = 4294967291u;
  static constexpr uint32_t WarpWidth = 32;

  __host__ __device__ GpuSlabHashContext()
      : NumberOfBuckets{}
      , HashX{}
      , HashY{}
      , BucketHeadSlabs{}
      , BucketHeadSlabsValue{}
      , BucketHeadSlabsMutex{} {}

  __host__ __device__ GpuSlabHashContext(
      const GpuSlabHashContext<KeyT,
                               ValueT,
                               AllocPolicy,
                               SlabHashTypeT::PhaseConcurrentMap>& TheContext)
      : NumberOfBuckets{TheContext.NumberOfBuckets}
      , HashX{TheContext.HashX}
      , HashY{TheContext.HashY}
      , BucketHeadSlabs{TheContext.BucketHeadSlabs}
      , BucketHeadSlabsValue{TheContext.BucketHeadSlabsValue}
      , BucketHeadSlabsMutex{TheContext.BucketHeadSlabsMutex}
      , TheAllocatorContext{TheContext.TheAllocatorContext} {}

  __host__ __device__ ~GpuSlabHashContext() {}

  static constexpr size_t getSlabUnitSize() {
    return sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::SlabTypeT);
  }

  static constexpr size_t getKeySlabUnitSize() {
    return sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::KeySlabTypeT);
  }

  static constexpr size_t getValueSlabUnitSize() {
    return sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::ValueSlabTypeT);
  }

  static std::string getSlabHashTypeName() {
    return PhaseConcurrentMapT<KeyT, ValueT>::getTypeName();
  }

  __host__ void initParameters(
      const uint32_t NumBuckets,
      uint32_t HashX,
      uint32_t HashY,
      int8_t* DeviceHeadSlabs,
      typename AllocPolicy::AllocatorContextT* TheAllocatorContext) {
    this->NumberOfBuckets = NumBuckets;
    this->HashX = HashX;
    this->HashY = HashY;
    this->BucketHeadSlabs =
        reinterpret_cast<typename PhaseConcurrentMapT<KeyT, ValueT>::SlabTypeT*>(
            DeviceHeadSlabs);
    this->TheAllocatorContext = *TheAllocatorContext;
  }

  __device__ __host__ __forceinline__ typename AllocPolicy::AllocatorContextT&
  getAllocatorContext() {
    return TheAllocatorContext;
  }

  __device__ __host__ __forceinline__
      typename PhaseConcurrentMapT<KeyT, ValueT>::SlabTypeT*
      getDeviceTablePointer() {
    return BucketHeadSlabs;
  }

  __device__ __host__ __forceinline__ uint32_t getNumBuckets() { return NumberOfBuckets; }
  __device__ __host__ __forceinline__ uint32_t getHashX() { return HashX; }
  __device__ __host__ __forceinline__ uint32_t getHashY() { return HashY; }

  __device__ __host__ __forceinline__ uint32_t computeBucket(const KeyT& TheKey) const {
    return (((HashX ^ TheKey) + HashY) % PrimeDivisor) % NumberOfBuckets;
  }

  __device__ __forceinline__ void insertPair(
      bool& ToBeInserted,
      const uint32_t& LaneID,
      const KeyT& TheKey,
      const ValueT& TheValue,
      const uint32_t BucketID,
      typename AllocPolicy::AllocatorContextT& TheAllocatorContext);

  __device__ __forceinline__ void insertPairUnique(
      bool& ToBeInserted,
      const uint32_t& LaneID,
      const KeyT& TheKey,
      const ValueT& TheValue,
      const uint32_t BucketID,
      typename AllocPolicy::AllocatorContextT& TheAllocatorContext);

  template <typename FilterMapTy>
  __device__ __forceinline__ void updatePair(bool& ToBeUpdated,
                                             const uint32_t& LaneID,
                                             const KeyT& TheKey,
                                             const ValueT& TheValue,
                                             const uint32_t BucketID,
                                             FilterMapTy* FilterMaps = nullptr);

  template <typename FilterTy>
  __device__ __forceinline__ UpsertStatusKind
  upsertPair(bool& ToBeUpserted,
             const uint32_t& LaneID,
             const KeyT& TheKey,
             const ValueT& TheValue,
             const uint32_t BucketID,
             typename AllocPolicy::AllocatorContextT& TheAllocatorContext,
             FilterTy* Filters = nullptr);

  __device__ __forceinline__ void searchKey(bool& ToBeSearched,
                                            const uint32_t& LaneID,
                                            const KeyT& TheKey,
                                            ValueT& TheValue,
                                            const uint32_t BucketID);

  __device__ __forceinline__ void searchKeyBulk(const uint32_t& LaneID,
                                                const KeyT& TheKey,
                                                ValueT& TheValue,
                                                const uint32_t BucketID);

  __device__ __forceinline__ void countKey(bool& ToBeSearched,
                                           const uint32_t& LaneID,
                                           const KeyT& TheKey,
                                           uint32_t& TheCount,
                                           const uint32_t BucketID);

  __device__ __forceinline__ bool deleteKey(bool& ToBeDeleted,
                                            const uint32_t& LaneID,
                                            const KeyT& TheKey,
                                            const uint32_t BucketID);

  __device__ __forceinline__ uint32_t* getPointerFromSlab(
      const SlabAddressT& TheSlabAddress,
      uint32_t LaneID) {
    return TheAllocatorContext.getPointerFromSlab(TheSlabAddress, LaneID);
  }

  __device__ __forceinline__ uint32_t* getPointerFromBucket(const uint32_t BucketID,
                                                            const uint32_t LaneID) {
    return reinterpret_cast<uint32_t*>(&BucketHeadSlabs[BucketID]) + LaneID;
  }

  using Iterator = iterator::SlabIterator<PhaseConcurrentMapPolicy<KeyT, AllocPolicy>>;
  using BucketIterator =
      iterator::BucketIterator<PhaseConcurrentMapPolicy<KeyT, AllocPolicy>>;

  __device__ Iterator Begin() {
    return Iterator(*this, 0, PhaseConcurrentMapT::A_INDEX_POINTER);
  }

  __device__ Iterator End() {
    return Iterator(*this, NumberOfBuckets, PhaseConcurrentMapT::A_INDEX_POINTER);
  }

  __device__ BucketIterator BeginAt(uint32_t BucketId) {
    return BucketIterator(*this, BucketId, PhaseConcurrentMapT::A_INDEX_POINTER);
  }

  __device__ BucketIterator EndAt(uint32_t BucketId) {
    return BucketIterator(*this, BucketId, PhaseConcurrentMapT::EMPTY_INDEX_POINTER);
  }

 private:
  friend class GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>;

  uint32_t NumberOfBuckets;
  uint32_t HashX;
  uint32_t HashY;

  typename PhaseConcurrentMapT<KeyT, ValueT>::SlabTypeT* BucketHeadSlabs;
  typename PhaseConcurrentMapT<KeyT, ValueT>::ValueSlabTypeT* BucketHeadSlabsValue;
  typename PhaseConcurrentMapT<KeyT, ValueT>::MutexSlabTypeT* BucketHeadSlabsMutex;

  typename AllocPolicy::AllocatorContextT TheAllocatorContext;

 private:
  __device__ __forceinline__ SlabAllocAddressT allocateSlab(const uint32_t& LaneID) {
    return TheAllocatorContext.warpAllocate(LaneID);
  }

  __device__ __forceinline__ SlabAllocAddressT
  allocateSlab(typename AllocPolicy::AllocatorContextT& TheAllocator,
               const uint32_t& LaneID) {
    return TheAllocator.warpAllocate(LaneID);
  }

  __device__ __forceinline__ void freeSlab(const SlabAllocAddressT SlabPtr) {
    TheAllocatorContext.freeUntouched(SlabPtr);
  }
};

template <typename KeyT, typename ValueT, typename AllocPolicy>
class GpuSlabHash<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap> {
 public:
  std::string to_string();
  double computeLoadFactor(int Flag);

  void buildBulk(KeyT* KeysDevPtr, ValueT* ValuesDevPtr, uint32_t NumberOfKeys);
  void buildBulkWithUniqueKeys(KeyT* KeysDevPtr,
                               ValueT* ValuesDevPtr,
                               uint32_t NumberOfKeys);

  void searchIndividual(KeyT* KeyQueriesDevPtr,
                        ValueT* ResultsDevPtr,
                        uint32_t NumberOfQueries);
  void searchBulk(KeyT* KeyQueriesDevPtr,
                  ValueT* ResultsDevPtr,
                  uint32_t NumberOfQueries);

  template <typename FilterMapTy>
  void updateBulk(KeyT* KeysDevPtr,
                  ValueT* ValuesDevPtr,
                  uint32_t NumberOfKeys,
                  FilterMapTy* FilterMaps = nullptr);

  template <typename FilterTy>
  void upsertBulk(KeyT* KeysDevPtr,
                  ValueT* ValuesDevPtr,
                  uint32_t NumberOfKeys,
                  FilterTy* Filters = nullptr);

  void deleteIndividual(KeyT* KeysDevPtr, uint32_t NumberOfKeys);

  void batchedOperation(KeyT* KeysDevPtr,
                        ValueT* ResultsDevPtr,
                        uint32_t NumberOfOperations);

  void countIndividual(KeyT* KeyQueriesDevPtr,
                       uint32_t* CountDevPtr,
                       uint32_t NumberOfQueries);

 private:
  static constexpr uint32_t PrimeDivisor = 4294967291u;
  static constexpr uint32_t BlockSize = 128;
  static constexpr uint32_t WarpWidth = 32;

  uint32_t NumberOfBuckets;

  typename AllocPolicy::DynamicAllocatorT* TheAllocator;
  uint32_t DeviceIndex;
  uint8_t* BucketHeadSlabs;
  uint8_t* BucketHeadSlabsValue;
  uint8_t* BucketHeadSlabsMutex;

  struct {
    uint32_t HashX;
    uint32_t HashY;
  } HashFunctionParameters;

  GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
      GpuContext;

 public:
  GpuSlabHash(const uint32_t NumberOfBuckets,
              typename AllocPolicy::DynamicAllocatorT* TheAllocator,
              uint32_t DeviceIndex,
              const time_t Seed = 0,
              const bool IdentityHash = false)
      : NumberOfBuckets(NumberOfBuckets)
      , TheAllocator(TheAllocator)
      , DeviceIndex(DeviceIndex)
      , BucketHeadSlabs{nullptr}
      , BucketHeadSlabsValue{nullptr}
      , BucketHeadSlabsMutex{nullptr}
      , HashFunctionParameters{0u, 0u}
      , GpuContext{} {
    int DeviceCount = 0;

    assert(TheAllocator && "The allocator is NULL");
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&DeviceCount));
    assert(DeviceIndex < DeviceCount && "Device Index >= Device Count");
    CHECK_CUDA_ERROR(cudaSetDevice(DeviceIndex));

    constexpr size_t KeySlabSize =
        sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::SlabTypeT);
    constexpr size_t ValueSlabSize =
        sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::ValueSlabTypeT);
    constexpr size_t MutexSlabSize =
        sizeof(typename PhaseConcurrentMapT<KeyT, ValueT>::MutexSlabTypeT);

    static_assert(KeySlabSize == 32 * sizeof(uint32_t));
    static_assert(ValueSlabSize == 32 * sizeof(uint32_t));
    static_assert(MutexSlabSize == 32 * sizeof(uint32_t));

    CHECK_CUDA_ERROR(cudaMalloc(&BucketHeadSlabs, KeySlabSize * NumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMalloc(&BucketHeadSlabsValue, ValueSlabSize * NumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMalloc(&BucketHeadSlabsMutex, MutexSlabSize * NumberOfBuckets));

    CHECK_CUDA_ERROR(cudaMemset(BucketHeadSlabs, 0xFF, KeySlabSize * NumberOfBuckets));
    CHECK_CUDA_ERROR(
        cudaMemset(BucketHeadSlabsValue, 0xFF, ValueSlabSize * NumberOfBuckets));
    CHECK_CUDA_ERROR(
        cudaMemset(BucketHeadSlabsMutex, 0xFF, MutexSlabSize * NumberOfBuckets));

    std::random_device RandomDevice;
    std::mt19937 RandomNumberGenerator(RandomDevice());
    std::uniform_int_distribution<uint32_t> Distribution(
        1, std::numeric_limits<uint32_t>::max());

    HashFunctionParameters = IdentityHash ? {0u, 0u} : {Distribution(), Distribution()};

    GpuContext.NumberOfBuckets = NumberOfBuckets;
    GpuContext.HashX = HashFunctionParameters.HashX;
    GpuContext.HashY = HashFunctionParameters.HashY;
    GpuContext.BucketHeadSlabs = reintepret_cast<> BucketHeadSlabs;
    GpuContext.BucketHeadSlabsValue = BucketHeadSlabsValue;
    GpuContext.BucketHeadSlabsMutex = BucketHeadSlabsMutex;
    GpuContext.TheAllocatorContext = *TheAllocator->getAllocatorContext();
  }
};

#endif  // PCMAP_CLASS_CUH_
