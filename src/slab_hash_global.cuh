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

#include <cstdint>
#include <type_traits>
#include "slab_alloc.cuh"

#define CHECK_CUDA_ERROR(call)                                                          \
  do {                                                                                  \
    cudaError_t err = call;                                                             \
    if (err != cudaSuccess) {                                                           \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  } while (0)

// internal parameters for slab hash device functions:
static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;
static constexpr uint32_t EMPTY_VALUE = 0xFFFFFFFF;
static constexpr uint32_t TOMBSTONE_KEY = 0xFFFFFFFE;
static constexpr uint32_t TOMBSTONE_VALUE = 0xFFFFFFFE;
static constexpr uint64_t TOMBSTONE_PAIR_64 = 0xFFFFFFFEFFFFFFFELL;
static constexpr uint32_t DELETED_KEY = 0xFFFFFFFE;
static constexpr uint32_t DELETED_VALUE = 0xFFFFFFFE;
static constexpr uint64_t EMPTY_PAIR_64 = 0xFFFFFFFFFFFFFFFFLL;
static constexpr uint32_t WARP_WIDTH = 32;
static constexpr uint32_t SEARCH_NOT_FOUND = 0xFFFFFFFF;

// only works with up to 32-bit key/values
template <typename KeyT, typename ValueT>
struct key_value_pair {
  KeyT key;
  ValueT value;
};

template <typename KeyT, typename ValueT>
struct __align__(32) concurrent_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 15u;
  key_value_pair<KeyT, ValueT> data[NUM_ELEMENTS_PER_SLAB];
  uint32_t ptr_index[2];
};

// this slab structure is meant to be used in either concurrent sets,
// or phase-concurrent maps.
// | key 0 | key 1 | key 2 | ... | key 30 | next_ptr |
template <typename KeyT>
struct __align__(32) key_only_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 31u;
  KeyT keys[NUM_ELEMENTS_PER_SLAB];
  uint32_t next_ptr_index[1];
};

#if 0
template <typename KeyT, typename ValueT>
struct __align__(32) phase_concurrent_slab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 31u;
  // main slab (128 bytes), contain keys
  key_only_slab<KeyT> keys;

  // value storage:
  ValueT values[NUM_ELEMENTS_PER_SLAB];
};
#endif

template <typename KeyT, typename ValueT>
struct __align__(32) PhaseConcurrentKeySlab {
  static constexpr uint32_t NUM_ELEMENTS_PER_SLAB = 29u;
  KeyT Keys[NUM_ELEMENTS_PER_SLAB];
  uint32_t MutexPtrIndex[1];
  uint32_t ValuesPtrIndex[1];
  uint32_t NextPtrIndex[1];
};

template <typename KeyT, typename ValueT>
using PhaseConcurrentSlab = PhaseConcurrentKeySlab<KeyT, ValueT>;

template <typename KeyT, typename ValueT>
struct __align__(32) PhaseConcurrentValueSlab {
  ValueT Values[PhaseConcurrentKeySlab::NUM_ELEMENTS_PER_SLAB];
  uint32_t _0 [1];
  uint32_t _1 [1];
  uint32_t _2 [1];
};

/*
 * Different types of slab hash:
 * 1. Concurrent map: it assumes that all operations can be performed
 * concurrently
 * 2. phase-concurrent map: supports concurrent updates, and concurrent
 * searches, but not a mixture of both
 */
enum class SlabHashTypeT { ConcurrentMap, ConcurrentSet, PhaseConcurrentMap };

template <uint32_t LogNumMemoryBlocks, uint32_t NumSuperBlocks, uint32_t NumReplicas = 1u>
struct LightAllocatorPolicy {
  static constexpr uint32_t LogNumberOfMemoryBlocks = LogNumMemoryBlocks;
  static constexpr uint32_t NumberOfSuperBlocks = NumSuperBlocks;
  static constexpr uint32_t NumberOfReplicas = NumReplicas;

  using DynamicAllocatorT = SlabAllocLight<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
  using AllocatorContextT = SlabAllocLightContext<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
};

template <uint32_t LogNumMemoryBlocks, uint32_t NumSuperBlocks, uint32_t NumReplicas = 1u>
struct FullAllocatorPolicy {
  static constexpr uint32_t LogNumberOfMemoryBlocks = LogNumMemoryBlocks;
  static constexpr uint32_t NumberOfSuperBlocks = NumSuperBlocks;
  static constexpr uint32_t NumberOfReplicas = NumReplicas;

  using DynamicAllocatorT = SlabAlloc<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
  using AllocatorContextT = SlabAllocLight<LogNumMemoryBlocks, NumSuperBlocks, NumReplicas>;
};

template <typename KeyT, typename ValueT>
class ConcurrentMapT {
 public:
  // fixed parameters for the data structure
  static constexpr uint32_t A_INDEX_POINTER = 0xFFFFFFFE;
  static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;
  static constexpr uint32_t BASE_UNIT_SIZE = 32;
  static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x30000000;
  static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x3FFFFFFF;
  static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x15555555;
  static constexpr uint32_t NEXT_PTR_LANE = 31u;

  using SlabTypeT = concurrent_slab<KeyT, ValueT>;

  static std::string getTypeName() { return std::string("ConcurrentMap"); }
};

template <typename KeyT>
class ConcurrentSetT {
 public:
  // fixed parameters for the data structure
  static constexpr uint32_t A_INDEX_POINTER = 0xFFFFFFFD;
  static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;
  static constexpr uint32_t BASE_UNIT_SIZE = 32;
  static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x80000000;
  static constexpr uint32_t REGULAR_NODE_DATA_MASK = 0x7FFFFFFF;
  static constexpr uint32_t REGULAR_NODE_KEY_MASK = 0x7FFFFFFF;
  static constexpr uint32_t NEXT_PTR_LANE = 31u;

  using SlabTypeT = key_only_slab<KeyT>;

  static std::string getTypeName() { return std::string("ConcurrentSet"); }
};

template <typename KeyT, typename ValueT>
class PhaseConcurrentMapT {
 public:
  /* Fixed parameters for the data structure */

  static constexpr uint32_t A_INDEX_POINTER     = 0xFFFFFFFD;
  static constexpr uint32_t EMPTY_INDEX_POINTER = 0xFFFFFFFF;

  static constexpr uint32_t BASE_UNIT_SIZE  = 32;
  static constexpr uint32_t MUTEX_PTR_LANE  = 29;
  static constexpr uint32_t VALUES_PTR_LANE = 30;
  static constexpr uint32_t NEXT_PTR_LANE   = 31;

  static constexpr uint32_t REGULAR_NODE_KEY_MASK     = 0x1FFFFFFF;
  static constexpr uint32_t REGULAR_NODE_DATA_MASK    = 0x20000000;
  static constexpr uint32_t REGULAR_NODE_MUTEX_MASK   = 0x40000000;
  static constexpr uint32_t REGULAR_NODE_ADDRESS_MASK = 0x80000000;

  using KeySlabTypeT    = PhaseConcurrentKeySlab<KeyT, ValueT>;
  using ValueSlabTypeT  = PhaseConcurrentValueSlab<KeyT, ValueT>;
  using SlabTypeT       = KeySlabTypeT;

  static std::string getTypeName() { return std::string("PhaseConcurrentMap"); }
}; 

// the main class to be specialized for different types of hash tables
template <typename KeyT, typename ValueT, typename AllocPolicy, SlabHashTypeT SlabHashT>
class GpuSlabHash;

template <typename KeyT, typename ValueT, typename AllocPolicy, SlabHashTypeT SlabHashT>
class GpuSlabHashContext;

// The custom allocator that is being used for this code:
// this might need to be a template paramater itself
namespace slab_alloc_par {
constexpr uint32_t log_num_mem_blocks = 8;
constexpr uint32_t num_super_blocks = 32;
constexpr uint32_t num_replicas = 1;
}  // namespace slab_alloc_par

using DynamicAllocatorT = SlabAllocLight<slab_alloc_par::log_num_mem_blocks,
                                    slab_alloc_par::num_super_blocks,
                                    slab_alloc_par::num_replicas>;

using AllocatorContextT = SlabAllocLightContext<slab_alloc_par::log_num_mem_blocks,
                                           slab_alloc_par::num_super_blocks,
                                           slab_alloc_par::num_replicas>;

using SlabAddressT = uint32_t;
using BucketAddressT = SlabAddressT;

template <typename FilterTy>
struct FilterCheck
    : typename std::conditional<
          std::is_same<decltype(std::declval<FilterTy>()(std::declval<uint32_t>())),
                       bool>::value,
          std::true_type,
          std::false_type>::type {};

template <typename MapTy>
struct MapCheck : typename std::conditional<
                      std::is_same<decltype(std::declval<MapTy>(std::declval<uint32_t>())(
                                       std::declval<uint32_t>())),
                                   uint32_t>::value,
                      std::true_type,
                      std::false_type>::type {};

struct AlwaysTrueFilter {
  __device__ __host__ AlwaysTrueFilter() = default;
  __device__ __host__ AlwaysTrueFilter(const AlwaysTrueFilter&) = default;

  __device__ bool operator()(uint32_t) { return true; }
};

struct IdentityMap {
  __device__ __host__ IdentityMap(uint32_t Current) : CurrentValue{Current} {}
  __device__ __host__ IdentityMap(const IdentityMap&) = default;

  __device__ uint32_t operator()(uint32_t NewFieldVal) { return CurrentValue; }

 private:
  uint32_t CurrentValue;
};

enum OperationKind {
  OK_INSERT = 1,
  OK_SEARCH,
  OK_DELETE,
  OK_UPDATE,
  OK_UPSERT,
};

enum UpsertStatusKind {
  USK_FAIL,
  USK_INSERT,
  USK_UPDATE,
};