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
namespace cset {
template <typename KeyT, typename AllocPolicy>
__global__ void build_table_kernel(
    KeyT* d_key,
    uint32_t num_keys,
    GpuSlabHashContext<KeyT, KeyT, AllocPolicy, SlabHashTypeT::ConcurrentSet> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_keys) {
    return;
  }

  // initializing the memory allocator on each warp:
  typename AllocPolicy::AllocatorContextT local_allocator_ctx(
      slab_hash.getAllocatorContext());
  local_allocator_ctx.initAllocator(tid, laneId);

  KeyT myKey = 0;
  uint32_t myBucket = 0;
  bool to_insert = false;

  if (tid < num_keys) {
    myKey = d_key[tid];
    myBucket = slab_hash.computeBucket(myKey);
    to_insert = true;
  }

  slab_hash.insertKey(to_insert, laneId, myKey, myBucket, local_allocator_ctx);
}

//=== Individual search kernel:
template <typename KeyT, typename AllocPolicy>
__global__ void search_table(
    KeyT* d_queries,
    KeyT* d_results,
    uint32_t num_queries,
    GpuSlabHashContext<KeyT, KeyT, AllocPolicy, SlabHashTypeT::ConcurrentSet> slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_queries) {
    return;
  }

  KeyT myQuery = 0;
  uint32_t myBucket = 0;
  bool to_search = false;
  if (tid < num_queries) {
    myQuery = d_queries[tid];
    myBucket = slab_hash.computeBucket(myQuery);
    to_search = true;
  }

  bool myResult = slab_hash.searchKey(to_search, laneId, myQuery, myBucket);

  // writing back the results:
  if (tid < num_queries) {
    d_results[tid] = myResult ? myQuery : SEARCH_NOT_FOUND;
  }
}

template <typename KeyT, typename AllocPolicy>
__global__ void delete_table_keys(
    KeyT* TheKeys,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, KeyT, AllocPolicy, SlabHashTypeT::ConcurrentSet>
        SlabHashCtxt) {
  uint32_t LaneID = threadIdx.x & 0x1F;
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;

  if ((ThreadID - LaneID) >= NumberOfKeys)
    return;

  KeyT TheKey{};
  uint32_t TheBucket = 0;
  bool ToDelete = false;

  if (ThreadID < NumberOfKeys) {
    TheKey = TheKeys[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToDelete = true;
  }

  SlabHashCtxt.deleteKey(ToDelete, LaneID, TheKey, TheBucket);
}
};  // namespace cset