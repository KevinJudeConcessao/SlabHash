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

#ifndef PCMAP_SEARCH_KERNELS_H_
#define PCMAP_SEARCH_KERNELS_H_

template <typename KeyT, typename ValueT>
__global__ void search_table(
    KeyT* TheKeys,
    ValueT* Results,
    uint32_t NumberOfQueries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap> SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfQueries)
    return;

  KeyT TheKey{};
  ValueT TheResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t TheBucket = 0;
  bool ToSearch = false;

  if (ThreadID < NumberOfQueries) {
    TheKey = TheKeys[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToSearch = true;
  }

  SlabHashCtxt.searchKey(ToSearch, LaneID, TheKey, TheResult, TheBucket);

  if (ThreadID < NumberOfQueries)
    KeyCounts[ThreadID] = TheResult;
}

template <typename KeyT, typename ValueT>
__global__ void search_table_bulk(
    KeyT* TheKeys,
    ValueT* Results,
    uint32_t NumberOfQueries,
    GpuSlabHashContext<KeyT, ValueT, SlabHashTypeT::PhaseConcurrentMap> SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfQueries)
    return;

  KeyT TheKey{};
  ValueT TheResult = static_cast<ValueT>(SEARCH_NOT_FOUND);
  uint32_t TheBucket = 0;
  bool ToSearch = false;

  if (ThreadID < NumberOfQueries) {
    TheKey = TheKeys[ThreadID];
    TheBucket = SlabHashCtxt.computeBucket(TheKey);
    ToSearch = true;
  }

  SlabHashCtxt.searchKey(ToSearch, LaneID, TheKey, TheResult, TheBucket);

  if (ThreadID < NumberOfQueries)
    KeyCounts[ThreadID] = TheResult;
}

#endif // PCMAP_SEARCH_KERNELS_H_
