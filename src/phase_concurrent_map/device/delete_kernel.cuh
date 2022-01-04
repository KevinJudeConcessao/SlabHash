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

#ifndef PCMAP_DELETE_KERNEL_H_
#define PCMAP_DELETE_KERNEL_H_

template <typename KeyT, typename ValueT, typename AllocPolicy>
__global__ void delete_table_keys(
    KeyT* TheKeys,
    uint32_t NumberOfKeys,
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::PhaseConcurrentMap>
        SlabHashCtxt) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

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

#endif  // PCMAP_DELETE_KERNEL_H_
