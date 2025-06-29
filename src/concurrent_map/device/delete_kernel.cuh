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

template <typename KeyT, typename ValueT, typename AllocPolicy>
__global__ void delete_table_keys(
    KeyT* d_key_deleted,
    uint32_t num_keys,
    GpuSlabHashContext<KeyT, ValueT, AllocPolicy, SlabHashTypeT::ConcurrentMap>
        slab_hash) {
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  uint32_t laneId = threadIdx.x & 0x1F;

  if ((tid - laneId) >= num_keys) {
    return;
  }

  KeyT myKey = 0;
  uint32_t myBucket = 0;
  bool to_delete = false;

  if (tid < num_keys) {
    myKey = d_key_deleted[tid];
    myBucket = slab_hash.computeBucket(myKey);
    to_delete = true;
  }

  // delete the keys:
  slab_hash.deleteKey(to_delete, laneId, myKey, myBucket);
}