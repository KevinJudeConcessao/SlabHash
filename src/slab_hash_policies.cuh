#ifndef SLAB_POLICIES_CUH_
#define SLAB_POLICIES_CUH_

template <typename KeyTy,
          typename ValueTy,
          typename SlabAllocPolicyTy,
          typename SlabHashTy,
          typename SlabHashContextTy>
struct ContainerPolicy {
  using KeyT = KeyTy;
  using ValueT = ValueTy;
  using AllocPolicyT = SlabAllocPolicyTy;
  using SlabHashT = SlabHashTy;
  using SlabHashContextT = SlabHashContextTy;
};

template <typename KeyTy, typename ValueTy, typename SlabAllocPolicyTy>
using ConcurrentSetPolicy = ContainerPolicy<
    KeyTy,
    ValueTy,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentSet>,
    GpuSlabHashContext<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentSet>>;

template <typename KeyTy, typename ValueTy, typename SlabAllocPolicyTy>
using ConcurrentMapPolicy = ContainerPolicy<
    KeyTy,
    ValueTy,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentMap>,
    GpuSlabHashContext<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentMap>>;

template <typename KeyTy, typename ValueTy, typename SlabAllocPolicyTy>
using PhaseConcurrentMapPolicy = ContainerPolicy<
    KeyTy,
    ValueTy,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::PhaseConcurrentMap>,
    GpuSlabHashContext<KeyTy,
                       ValueTy,
                       SlabAllocPolicyTy,
                       SlabHashTypeT::PhaseConcurrentMap>>;

#endif  // SLAB_POLICIES_CUH_
