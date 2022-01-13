#ifndef SLAB_POLICIES_CUH_
#define SLAB_POLICIES_CUH_

template <typename KeyTy,
          typename ValueTy,
          typename SlabAllocPolicyTy,
          typename SlabHashTy,
          typename SlabHashContextTy,
          typename SlabInfoTy>
struct ContainerPolicy {
  using KeyT = KeyTy;
  using ValueT = ValueTy;
  using AllocPolicyT = SlabAllocPolicyTy;
  using SlabHashT = SlabHashTy;
  using SlabHashContextT = SlabHashContextTy;
  using SlabInfoT = SlabInfoTy;
};

template <typename KeyTy, typename SlabAllocPolicyTy>
using ConcurrentSetPolicy = ContainerPolicy<
    KeyTy,
    void,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, KeyTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentSet>,
    GpuSlabHashContext<KeyTy, KeyTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentSet>,
    ConcurrentSetT<KeyTy>>;

template <typename KeyTy, typename ValueTy, typename SlabAllocPolicyTy>
using ConcurrentMapPolicy = ContainerPolicy<
    KeyTy,
    ValueTy,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentMap>,
    GpuSlabHashContext<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::ConcurrentMap>,
    ConcurrentMapT<KeyTy, ValueTy>>;

template <typename KeyTy, typename ValueTy, typename SlabAllocPolicyTy>
using PhaseConcurrentMapPolicy = ContainerPolicy<
    KeyTy,
    ValueTy,
    SlabAllocPolicyTy,
    GpuSlabHash<KeyTy, ValueTy, SlabAllocPolicyTy, SlabHashTypeT::PhaseConcurrentMap>,
    GpuSlabHashContext<KeyTy,
                       ValueTy,
                       SlabAllocPolicyTy,
                       SlabHashTypeT::PhaseConcurrentMap>,
    PhaseConcurrentMapT<KeyTy, ValueTy>>;

#endif  // SLAB_POLICIES_CUH_
