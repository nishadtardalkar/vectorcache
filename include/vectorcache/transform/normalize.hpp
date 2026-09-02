#pragma once

#include <span>

namespace vectorcache::transform {

/// L2-normalize vector in place. Zero vectors are left unchanged.
void l2_normalize_in_place(std::span<float> vector);

}  // namespace vectorcache::transform
