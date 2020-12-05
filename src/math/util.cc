#pragma once

#include <cstdlib>

namespace mlcc {

r32
ScaleRange(r32 v, r32 smin, r32 smax, r32 tmin, r32 tmax)
{
  return (((v - smin) * (tmax - tmin)) / (smax - smin)) + tmin;
}

r32
ScaleRange(r32 v, r32 smax, r32 tmax)
{
  return ((v) * (tmax)) / (smax);
}

r32
Random(r32 min, r32 max)
{
  return ScaleRange((r32)rand() / RAND_MAX, 0.f, 1.f, min, max);
}

template <typename T>
T
Max(T x, T y)
{
  return x > y ? x : y;
}

template <typename T>
T
Min(T x, T y)
{
  return x < y ? x : y;
}

// Goes from 0 to 360.
r32
Atan2(r32 y, r32 x)
{
  r32 angle = atan2(y, x) * 180.f / PI;
  if (angle < 0.f) angle += 360.f;
  return angle;
}

}
