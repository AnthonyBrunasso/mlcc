#include "math/math.cc"

#include <cassert>
#include <cstdio>
#include <vector>

r32 f1(r32 x) {
  return -x;
}

r32 f2(r32 x) {
  return x * x;
}

r32 Covariance(const std::vector<r32>& a, const std::vector<r32>& b) {
  assert(a.size() == b.size());
  u32 n = a.size();
  r32 sum = 0.f;
  r32 mean_a = 0.f;
  for (auto x : a) mean_a += x;
  mean_a /= n;
  r32 mean_b = 0.f;
  for (auto x : b) mean_b += x;
  mean_b /= n;
  for (u32 i = 0; i < a.size(); ++i) {
    sum += (a[i] - mean_a) * (b[i] - mean_b);
  }
  return sum / n;
}

int main(int argc, char** argv) {
  std::vector<r32> v1;
  std::vector<r32> v2;
  u32 N = 100;
  v1.reserve(N);
  v2.reserve(N);
  for (u32 i = 0; i < N; ++i) {
    v1.push_back(f1((r32)i));
    v2.push_back(f2((r32)i));
  }
  printf("Covariance: %.2f\n", Covariance(v1, v2));
  return 0;
}
