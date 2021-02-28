#include "math/math.cc"

#include <cmath>
#include <cstdio>

void Softmax(const std::vector<r32*>& v) {
  std::vector<r32> e(v.size());
  r32 sum = 0;
  for (u32 i = 0; i < e.size(); ++i) {
    e[i] = exp(*v[i]);
    sum += e[i];
  }
  for (u32 i = 0; i < v.size(); ++i) {
    *v[i] = e[i] / sum;
  }
}

int main(int argc, char** argv) {
  INIT_MATRIX(m, 3, 3, (
    5.f, 4.f, 2.f,
    4.f, 2.f, 8.f,
    4.f, 4.f, 1.f
  ));
  m.ApplyToRows(Softmax);
  m.DebugPrint();
  return 0;
}
