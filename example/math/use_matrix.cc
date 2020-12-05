#include <cstdio>

#include "math/math.cc"

int main(int argc, char** argv) {
  INIT_MATRIX(a, 2, 3, (
    1.f, 2.f, 3.f,
    4.f, 5.f, 6.f
  ));

  INIT_MATRIX(b, 3, 2, (
    1.f, 2.f,
    5.f, 6.f,
    9.f, 10.f
  ));

  mlcc::Matrix r = a * b;

  a *= 2.f;

  a.DebugPrint();

  return 0;
}
