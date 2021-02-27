#include "math/math.cc"

#include <cstdio>

mlcc::Matrix Predict(const mlcc::Matrix& x, const mlcc::Matrix& w, r32 b) {
  return x * w + b;
}

mlcc::Matrix Relu(const mlcc::Matrix& x) {
  mlcc::Matrix m = x;
  for (u32 i = 0; i < m.data.size(); ++i) {
    if (m.data[i] > 0.f) continue;
    else m.data[i] = 0.f;
  }
  return m;
}

r32 MeanSquaredError(const mlcc::Matrix& a, const mlcc::Matrix& b) {
  assert(a.rows == b.rows && a.cols == b.cols && a.cols == 1);
  r32 res = 0.f;
  for (u32 i = 0; i < a.rows; ++i) {
    r32 diff = a.val(i, 0) - b.val(i, 0);
    diff = diff * diff;
    res += diff;
  }
  res /= a.rows;
  return res;
}

int main(int argc, char** argv) {
  INIT_MATRIX(xor_input, 4, 2, (
      0.f, 0.f,
      1.f, 0.f,
      0.f, 1.f,
      1.f, 1.f
  ));
  INIT_MATRIX(xor_output, 4, 1, (
      0.f,
      1.f,
      1.f,
      0.f
  ));
  INIT_MATRIX(weights_one, 2, 2, (
    1.f, 1.f,
    1.f, 1.f
  ));
  INIT_MATRIX(bias, 1, 2, (
    0.f, -1.f
  ));
  INIT_MATRIX(weights_two, 2, 1, (
    1.f,
    -2.f
  ));

  (Relu((xor_input * weights_one + bias)) * weights_two).DebugPrint();

  return 0;
}
