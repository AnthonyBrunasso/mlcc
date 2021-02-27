#include "math/math.cc"

#include <cstdio>

mlcc::Matrix Predict(const mlcc::Matrix& x, const mlcc::Matrix& w, r32 b) {
  return x * w + b;
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
  INIT_MATRIX(weights, 2, 1, (
      1.f, 1.f
  ));
  r32 b = 0.f;

  xor_input.DebugPrint();
  weights.DebugPrint();
  printf("Predicted Output\n");
  Predict(xor_input, weights, b).DebugPrint();
  printf("Actual Output\n");
  xor_output.DebugPrint();
  printf("MSE: %.2f\n", MeanSquaredError(xor_output, Predict(xor_input, weights, b)));
  return 0;
}
