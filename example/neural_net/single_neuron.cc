#include "math/math.cc"

#include <cstdio>
#include <cmath>

r32 Sigmoid(r32 x) {
  return (1.f / (1.f + exp(-x)));
}

r32 SigmoidPrime(r32 x) {
  r32 s = Sigmoid(x);
  return s * (1.f - s);
}

r32 Predict(r32 x, r32 w, r32 b) {
  return Sigmoid(x * w + b);
}

r32 Cost(r32 ytrue, r32 ypredict) {
  // Dividing by 2 simply makes the derivative easier to calculate due to the
  // square term being brought down to cancel with the denominator.
  return ((ytrue - ypredict) * (ytrue - ypredict)) / 2.f;
}

// (ytrue - sigmoid(x * w + b))^2 / 2
// (d/dw)(ytrue - sigmoid(x * w + b)^2 / 2
// ((ytrue - sigmoid(x * w + b)) * d/dw(ytrue - sigmoid(x * w + b)))
// ((ytrue - sigmoid(x * w + b)) * (0 - sigmoid_prime(x * w + b) * x))
// Let:
// z = x * w + b
// a = sigmoid(x * w + b)
// a = sigmoid(z)
// ((ytrue - a) * -sigmoid_prime(z) * x)
// Multiply through by -
// ((-ytrue + a) * sigmoid_prime(z) * x)
// (a - ytrue) * sigmoid_prime(z) * x
r32 CostDerivative_W(r32 ytrue, r32 x, r32 w, r32 b) {
  r32 a = Predict(x, w, b);
  r32 z = x * w + b;
  return (a - ytrue) * SigmoidPrime(z) * x;
}

// (ytrue - sigmoid(x * w + b))^2 / 2
// (d/db)(ytrue - sigmoid(x * w + b)^2 / 2
// ((ytrue - sigmoid(x * w + b)) * d/db(ytrue - sigmoid(x * w + b)))
// ((ytrue - sigmoid(x * w + b)) * (0 - sigmoid_prime(x * w + b) * 1))
// Let:
// z = x * w + b
// a = sigmoid(x * w + b)
// a = sigmoid(z)
// ((ytrue - a) * -sigmoid_prime(z))
// Multiply through by -
// ((-ytrue + a) * sigmoid_prime(z))
// (a - ytrue) * sigmoid_prime(z)
r32 CostDerivative_b(r32 ytrue, r32 x, r32 w, r32 b) {
  r32 a = Predict(x, w, b);
  r32 z = x * w + b;
  return (a - ytrue) * SigmoidPrime(z);
}

int main(int argc, char** argv) {
  r32 x = 1.f;
  r32 w = .60f;
  r32 b = .90f;
  r32 y = 0.f;
  r32 alpha = 0.15f;
  u32 epochs = 300;

  for (int i = 0; i < epochs; ++i) {
    r32 cd_w = CostDerivative_W(y, x, w, b);
    r32 cd_b = CostDerivative_b(y, x, w, b);
    printf("Predict: %.2f Cost: %.2f\n",
           Predict(x, w, b),
           Cost(y, Predict(x, w, b)));
    w = w - alpha * cd_w;
    b = b - alpha * cd_b;
  }

  return 0;
}
