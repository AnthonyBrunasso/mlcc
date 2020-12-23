#include <cstdio>
#include <vector>

#include "math/math.cc"

float kTrainingRate = 0.1f;

struct TrainingInput {
  float x1;
  float x2;
  float y;
};

struct Weights {
  float w1;
  float w2;
  float b;
};

Weights RandomWeights() {
  Weights w;
  w.w1 = mlcc::Random(-1.f, 1.f);
  w.w2 = mlcc::Random(-1.f, 1.f);
  w.b = mlcc::Random(-1.f, 1.f);
  return w;
}

float CalculatePartial(float x, float predicted, float actual) {
  return -2.f * x * (actual - predicted);
}

void TrainOnce(const std::vector<TrainingInput>& training_input, Weights* w,
               bool with_bias) {
  float dw1 = 0.0f;
  float dw2 = 0.0f;
  float db = 0.f;
  for (const auto& input : training_input) {
    float res = input.x1 * w->w1 + input.x2 * w->w2 + w->b;
    if (with_bias) {
      res = input.x1 * w->w1 + input.x2 * w->w2 + w->b;
    } else {
      res = input.x1 * w->w1 + input.x2 * w->w2;
    }
    dw1 += CalculatePartial(input.x1, res, input.y);
    dw2 += CalculatePartial(input.x2, res, input.y);
    db += CalculatePartial(w->b, res, input.y);
    printf("x1=%.2f x2=%.2f w1=%.2f w2=%.2f actual=%.2f predicted=%.2f\n",
           input.x1, input.x2, w->w1, w->w2, input.y, res);
  }
  dw1 /= training_input.size();
  dw2 /= training_input.size();
  db /= training_input.size();

  if (with_bias) {
    printf("dw1=%.4f dw2=%.4f db=%.4f\n", dw1, dw2, db);
  } else {
    printf("dw1=%.4f dw2=%.4f\n", dw1, dw2);
  }

  w->w1 -= kTrainingRate * dw1;
  w->w2 -= kTrainingRate * dw2;
  w->b -= kTrainingRate * db;
}

int main(int argc, char** argv) {
#if 1
  std::vector<TrainingInput> and_input {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 1.0},
  };

  Weights and_weights = RandomWeights();
  for (int i = 0; i < 1000; ++i) {
    TrainOnce(and_input, &and_weights, false);
  }
#endif

#if 0
  std::vector<TrainingInput> or_input {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 1.0},
    {0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0},
  };

  Weights or_weights = RandomWeights();
  for (int i = 0; i < 1000; ++i) {
    TrainOnce(or_input, &or_weights, true);
  }
#endif

  return 0;
}
