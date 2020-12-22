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
  float w1 = 0.5f;
  float w2 = 0.5f;
};

Weights RandomWeights() {
  Weights w;
  w.w1 = mlcc::Random(-1.f, 1.f);
  w.w2 = mlcc::Random(-1.f, 1.f);
  return w;
}

float CalculatePartial(float x, float predicted, float actual) {
  return -2.f * x * (actual - predicted);
}

void TrainOnce(const std::vector<TrainingInput>& training_input, Weights* w) {
  float dw1 = 0.0f;
  float dw2 = 0.0f;
  for (const auto& input : training_input) {
    float res = input.x1 * w->w1 + input.x2 * w->w2;
    dw1 += CalculatePartial(input.x1, res, input.y);
    dw2 += CalculatePartial(input.x2, res, input.y);
    printf("x1=%.2f x2=%.2f w1=%.2f w2=%.2f actual=%.2f predicted=%.2f\n",
           input.x1, input.x2, w->w1, w->w2, input.y, res);
  }
  printf("dw1=%.4f dw2=%.4f\n", dw1 / training_input.size(), dw2 / training_input.size());

  w->w1 -= kTrainingRate * dw1;
  w->w2 -= kTrainingRate * dw2;
}

int main(int argc, char** argv) {
#if 0
  std::vector<TrainingInput> and_input {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {1.0, 1.0, 1.0},
  };

  Weights and_weights = RandomWeights();
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
  TrainOnce(and_input, &and_weights);
#endif

#if 1
  std::vector<TrainingInput> or_input {
    {0.0, 0.0, 0.0},
    {1.0, 0.0, 1.0},
    {0.0, 1.0, 1.0},
    {1.0, 1.0, 1.0},
  };

  Weights or_weights = RandomWeights();
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
  TrainOnce(or_input, &or_weights);
#endif

  return 0;
}
