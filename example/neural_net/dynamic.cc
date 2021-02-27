#include "ml/nnet.cc"

// x x
//     x
// x x
//
// b b   

// 

int main(int argc, char** argv) {
  mlcc::NNet nn;
  nn.AddLayer({
    .size = 2,
    .use_bias = false
  });
  nn.AddLayer({
    .size = 2,
    .activation = mlcc::NNLayer::kActivationSigmoid,
    .use_bias = false
  });
  nn.AddLayer({
    .size = 1,
    .activation = mlcc::NNLayer::kActivationSigmoid
  });
  INIT_MATRIX(in, 2, 4, (
    0.0f, 1.0f, 0.0f, 1.f,
    0.0f, 0.0f, 1.0f, 1.f
  ));
  INIT_MATRIX(y, 1, 4, ( 0.f, 1.f, 1.f, 1.f ));

  // x x 
  //     x
  // x x
  //
  // 1 1

#if 1
  for (int i = 0; i < 1; ++i) { 
    std::vector<mlcc::Matrix> weight_delta;
    nn.BackProp(in, y, &weight_delta);
    for (int j = 0; j < weight_delta.size(); ++j) {
      printf("WDELTA[%i]\n", j);
      nn.weight_[j].DebugPrint();
      weight_delta[j].DebugPrint();
      nn.weight_[j] -= (weight_delta[j] * 0.1f);
    }
  }
#endif

  return 0;
}
