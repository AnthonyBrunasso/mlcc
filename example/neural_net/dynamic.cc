#include "ml/nnet.cc"

int main(int argc, char** argv) {
  mlcc::NNet nn;
  nn.AddLayer({
    .size = 2,
    .use_bias = true
  });
  nn.AddLayer({
    .size = 1,
    .activation = mlcc::NNLayer::kActivationSigmoid
  });
  INIT_MATRIX(in, 2, 4, (
    0.0f, 1.0f, 0.0f, 1.f,
    0.0f, 0.0f, 1.0f, 1.f
  ));
  INIT_MATRIX(y, 1, 4, ( 0.f, 0.f, 0.f, 1.f ));
  //in.DebugPr

//  nn.FeedForward(in).DebugPrint();

#if 1
  for (int i = 0; i < 10000; ++i) { 
    //printf("1\n");
    std::vector<mlcc::Matrix> weight_delta;
    nn.BackProp(in, y, &weight_delta);
    //printf("2\n");
    //weight_delta[0] /= 4.f;
    //printf("WEIGHTS:\n");nn.weight_[0].DebugPrint();
    //printf("WEIGHTDELTA:\n");weight_delta[0].DebugPrint();
    //weight_delta[0] *= 0.1f;
    //weight_delta[0].DebugPrint();
    //printf("RESULT:\n");
    nn.weight_[0] -= weight_delta[0] * 0.1f;
    nn.FeedForward(in).DebugPrint();
  }
#endif

/*
  for (int i = 0; i < weight_delta.size(); ++i) {
    printf("weight_delta %i:\n", i);
    weight_delta[i].DebugPrint();
  }

  for (int i = 0; i < nn.weights_.size(); ++i) {
    printf("weight_matrix %i\n", i);
    nn.weights_[i].DebugPrint();
  }
  */

  return 0;
}
