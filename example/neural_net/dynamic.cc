#include "ml/nnet.cc"

int main(int argc, char** argv) {
  mlcc::NNet nn;
  nn.AddLayer(2);
  nn.AddLayer(1);
  INIT_MATRIX(in, 2, 4, (
    0.0f, 1.0f, 0.0f, 1.f,
    0.0f, 0.0f, 1.0f, 1.f
  ));
  INIT_MATRIX(y, 1, 4, ( 0.f, 0.f, 0.f, 1.f ));
  //in.DebugPr
  for (int i = 0; i < 5; ++i) { 
    std::vector<mlcc::Matrix> weight_delta;
    nn.BackProp(in, y, &weight_delta);
    weight_delta[0] /= 4.f;
    weight_delta[0] *= 0.1f;
    printf("WEIGHTS:\n");nn.weights_[0].DebugPrint();
    printf("WEIGHTDELTA:\n");weight_delta[0].DebugPrint();
    nn.weights_[0] -= weight_delta[0];
    //weight_delta[0].DebugPrint();
    printf("RESULT:\n");
    nn.FeedForward(in).DebugPrint();
  }

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
