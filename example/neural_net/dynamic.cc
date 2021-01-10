#include "ml/nnet.cc"

int main(int argc, char** argv) {
  mlcc::NNet nn;
  nn.AddLayer(2);
  nn.AddLayer(1);
  INIT_MATRIX(in, 4, 2, (
    0.f, 0.f,
    1.f, 0.f,
    0.f, 1.f,
    1.f, 1.f
  ));
  INIT_MATRIX(y, 4, 1, ( 0.f, 0.f, 0.f, 1.f ));
  in.DebugPrint();
  std::vector<mlcc::Matrix> weight_delta;
  nn.BackProp(in, y, &weight_delta);


  for (int i = 0; i < weight_delta.size(); ++i) {
    printf("weight_delta %i:\n", i);
    weight_delta[i].DebugPrint();
  }

  for (int i = 0; i < nn.weights_.size(); ++i) {
    printf("weight_matrix %i\n", i);
    nn.weights_[i].DebugPrint();
  }

  return 0;
}
