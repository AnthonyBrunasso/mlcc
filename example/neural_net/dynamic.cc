#include "ml/nnet.cc"

int main(int argc, char** argv) {
  mlcc::NNet nn;
  nn.AddLayer(2);
  nn.AddLayer({
    .size = 2,
    .activation = mlcc::NNLayer::kActivationSigmoid
  });
  nn.AddLayer(1);
  INIT_MATRIX(in, 4, 2, (
    0.f, 0.f,
    1.f, 0.f,
    0.f, 1.f,
    1.f, 1.f
  ));
  printf("Input:\n");
  in.DebugPrint();
  printf("Output:\n");
  std::vector<mlcc::Matrix> activations;
  nn.FeedForward(in, &activations);
  printf("ACTIVATIONS\n");
  for (const auto& act : activations) {
    act.DebugPrint();
  }
  return 0;
}
