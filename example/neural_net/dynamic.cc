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
  printf("Input:\n");
  in.DebugPrint();
  printf("Output:\n");
  nn.FeedForward(in).DebugPrint();
  return 0;
}
