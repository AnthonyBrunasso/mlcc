#pragma once

#include "src/math/math.cc"

namespace mlcc {

struct NNLayer {
  s32 size;
};

class NNet {
 public:
  // Layers assumed fully connected I guess.
  void AddLayer(s32 size);

  Matrix FeedForward(const Matrix& input);

  std::vector<NNLayer> layers_;
  // weights_[i] associated with layers[i] and layers[i+1].
  std::vector<Matrix> weights_;
};

// 0 -> 1 -> 2 -> 3 -> 4
//   0    1    2    3

// NNet nn;
// nn.AddLayer(2);  -- input
// nn.AddLayer(1);  -- output

// 1x2   2x3


void NNet::AddLayer(s32 size) {
  if (layers_.size() > weights_.size()) {
    Matrix matrix(layers_.back().size, size, Matrix::RAND_NEG_1_POS_1);
    weights_.push_back(matrix);
  }
  NNLayer layer;
  layer.size = size;
  layers_.push_back(layer);
}

Matrix NNet::FeedForward(const Matrix& input) {
  assert(layers_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.cols == layers_[0].size);
  Matrix acc(input);
  acc.DebugPrint();
  for (const auto& w : weights_) {
    acc *= w; 
  }
  return acc;
}

}
