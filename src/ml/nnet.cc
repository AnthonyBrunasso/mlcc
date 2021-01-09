#pragma once

#include "src/math/math.cc"

namespace mlcc {

struct NNLayer {
  enum Activation {
    kActivationNone = 0,
    kActivationSigmoid = 1,
    kActivationRelu = 2,
  };
  s32 size;
  Activation activation;
};

class NNet {
 public:
  // Layers assumed fully connected I guess.
  void AddLayer(s32 size);
  void AddLayer(NNLayer layer);

  Matrix FeedForward(const Matrix& input);
  void FeedForward(const Matrix& input, std::vector<Matrix>* activations);

  std::vector<NNLayer> layers_;
  // weights_[i] associated with layers[i] and layers[i+1].
  std::vector<Matrix> weights_;
};

void NNet::AddLayer(s32 size) {
  NNLayer layer;
  layer.size = size;
  layer.activation = NNLayer::kActivationNone;
  AddLayer(layer);
}

void NNet::AddLayer(NNLayer layer) {
  if (layers_.size() > weights_.size()) {
    Matrix matrix(layers_.back().size, layer.size, Matrix::RAND_NEG_1_POS_1);
    weights_.push_back(matrix);
  }
  layers_.push_back(layer);
}

Matrix NNet::FeedForward(const Matrix& input) {
  assert(layers_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.cols == layers_[0].size);
  Matrix acc(input);
  for (int i = 0; i < weights_.size(); ++i) {
    const NNLayer& out_layer = layers_[i + 1];
    acc *= weights_[i]; 
    switch (out_layer.activation) {
      case NNLayer::kActivationNone: break;
      case NNLayer::kActivationSigmoid: {
        for (r32& x : acc.data) {
          x = Sigmoid(x);
        }
      } break;
      case NNLayer::kActivationRelu: {
      } break;
      default: break;
    }
  }
  return acc;
}

void NNet::FeedForward(const Matrix& input, std::vector<Matrix>* activations) {
  assert(layers_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.cols == layers_[0].size);
  Matrix acc(input);
  for (int i = 0; i < weights_.size(); ++i) {
    const NNLayer& out_layer = layers_[i + 1];
    acc *= weights_[i]; 
    switch (out_layer.activation) {
      case NNLayer::kActivationNone: break;
      case NNLayer::kActivationSigmoid: {
        for (r32& x : acc.data) {
          x = Sigmoid(x);
        }
      } break;
      case NNLayer::kActivationRelu: {
      } break;
      default: break;
    }
    activations->push_back(acc);
  }
}

}
