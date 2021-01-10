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
  enum CostFunction {
    kCostMeanSquared = 0
  };
  // Layers assumed fully connected I guess.
  void AddLayer(s32 size);
  void AddLayer(NNLayer layer);

  Matrix FeedForward(const Matrix& input);
  void FeedForward(const Matrix& input, std::vector<Matrix>* pre_activation, std::vector<Matrix>* post_activation);

  void BackProp(const Matrix& input, const Matrix& y, std::vector<Matrix>* weight_delta);

  std::vector<NNLayer> layers_;
  // weights_[i] associated with layers[i] and layers[i+1].
  std::vector<Matrix> weights_;
  CostFunction cost_ = kCostMeanSquared;
};

void ApplyActivation(Matrix* m, const NNLayer& layer) {
  assert(m != nullptr);
  switch (layer.activation) {
    case NNLayer::kActivationNone: break;
    case NNLayer::kActivationSigmoid: {
      for (r32& x : m->data) {
        x = Sigmoid(x);
      }
    } break;
    case NNLayer::kActivationRelu: {
    } break;
    default: break;
  }
}

void ApplyActivationDerivative(Matrix* m, const NNLayer& layer) {
  assert(m != nullptr);
  switch (layer.activation) {
    case NNLayer::kActivationNone: break;
    case NNLayer::kActivationSigmoid: {
      for (r32& x : m->data) {
        x = SigmoidDerivative(x);
      }
    } break;
    case NNLayer::kActivationRelu: {
    } break;
    default: break;
  }
}

Matrix CostDerivative(const Matrix& in, const Matrix& y,
                      NNet::CostFunction cost_function) {
  assert(in.cols == y.cols);
  Matrix r(in.rows, in.cols);
  switch (cost_function) {
    case NNet::kCostMeanSquared: {
      for (int i = 0; i < in.data.size(); ++i) {
        r.data[i] = in.data[i] - y.data[i];
      }
    } break;
    default: break;
  }
  return r;
}

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
    ApplyActivation(&acc, out_layer);
  }
  return acc;
}

void NNet::FeedForward(const Matrix& input, std::vector<Matrix>* pre_activation,
                       std::vector<Matrix>* post_activation) {
  assert(layers_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.cols == layers_[0].size);
  Matrix acc(input);
  for (s32 i = 0; i < weights_.size(); ++i) {
    const NNLayer& out_layer = layers_[i + 1];
    acc *= weights_[i]; 
    pre_activation->push_back(acc);
    ApplyActivation(&acc, out_layer);
    post_activation->push_back(acc);
  }
}

void NNet::BackProp(const Matrix& input, const Matrix& y, std::vector<Matrix>* weight_delta) {
  std::vector<Matrix> pre_activation, post_activation;
  FeedForward(input, &pre_activation, &post_activation);
  Matrix pre_activation_derivative = pre_activation.back();
  ApplyActivationDerivative(&pre_activation_derivative, layers_.back());
  Matrix delta = HadamardProduct(
      CostDerivative(post_activation.back(), y, cost_), pre_activation_derivative);
  //delta.DebugPrint();
  //post_activation[post_activation.size() - 2].DebugPrint();
  // 4x1 * 4x2
  // 2x4 * 4x1
  delta.DebugPrint();
  post_activation[post_activation.size() - 1].DebugPrint();
  weight_delta->push_back(
      post_activation[post_activation.size() - 1].Transpose() * delta);
#if 0
  weight_delta->push_back(
      post_activation[post_activation.size() - 2].Transpose() * delta);
  for (s32 i = layers_.size() - 2; i >= 0; --i) {
    Matrix& zp = pre_activation[i];
    ApplyActivationDerivative(&zp, layers_[i]);
    // 2x1
    // 4x1
    printf("Weights[%i]\n", i);
    weights_[i].DebugPrint();
    printf("delta\n", i);
    delta.DebugPrint();
  }
#endif
  std::reverse(weight_delta->begin(), weight_delta->end());
}

}
