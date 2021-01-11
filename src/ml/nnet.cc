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
  b8 use_bias = false;
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
  void FeedForward(const Matrix& input, std::vector<Matrix>* pre_activation,
                   std::vector<Matrix>* post_activation);
  void BackProp(const Matrix& input, const Matrix& y, std::vector<Matrix>* weight_delta);

  std::vector<NNLayer> layer_;
  // weight_[i] associated with layers[i] and layers[i+1].
  std::vector<Matrix> weight_;
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
  layer.use_bias = false;
  AddLayer(layer);
}

void NNet::AddLayer(NNLayer layer) {
  if (layer_.size() > weight_.size()) {
    if (layer_.back().use_bias) {
      Matrix matrix(layer.size, layer_.back().size + 1, Matrix::RAND_NEG_1_POS_1);
      weight_.push_back(matrix);
    } else {
      Matrix matrix(layer.size, layer_.back().size, Matrix::RAND_NEG_1_POS_1);
      weight_.push_back(matrix);
    }
  }
  layer_.push_back(layer);
}

Matrix NNet::FeedForward(const Matrix& input) {
  assert(layer_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.rows == layer_[0].size);
  Matrix acc(input);
  for (s32 i = 0; i < weight_.size(); ++i) {
    const NNLayer& in_layer = layer_[i];
    const NNLayer& out_layer = layer_[i + 1];
    if (in_layer.use_bias) {
      // Add a 1.0 to the input.
      Matrix new_acc(acc.rows + 1, acc.cols, acc);
      for (s32 j = 0; j < acc.cols; ++j) {
        new_acc.data[new_acc.idx(acc.rows, j)] = 1.f;
      }
      acc = std::move(new_acc);
    }
    //acc.DebugPrint();
    //weight_[i].DebugPrint();
    acc = weight_[i] * acc; 
    ApplyActivation(&acc, out_layer);
  }
  return acc;
}

void NNet::FeedForward(const Matrix& input, std::vector<Matrix>* pre_activation,
                       std::vector<Matrix>* post_activation) {
  assert(layer_.size() > 0);
  // Input should be a column vector the size of the input.
  assert(input.rows == layer_[0].size);
  Matrix acc(input);
  for (s32 i = 0; i < weight_.size(); ++i) {
    const NNLayer& in_layer = layer_[i];
    const NNLayer& out_layer = layer_[i + 1];
    //printf("weight_[i]\n");weight_[i].DebugPrint();
    //printf("acc\n");acc.DebugPrint();
    if (in_layer.use_bias) {
      // Add a 1.0 to the input.
      Matrix new_acc(acc.rows + 1, acc.cols, acc);
      for (s32 j = 0; j < acc.cols; ++j) {
        new_acc.data[new_acc.idx(acc.rows, j)] = 1.f;
      }
      acc = std::move(new_acc);
    }
    if (pre_activation->empty()) pre_activation->push_back(acc);
    acc = weight_[i] * acc; 
    pre_activation->push_back(acc);
    ApplyActivation(&acc, out_layer);
    post_activation->push_back(acc);
  }
}

void NNet::BackProp(const Matrix& input, const Matrix& y,
                    std::vector<Matrix>* weight_delta) {
  std::vector<Matrix> pre_activation, post_activation;
  FeedForward(input, &pre_activation, &post_activation);
  Matrix pre_activation_derivative = pre_activation.back();
  ApplyActivationDerivative(&pre_activation_derivative, layer_.back());
  //printf("POST_ACTIVATION:\n");post_activation.back().DebugString();
  Matrix delta = HadamardProduct(
      CostDerivative(post_activation.back(), y, cost_), pre_activation_derivative);
  //printf("ACTIV:\n");post_activation.back().DebugPrint();
  //printf("ACTUA:\n");y.DebugPrint();
  //printf("COSTD\n");CostDerivative(post_activation.back(), y, cost_).DebugPrint();
  weight_delta->push_back(
      delta * pre_activation[pre_activation.size() - 2].Transpose());
#if 0
  weight_delta->push_back(
      post_activation[post_activation.size() - 2].Transpose() * delta);
  for (s32 i = layer_.size() - 2; i >= 0; --i) {
    Matrix& zp = pre_activation[i];
    ApplyActivationDerivative(&zp, layer_[i]);
    // 2x1
    // 4x1
    printf("Weights[%i]\n", i);
    weight_[i].DebugPrint();
    printf("delta\n", i);
    delta.DebugPrint();
  }
#endif
  std::reverse(weight_delta->begin(), weight_delta->end());
}

}
