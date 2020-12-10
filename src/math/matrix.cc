#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>

#include "type.cc"

namespace mlcc {

#define MAT_ARGS(...) __VA_ARGS__

// LOL! This is neat.
#define INIT_MATRIX(name, r, c, arr_data)  \
  r32 init_##name[] = {MAT_ARGS arr_data}; \
  mlcc::Matrix name(r, c, init_##name);

struct Matrix {
  enum InitType {
    ZERO,
    RAND_NEG_1_POS_1,
  };

  Matrix() = default;
  Matrix(u32 m, u32 n);
  Matrix(u32 m, u32 n, InitType init_type);
  Matrix(u32 m, u32 n, r32* arr);

  ~Matrix() = default;

  void operator*=(const Matrix& rhs);
  Matrix operator*(const Matrix& rhs);

  void operator*=(r32 rhs);
  Matrix operator*(r32 rhs);

  void operator+=(r32 rhs);
  Matrix operator+(r32 rhs);

  void operator-=(r32 rhs);
  Matrix operator-(r32 rhs);
  
  Matrix Transpose() const;

  void DebugPrint();

  u32 idx(u32 i, u32 j) const;

  std::vector<r32> data;
  u32 rows = 0;
  u32 cols = 0;
};

Matrix::Matrix(u32 m, u32 n) : rows(m), cols(n) {
  data.resize(m * n, 0.f);
}

Matrix::Matrix(u32 m, u32 n, InitType init_type) : Matrix(m, n) {
  switch (init_type) {
    case ZERO: break;
    case RAND_NEG_1_POS_1: {
      for (u32 i = 0; i < data.size(); ++i) {
        data[i] = Random(-1.f, 1.f);
      }
    } break;
    default: break;
  };
}

Matrix::Matrix(u32 m, u32 n, r32* arr) : Matrix(m, n) {
  for (u32 i = 0; i < m * n; ++i) {
    data[i] = arr[i];
  }
}

void Matrix::operator*=(const Matrix& rhs) {
  assert(cols == rhs.rows);
  if (cols != rhs.cols) {
    *this = Matrix(rows, rhs.cols);
  }
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < rhs.cols; ++j) {
      r32 sum = 0;
      for (int k = 0; k < cols; ++k) {
        sum += data[idx(i, k)] * rhs.data[rhs.idx(k, j)];
      }
      data[idx(i, j)] = sum;
    }
  }
}

Matrix Matrix::operator*(const Matrix& rhs) {
  assert(cols == rhs.rows);
  Matrix result(rows, rhs.cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < rhs.cols; ++j) {
      r32 sum = 0;
      for (int k = 0; k < cols; ++k) {
        sum += data[idx(i, k)] * rhs.data[rhs.idx(k, j)];
      }
      result.data[result.idx(i, j)] = sum;
    }
  }
  return result;
}

void Matrix::operator*=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] *= rhs;
  }
}

Matrix Matrix::operator*(r32 rhs) {
  Matrix m(*this);
  m *= rhs;
  return m;
}

void Matrix::operator+=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] += rhs;
  }
}

Matrix Matrix::operator+(r32 rhs) {
  Matrix m(*this);
  m += rhs;
  return m;
}

void Matrix::operator-=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] -= rhs;
  }
}

Matrix Matrix::operator-(r32 rhs) {
  Matrix m(*this);
  m -= rhs;
  return m;
}

Matrix Matrix::Transpose() const {
  Matrix m(cols, rows);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      m.data[m.idx(j, i)] = data[idx(i, j)];
    }
  }
  return m;
}

void Matrix::DebugPrint() {
  for (u32 i = 0; i < rows; ++i) {
    for (u32 j = 0; j < cols; ++j) {
      printf("%.3f ", data[idx(i, j)]);
    }
    printf("\n");
  }
}

u32 Matrix::idx(u32 i, u32 j) const {
  return i * cols + j;
}

}  // namespace mlcc
