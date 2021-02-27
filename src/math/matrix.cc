#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>

#include "type.cc"
#include "util.cc"

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
  Matrix(u32 m, u32 n, const Matrix& copy);
  Matrix(u32 m, u32 n, InitType init_type);
  Matrix(u32 m, u32 n, r32* arr);

  ~Matrix() = default;

  void operator*=(const Matrix& rhs);
  Matrix operator*(const Matrix& rhs) const;

  void operator*=(r32 rhs);
  Matrix operator*(r32 rhs) const;

  void operator+=(r32 rhs);
  Matrix operator+(r32 rhs) const;

  void operator+=(const Matrix& rhs);
  Matrix operator+(const Matrix& rhs) const;

  void operator-=(r32 rhs);
  Matrix operator-(r32 rhs) const;

  void operator-=(const Matrix& rhs);
  Matrix operator-(const Matrix& rhs) const;

  void operator/=(r32 rhs);
  Matrix operator/(r32 rhs) const;

  Matrix Transpose() const;

  void DebugPrint() const;

  u32 idx(u32 i, u32 j) const;

  r32 val(u32 i, u32 j) const;

  std::vector<r32> data;
  u32 rows = 0;
  u32 cols = 0;
};

Matrix::Matrix(u32 m, u32 n) : rows(m), cols(n) {
  data.resize(m * n, 0.f);
}


Matrix::Matrix(u32 m, u32 n, const Matrix& copy) : Matrix(m, n) {
  assert(copy.rows <= m && copy.cols <= n);
  for (s32 i = 0; i < copy.rows; ++i) {
    for (s32 j = 0; j < copy.cols; ++j) {
      data[idx(i, j)] = copy.data[idx(i, j)];
    }
  }
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
  *this = *this * rhs;
}

Matrix Matrix::operator*(const Matrix& rhs) const {
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

Matrix Matrix::operator*(r32 rhs) const {
  Matrix m(*this);
  m *= rhs;
  return m;
}

void Matrix::operator+=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] += rhs;
  }
}

Matrix Matrix::operator+(r32 rhs) const {
  Matrix m(*this);
  m += rhs;
  return m;
}

void Matrix::operator+=(const Matrix& rhs) {
  assert(data.size() == rhs.data.size());
  for (int i = 0; i < data.size(); ++i) {
    data[i] += rhs.data[i];
  }
}

Matrix Matrix::operator+(const Matrix& rhs) const {
  Matrix r(*this);
  r += rhs;
  return r;
}

void Matrix::operator-=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] -= rhs;
  }
}

Matrix Matrix::operator-(r32 rhs) const {
  Matrix m(*this);
  m -= rhs;
  return m;
}

void Matrix::operator-=(const Matrix& rhs) {
  assert(data.size() == rhs.data.size());
  for (int i = 0; i < data.size(); ++i) {
    data[i] -= rhs.data[i];
  }
}

Matrix Matrix::operator-(const Matrix& rhs) const {
  Matrix r(*this);
  r -= rhs;
  return r;
}

void Matrix::operator/=(r32 rhs) {
  for (u32 i = 0; i < data.size(); ++i) {
    data[i] /= rhs;
  }
}

Matrix Matrix::operator/(r32 rhs) const {
  Matrix m(*this);
  m /= rhs;
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

void Matrix::DebugPrint() const {
  printf("Matrix %ix%i\n", rows, cols);
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

r32 Matrix::val(u32 i, u32 j) const {
  return data[idx(i, j)];
}

Matrix HadamardProduct(const Matrix& a, const Matrix& b) {
  assert(a.rows == b.rows);
  assert(a.cols == b.cols);
  Matrix r(a);
  for (s32 i = 0; i < b.data.size(); ++i) {
    r.data[i] *= b.data[i];
  }
  return r;
}

Matrix RowWiseProduct(const Matrix& a, const Matrix& b) {
  assert(a.cols == b.cols);
  Matrix res(a);
  for (s32 r = 0; r < a.rows; ++r) {
    for (s32 c = 0; c < a.cols; ++c) {
      res.data[res.idx(r, c)] *= b.data[b.idx(0, c)];
    }
  }
  return res;
}

}  // namespace mlcc
