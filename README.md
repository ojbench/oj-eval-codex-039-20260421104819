# Problem 039 - 稀疏矩阵存储

**ACMOJ Problem ID**: 2621

## Table of Contents

- [Problem 039 - 稀疏矩阵存储](#problem-039-稀疏矩阵存储)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Background](#background)
  - [Assignment Description](#assignment-description)
    - [Grade Composition](#grade-composition)
  - [Assignment Requirements](#assignment-requirements)
    - [Input Format](#input-format)
    - [Output Format](#output-format)
    - [Samples](#samples)
    - [Data Constraints](#data-constraints)
  - [Per-Testcase Resource Limits](#per-testcase-resource-limits)
  - [Test Data](#test-data)
  - [Submission Requirements](#submission-requirements)
    - [OJ Git Repository Compilation Process](#oj-git-repository-compilation-process)
    - [Git Configuration Requirements](#git-configuration-requirements)
    - [Submission Guidelines](#submission-guidelines)
    - [Evaluation Notes](#evaluation-notes)
    - [Academic Integrity](#academic-integrity)

## Introduction

### Background

Implement CSR (Compressed Sparse Row) sparse matrix with get/set operations and matrix operations.

## Assignment Description

### Grade Composition

| Grading Component | Percentage |
| :--: | :--: |
| Pass **2621. 稀疏矩阵存储** | 80% |
| Code Review | 20% |

Here are several points that need clarification:

- In the Code Review, we will **strictly examine your code style and repository organization structure, etc.**

- This assignment provides some sample data for testing, stored in the `/workspace/data/039/data_test/` directory. Note that these are not the test cases on the Online Judge. Passing all local test cases does not guarantee that you will pass the OJ tests.

- Besides the provided sample data, we also encourage you to design your own test data based on your program logic to assist debugging.

## Assignment Requirements

### Problem Description

**upd:**
1. get(i,j) 中如果 (i,j) 为 “0”（未在 data 中记录），请返回一个默认构造，无需修改矩阵
2. set(i,j,value) 无需删除元素。
3. 所有下标都是 0-base

**本题允许使用std::vector**

在数据科学和深度学习等领域会采用矩阵来存储数据，但当矩阵较为庞大且非零元素较少时，运算效率和存储有效率并不高。所以，一般情况我们采用稀疏矩阵的方式来存储矩阵，来提高存储和运算效率。

你需要实现一个 Compressed Sparse Row (CSR) 存储格式的稀疏矩阵类。

<img src="https://acm.sjtu.edu.cn/OnlineJudge/oj-images/c2ef7d0a-e721-4480-86ec-e71d0152dd98" style="width: 100%">

CSR_Matrix 是按行对矩阵进行压缩的。通过 indptr, indices, data 三个数组来确定矩阵。

对于一个 $n$ 行，$m$ 列的矩阵，indptr 数组的长度为 $n+1$，indices 和 data 数组的长度为非零元素的个数。

对于第 $i$ 行而言，该行中非零元素的列索引为 $\Big[\text{indices}[\text{indptr}[i]],\text{indices}[\text{indptr}[i+1]]\Big)$

你需要完善 `CSRMatrix` 类，实现以下功能：

```cpp
#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override {
        return "Size mismatch";
    }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override {
        return "Index out of range";
    }
};

// TODO: Implement a CSR matrix class
// You only need to implement the TODOs in this file
// DO NOT modify other parts of this file
// DO NOT include any additional headers
// DO NOT use STL other than std::vector

template <typename T>
class CSRMatrix {

private:
    // TODO: add your private member variables here
    
public:
    // Assignment operators are deleted
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    // Constructor for empty matrix with dimensions
    // TODO: Initialize an empty CSR matrix with n rows and m columns
    CSRMatrix(size_t n, size_t m);

    // Constructor with pre-built CSR components
    // TODO: Initialize CSR matrix from existing CSR format data, validate sizes
    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr, 
        const std::vector<size_t> &indices,
        const std::vector<T> &data);

    // Copy constructor
    CSRMatrix(const CSRMatrix &other) = default;

    // Move constructor
    CSRMatrix(CSRMatrix &&other) = default;

    // Constructor from dense matrix format (given as vector of vectors)
    // TODO: Convert dense matrix representation to CSR format
    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &data);

    // Destructor
    ~CSRMatrix() = default;

    // Get dimensions and non-zero count
    // TODO: Return the number of rows
    size_t getRowSize() const;

    // TODO: Return the number of columns
    size_t getColSize() const;

    // TODO: Return the count of non-zero elements
    size_t getNonZeroCount() const;

    // Element access
    // TODO: Retrieve element at position (i,j)
    T get(size_t i, size_t j) const;

    // TODO: Set element at position (i,j), updating CSR structure as needed
    void set(size_t i, size_t j, const T &value);

    // Access CSR components
    // TODO: Return the row pointer array
    const std::vector<size_t> &getIndptr() const;

    // TODO: Return the column indices array
    const std::vector<size_t> &getIndices() const;

    // TODO: Return the data values array
    const std::vector<T> &getData() const;

    // Convert to dense matrix format
    // TODO: Convert CSR format to dense matrix representation
    std::vector<std::vector<T>> getMatrix() const;

    // Matrix-vector multiplication
    // TODO: Implement multiplication of this matrix with vector vec
    std::vector<T> operator*(const std::vector<T> &vec) const;

    // Row slicing
    // TODO: Extract submatrix containing rows [l,r)
    CSRMatrix getRowSlice(size_t l, size_t r) const;
};

}

#endif // CSR_MATRIX_HPP

```

### Input Format

See the problem description above.

### Output Format

See the problem description above.

### Samples

No sample data provided for this problem.

### Data Constraints

| subtask | 前置  |  额外要求  | 分值  |
| --- | --- | --- | --- |
| 1   | 0   | 构造函数 / getRowSize() / getColSize() / getNonZeroCount() / getIndptr() / getIndices() / getData() | 20  |
| 2   |  1  |  get() set()   |  20  |
| 3   |  1  |  getMatrix()   |  10  |
| 4   |  1  |  operator*()   |  10  |
| 5   |  1  |  getRowSlice() |  10  |
| 6   |  2,3,4,5 | 异常处理   |  10  |
| 7   |  2,3,4,5 | 效率测试   |  20  |

## Per-Testcase Resource Limits

- **Time Limit (per test case)**: 1000 ms
- **Memory Limit (per test case)**: 256 MiB
- **Disk Usage**: No disk usage is permitted.

## Test Data

The test data for this problem is located at `/workspace/data/039/data_test/`.

## Submission Requirements

### OJ Git Repository Compilation Process

For Git compilation, we will first clone the repository using a command similar to:
```bash
git clone <repo_url> . --depth 1 --recurse-submodules --shallow-submodules --no-local
```

Then we check if there is a `CMakeLists.txt` file. If it exists, we run (if not, a warning message will be displayed):
```bash
cmake .
```

Finally, we check if there is any of `GNUmakefile`/`makefile`/`Makefile` (if cmake was run previously, this will be the generated Makefile). If it exists, we run (if not, a warning message will be displayed):
```bash
make
```

After this process is complete, we will use the `code` file in the project root directory as the compilation result.

A `CMakeLists.txt` file is provided in the project. You can use or modify it as needed. The local environment has gcc-13 and g++-13 available.

### Git Configuration Requirements

**IMPORTANT**: You must create a `.gitignore` file in your project root directory to avoid OJ evaluation conflicts.

The `.gitignore` file should include at least the following entries:

```gitignore
CMakeFiles/
CMakeCache.txt
```

### Submission Guidelines

- The submitted code must be able to compile successfully through the above compilation process
- The compiled executable file name must be `code`
- The program needs to be able to read data from standard input and write results to standard output
- Please ensure the code runs correctly within the given time and space limits
- **You must use C++ or C language** to implement this assignment

### Evaluation Notes

- The evaluation system will test your program using the provided test data
- The program output must exactly match the expected output (including format)
- Exceeding time or memory limits will be judged as the corresponding error type
- Please pay attention to the overall time performance of your code and the time complexity of each part of your algorithm.

### Academic Integrity

If any violations are found during evaluation or code review (including but not limited to using unconventional methods to pass test cases), your final score may be significantly reduced or become **0 points**.
