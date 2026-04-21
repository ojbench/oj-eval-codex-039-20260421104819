#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include <vector>
#include <exception>

namespace sjtu {

class size_mismatch : public std::exception {
public:
    const char *what() const noexcept override { return "Size mismatch"; }
};

class invalid_index : public std::exception {
public:
    const char *what() const noexcept override { return "Index out of range"; }
};

template <typename T>
class CSRMatrix {
private:
    size_t n_rows{};
    size_t n_cols{};
    std::vector<size_t> indptr;   // size n_rows+1
    std::vector<size_t> indices;  // size nnz
    std::vector<T> data;          // size nnz

    // binary search column in [l,r)
    static long find_col(const std::vector<size_t>& idx, size_t l, size_t r, size_t j){
        size_t lo=l, hi=r;
        while(lo<hi){
            size_t mid=(lo+hi)>>1;
            if(idx[mid]<j) lo=mid+1; else hi=mid;
        }
        if(lo<r && idx[lo]==j) return (long)lo;
        return -((long)lo)-1; // insertion point
    }

public:
    CSRMatrix &operator=(const CSRMatrix &other) = delete;
    CSRMatrix &operator=(CSRMatrix &&other) = delete;

    CSRMatrix(size_t n, size_t m): n_rows(n), n_cols(m), indptr(n+1, 0) {}

    CSRMatrix(size_t n, size_t m, size_t count,
        const std::vector<size_t> &indptr_,
        const std::vector<size_t> &indices_,
        const std::vector<T> &data_)
        : n_rows(n), n_cols(m), indptr(indptr_), indices(indices_), data(data_) {
        if(indptr.size()!=n_rows+1) throw size_mismatch();
        if(indices.size()!=count || data.size()!=count) throw size_mismatch();
        if(indptr.front()!=0 || indptr.back()!=count) throw size_mismatch();
        for(size_t i=1;i<indptr.size();++i) if(indptr[i]<indptr[i-1]) throw size_mismatch();
        for(size_t i=0;i<indices.size();++i) if(indices[i]>=n_cols) throw invalid_index();
    }

    CSRMatrix(const CSRMatrix &other) = default;
    CSRMatrix(CSRMatrix &&other) = default;

    CSRMatrix(size_t n, size_t m, const std::vector<std::vector<T>> &dense)
        : n_rows(n), n_cols(m), indptr(n+1,0) {
        if(dense.size() != n_rows) throw size_mismatch();
        for(const auto &row : dense) if(row.size()!=n_cols) throw size_mismatch();
        indices.clear(); data.clear();
        indices.reserve(n_rows? (n_cols* n_rows/4 + 1) : 1);
        data.reserve(indices.capacity());
        for(size_t i=0;i<n_rows;++i){
            const auto &row = dense[i];
            for(size_t j=0;j<n_cols;++j){
                T v = row[j];
                if(v!=T{}){ indices.push_back(j); data.push_back(v); }
            }
            indptr[i+1]=indices.size();
        }
    }

    ~CSRMatrix() = default;

    size_t getRowSize() const { return n_rows; }
    size_t getColSize() const { return n_cols; }
    size_t getNonZeroCount() const { return data.size(); }

    T get(size_t i, size_t j) const {
        if(i>=n_rows || j>=n_cols) throw invalid_index();
        size_t l = indptr[i], r = indptr[i+1];
        long pos = find_col(indices, l, r, j);
        if(pos>=0) return data[(size_t)pos];
        return T{};
    }

    void set(size_t i, size_t j, const T &value){
        if(i>=n_rows || j>=n_cols) throw invalid_index();
        size_t l = indptr[i], r = indptr[i+1];
        long pos = find_col(indices, l, r, j);
        if(pos>=0){
            data[(size_t)pos] = value;
            return;
        }
        size_t ins = (size_t)(-pos-1);
        indices.insert(indices.begin()+ins, j);
        data.insert(data.begin()+ins, value);
        for(size_t k=i+1;k<indptr.size();++k) ++indptr[k];
    }

    const std::vector<size_t> &getIndptr() const { return indptr; }
    const std::vector<size_t> &getIndices() const { return indices; }
    const std::vector<T> &getData() const { return data; }

    std::vector<std::vector<T>> getMatrix() const {
        std::vector<std::vector<T>> dense(n_rows, std::vector<T>(n_cols, T{}));
        for(size_t i=0;i<n_rows;++i){
            for(size_t k=indptr[i]; k<indptr[i+1]; ++k){
                dense[i][indices[k]] = data[k];
            }
        }
        return dense;
    }

    std::vector<T> operator*(const std::vector<T> &vec) const {
        if(vec.size()!=n_cols) throw size_mismatch();
        std::vector<T> res(n_rows, T{});
        for(size_t i=0;i<n_rows;++i){
            T sum = T{};
            for(size_t k=indptr[i]; k<indptr[i+1]; ++k){
                sum = sum + data[k] * vec[indices[k]];
            }
            res[i]=sum;
        }
        return res;
    }

    CSRMatrix getRowSlice(size_t l, size_t r) const {
        if(l>r || r>n_rows) throw invalid_index();
        size_t new_n = r-l;
        CSRMatrix<T> sub(new_n, n_cols);
        size_t start = indptr[l];
        size_t end = indptr[r];
        sub.indices.assign(indices.begin()+start, indices.begin()+end);
        sub.data.assign(data.begin()+start, data.begin()+end);
        sub.indptr.resize(new_n+1);
        sub.indptr[0]=0;
        for(size_t i=0;i<new_n;++i){
            sub.indptr[i+1] = sub.indptr[i] + (indptr[l+i+1]-indptr[l+i]);
        }
        return sub;
    }
};

}

#endif // CSR_MATRIX_HPP
