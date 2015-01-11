#include "IPCA.hpp"
#include <algorithm>
#include <vector>
#include <cassert>

using namespace Eigen;
using namespace std;

CCIPCA::CCIPCA(int dim_subspace, int dim_data, const float *p_mean, const float *p_init_pca) {
    m_eigenvecs = Map<const RowMatrixXf>(p_init_pca, dim_data, dim_subspace);
    m_mean = Map<const RowVectorXf>(p_mean, dim_data);
    m_eigenvecs_norms = m_eigenvecs.colwise().norm();
}

void CCIPCA::update(const float *pp) {

    const int D = m_mean.cols();
    const int K = m_eigenvecs.cols();

    //< TODO amnesic weights
    const float w1 = 0.5f, w2 = 0.5f; 

    RowVectorXf vec_pp = Map<const RowVectorXf>(pp, D) - m_mean;
    for (int k = 0; k < K; k++) {
        VectorXf v = m_eigenvecs.col(k);
        m_eigenvecs.col(k) = w1 * v + w2 *(v*vec_pp)* vec_pp.transpose() / m_eigenvecs_norms(k);
        m_eigenvecs_norms(k) = m_eigenvecs.col(k).norm();
        v.normalize();
        vec_pp -= (vec_pp*v)*v.transpose();
    }
}

void CCIPCA::sorted_eigen(RowMatrixXf& eigenvectors, Eigen::RowVectorXf& eigenvalues, int k) const {

    assert (k <= m_eigenvecs.cols());

    const int D = m_mean.cols();
    const int K = m_eigenvecs.cols();

    vector<pair<float, int> > indexed_vals(K);
    for (int i = 0; i < K; i++) {
        indexed_vals[i] = make_pair(m_eigenvecs_norms(i), i);
    }
    std::sort(indexed_vals.begin(), indexed_vals.end());
    eigenvalues = RowVectorXf(k);
    eigenvectors = RowMatrixXf(D, k);
    for (int n = 0; n < k; n++) {
        const int i = indexed_vals[n].second;
        eigenvalues(n) = m_eigenvecs_norms(i);
        eigenvectors.col(n) = m_eigenvecs.col(i).normalized();
    }
}
