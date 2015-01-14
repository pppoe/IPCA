#include "IPCA.hpp"
#include <algorithm>
#include <vector>
#include <cassert>
#include <iostream>

using namespace Eigen;
using namespace std;

CCIPCA::CCIPCA(int dim_subspace, int dim_data, 
        const float *p_mean, const float *p_init_pca, int num_data_points) {
    m_eigenvecs = Map<const RowMatrixXf>(p_init_pca, dim_data, dim_subspace);
    m_mean = Map<const RowVectorXf>(p_mean, dim_data);
    m_eigenvecs_norms = m_eigenvecs.colwise().norm();
    m_num_data_points = num_data_points;
}

void CCIPCA::update(const float *pp) {

    const int D = m_mean.cols();
    const int K = m_eigenvecs.cols();

    m_num_data_points++;

    //< TODO amnesic weights
    const float w1 = (m_num_data_points-1)/(float)m_num_data_points;
    const float w2 = (1)/(float)m_num_data_points;

    VectorXf vec_pp = Map<const VectorXf>(pp, D) - m_mean.transpose();
    for (int k = 0; k < K; k++) {
        VectorXf v = m_eigenvecs.col(k);
        m_eigenvecs.col(k) = w1 * v + (v.dot(vec_pp))*vec_pp*(w2/m_eigenvecs_norms(k));
        m_eigenvecs_norms(k) = m_eigenvecs.col(k).norm();
        VectorXf nrm_v = m_eigenvecs.col(k).normalized();
        vec_pp -= (vec_pp.dot(nrm_v))*nrm_v;
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
    std::sort(indexed_vals.begin(), indexed_vals.end(), std::greater<std::pair<float,int> > ());
    eigenvalues = RowVectorXf(k);
    eigenvectors = RowMatrixXf(D, k);
    for (int n = 0; n < k; n++) {
        const int i = indexed_vals[n].second;
        eigenvalues(n) = m_eigenvecs_norms(i);
        eigenvectors.col(n) = m_eigenvecs.col(i).normalized();
    }
}
