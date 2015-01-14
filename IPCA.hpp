/**
 * @file IPCA.hpp
 * @brief Following the paper, 
 * Juyang Weng, Yilu Zhang and Wey-Shiuan Hwang, "Candid Covariance-free Incremental Principal Component Analysis", TPAMI 
 * @author Haoxiang Li
 * @version 1.1
 * @date 2015-01-11
 */
#include <Eigen/Eigen>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> RowMatrixXf;

/**
 * @brief Candid convariance-free IPCA
 */
class CCIPCA {
public:

    /**
     * @brief Initialize CCIPCA
     *
     * @param dim_subspace number of eigen-vectors expected
     * @param dim_data dimensionality of the data points (K)
     * @param p_mean pre-computed data mean vector (D)
     * @param p_init_pca pre-computed eigen-vectors (D -by- K)
     * @param num_data_points number of data points used in pre-computation
     */
    CCIPCA(int dim_subspace, int dim_data, const float *p_mean, const float *p_init_pca, int num_data_points);

    /**
     * @brief Update the eigen-vectors given
     */
    void update(const float *pp);

    /**
     * @brief Obtain sorted (descending) eigen vectors && values
     */
    void sorted_eigen(RowMatrixXf& eigenvectors, Eigen::RowVectorXf& eigenvalues, int k) const; 

private:
    int m_num_data_points;
    RowMatrixXf m_eigenvecs;    
    Eigen::RowVectorXf m_eigenvecs_norms;
    Eigen::RowVectorXf m_mean; 
};
