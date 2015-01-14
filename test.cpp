#include <iostream>
#include "IPCA.hpp"

using namespace Eigen;
using namespace std;

int main(int argc, char **argv) {

    const int num_points = 1000;
    const int init_points = 80;
    const int D = 10;
    const int K = 5;

    const RowMatrixXf random_data = RowMatrixXf::Random(num_points, D);
    RowVectorXf data_mean = random_data.colwise().mean();

    RowMatrixXf init_data = random_data.topRows(init_points);
    init_data.rowwise() -= data_mean;
    RowMatrixXf data_cov = init_data.transpose() * init_data;

    SelfAdjointEigenSolver<RowMatrixXf> eig;
    eig.compute(data_cov);

    //< corresponded eigen values are in descending order
    RowMatrixXf init_pca(D, K);
    RowVectorXf init_eignvals(K);
    for (int k = 0; k < K; k++) {
        init_pca.col(k) = eig.eigenvectors().col(D - 1 - k);
        init_eignvals(k) = eig.eigenvalues()(D - 1 - k);
        init_pca.col(k) *= init_eignvals(k);
    }

    CCIPCA ccipca(K, D, data_mean.data(), init_pca.data(), init_points);
    for (int i = init_points; i < num_points; i++) {
        ccipca.update(random_data.row(i).data());
    }

    RowVectorXf ccipca_eig_vals;
    RowMatrixXf ccipca_eig_vecs;
    ccipca.sorted_eigen(ccipca_eig_vecs, ccipca_eig_vals, K);

    cout << ccipca_eig_vals << endl;
    cout << ccipca_eig_vecs << endl;

    return 0;
}

