#include <iostream>
using namespace std;

#include <ctime>

// Eigen core
#include <Eigen/Core>

// Eigen matrix calculation
#include <Eigen/Dense>
using namespace Eigen;

#define MATRIX_SIZE 50

int main(int agrc, char **argv){

    // Some ERRORS in VSCode, but can successfully compiled in the terminal.
    Matrix<float, 2, 3> matrix_23;
    //MatrixXd matrix_23(2, 3);


    // they are the same construct
    Vector3d v_3d;              // double type
    Matrix<float, 3, 1> vd_3d;  // float type

    // MatrixMd:row = column = M
    Matrix3d matrix_3_3 = Matrix3d::Zero();

    // unknown row and column
    Matrix<double, Dynamic, Dynamic> matrix_dynamic;

    // initialize matrix_23
    matrix_23 << 1, 2, 3, 4, 5, 6;

    // output matrix_23
    cout << "matrix_2x3 from 1 to 6: \n" << matrix_23 << endl;

    cout << "print matrix 2x3:" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j ++) cout << matrix_23(i, j) << "\t";
        cout << endl;
    }

    v_3d << 1, 2, 3;
    vd_3d << 4, 5, 6;

    // can't mixture two different type of matrix
    // Matrix<double, 2, 1> result_wrong = matrix_23 * v_3d;
    // cout << result_wrong << endl;

    // explicit conversion
    Matrix<double, 2, 1> result_right = matrix_23.cast<double>() * v_3d;
    cout << "[1, 2, 3; 4, 5, 6] * [1, 2, 3]=" << result_right.transpose() << endl;

    // same matrix type
    Matrix<float, 2, 1> result_right2 = matrix_23 * vd_3d;
    cout << "[1, 2, 3; 4, 5, 6] * [4, 5, 6]=" << result_right2.transpose() << endl;

    Matrix3d matrix_33 = Matrix3d::Random();
    cout << "random_matrix: \n"  << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;
    cout << "sum: \n" << matrix_33.sum() << endl;
    cout << "trace: \n" << matrix_33.trace() << endl;
    cout << "inverse: \n" << matrix_33.inverse() << endl;
    cout << "det: \n" << matrix_33.determinant() << endl;

    return 0;
}