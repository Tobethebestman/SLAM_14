#include <iostream>
#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// gaussNewton: delta_x* = argminF(x), f(x) = Y - F(x), f(x + delta_x) = f(x) + J(x).T * delta_x
// ---------->  Eq: J(x) * J(x).T * delta_x = -1 * J(x) * f(x), x_k+1 = x_k + delta_x_k
int main(int agrc, char **argv) {

    double ar = 1.0, br = 2.0, cr = 1.0;    // true parameters
    double ae = 4.0, be = -3.0, ce = 5.0;   // esti parameters
    int N = 100;                            // data count
    double w_sigma = 1.0;                   // noise's sigma
    double inv_sigma = 1.0 / w_sigma;       
    cv::RNG rng;                            // random creation

    // create dot data with noise
    vector<double> x_data, y_data;
    for (int i =0; i < N; i++) {
        double x = i / 100.0;
        double y = exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma);
        x_data.push_back(x);
        y_data.push_back(y);
    }

    // Gauss-Newton iteration
    int iterations = 100;
    double cost = 0, lastCost = 0;  // this time & last time cost

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        // initialize
        Matrix3d H = Matrix3d::Zero();  // Hessian Matrix: J * W^{-1} * J.T
        Vector3d b = Vector3d::Zero();  // Bias Matrix: -1 * J * W^{-1} * e_i
        cost = 0;

        // iteration
        for (int i = 0; i < N; i++) {
            // compute error
            double xi = x_data[i], yi = y_data[i];
            double error = yi - exp(ae * xi * xi + be * xi + ce);

            // get Jacobian
            Vector3d J;
            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);
            J[2] = -exp(ae * xi * xi + be * xi + ce);

            // update H&b
            H += inv_sigma * inv_sigma * J * J.transpose();
            b += -inv_sigma * inv_sigma * error * J;

            // update cost
            cost += error * error;
        }

        // solve linear_function: Hx = b
        Vector3d dx = H.ldlt().solve(b);

        // check divergence
        if (isnan(dx[0])) {      
            cout << "result is nan!" << endl;
            break;
        }

        // check minimum
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break; 
        }

        // update parameters
        ae += dx[0];
        be += dx[1];
        ce += dx[2];

        // update cost
        lastCost = cost;

        // output info
        if (iter % 5 == 0) {
            cout << "iteration: " << iter << "\ttotal cost: " << cost << ", \t\tupdate: " << dx.transpose() << "\t\testimated params: " << ae << "," << be << "," << ce << endl;
        }
    }

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds." << endl;
    cout << "estimated abc = " << ae << "," << be << "," << ce << endl;

    return 0;
}