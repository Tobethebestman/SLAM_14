#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// computation model for cost function
struct CURVE_FITTING_COST {

    // constructor
    CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

    // residual computation
    template<typename T>        // function module to support muti-type parameters
    // abc: model parameters
    bool operator() (const T *const abc, T *residual) const {
        // reult is saved in residual[0]
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    const double _x, _y;
};

int main(int argc, char **agrv) {
    // initialize parameters
    double ar = 1.0, br = 2.0, cr = 1.0;
    double ae = 2.0, be = -1.0, ce = 5.0;
    int N = 100;
    double w_sigma = 1.0;
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;

    // get data
    vector<double> x_data, y_data;
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        double y = exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma); // mean = 0, std = w_sigma ** 2
        x_data.push_back(x);
        y_data.push_back(y);
    }

    double abc[3] = {ae, be, ce};   // optimization variable

    // construct Least Squares Problem
    ceres::Problem problem;
    for (int i = 0; i < N; i++) {
        // add residual
        // module param:<constructor, residual dimension, parameters dimension>(constructor, kernel function, optimization parameters)
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(new CURVE_FITTING_COST(x_data[i], y_data[i])), nullptr, abc);
    }

    // configure the solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // solve increment function
    options.minimizer_progress_to_stdout = true;                // output to cout
    ceres::Solver::Summary summary;                             // log the optimization data

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();

    // Optimization
    ceres::Solve(options, &problem, &summary);      

    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();  
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds." << endl;

    // print result
    cout << summary.BriefReport() << endl;
    cout << "estimated a, b, c = ";
    for (auto a:abc) cout << a << " ";
    cout << endl;

    return 0;      
}
