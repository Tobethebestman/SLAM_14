#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void solveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.normalize();                // data normalization
    bal_problem.perturb(0.1, 0.5, 0.5);     // noise addation
    bal_problem.writeToPLYFile("initial.ply");
    solveBA(bal_problem);
    bal_problem.writeToPLYFile("final.ply");

    return 0;
} 

void solveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    const double *observations = bal_problem.observations();

    // create ceres problem
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); i++) {
        ceres::CostFunction *cost_function;
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // add robust kernel function
        ceres::LossFunction *loss_function = new ceres:: HuberLoss(1.0);

        // add cam-pose & landmark-coord
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        // add residual
        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    cout << "bal problem file loaded..." << endl;
    cout << "bal problem have " << bal_problem.num_cameras() << " cameras and " << bal_problem.num_points() << " points. " << endl;
    cout << "Forming " << bal_problem.num_observations() << " observations." << endl;

    cout << "Solving ceres BA ..." << endl;

    // set ceres solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;

    // solve
    ceres::Solve(options, &problem, &summary);

    // output
    cout << summary.FullReport() << endl;
}