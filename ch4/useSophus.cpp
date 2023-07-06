#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"   // need installation

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

    // rotation_matrix which rotates 90 degrees of z-axis
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();

    // or quaternion
    Quaterniond q(R);

    // they are the same
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    cout << "SO(3) from matrix : \n" << SO3_R.matrix() << endl;
    cout << "SO(3) from quaternion : \n" << SO3_q.matrix() << endl;

    // use log-mapping to get so(3)
    Vector3d so3 = SO3_R.log();
    cout << "so3 = " << so3.transpose() << endl;

    // hat(^): vector --> antisymmetry matrix
    cout << "so3^ = " << Sophus::SO3d::hat(so3) << endl;

    // vee(anti^): antisymmetry matrix --> vector
    // so3 and so3^_anti are same
    cout << "so3^_anti = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

    // add disturbance
    Vector3d so3_disturb(1e-4, 0, 0);
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(so3_disturb) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

    cout << "***************************" << endl;

    // initialize SE(3)
    Vector3d t(1, 0, 0);            // translation
    Sophus::SE3d SE3_Rt(R, t);      // from matrix
    Sophus::SE3d SE3_qt(q, t);      // from quaternion

    // they are the same
    cout << "SE3 from R,t = \n" << SE3_Rt.matrix() << endl;
    cout << "SE3 from q,t = \n" << SE3_qt.matrix() << endl;

    // use log-mapping to get se(3)
    typedef Eigen::Matrix<double, 6, 1> Vector6d;   // define a new type"---Vector6d"
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;    // (translation, rotation)

    // ^ and ^_anti
    cout << "se3^ = " << Sophus::SE3d::hat(se3) << endl;
    cout << "se3^_anti = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;    // se3 = se3^_anti

    // add disturbance
    Vector6d se3_disturb;
    se3_disturb.setZero();
    se3_disturb(0, 0) = 1e-4d;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(se3_disturb) * SE3_Rt;
    cout << "SE3 updated = \n" << SE3_updated.matrix() << endl;

    return 0;
}