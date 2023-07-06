#include <iostream>
#include <cmath>
using namespace std;

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

using namespace Eigen;

int main(int argc, char **argv) {

    // initialize rotation_matrix
    Matrix3d rotation_matrix = Matrix3d::Identity();

    // initialize rotation_vector(z-axis 45 degrees)
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1));
    
    cout.precision(3);  // control the output's precision

    // vector.matrix() and vector.toRotationMatrix() are same
    cout << "rotation_matrix= \n" << rotation_vector.matrix() << endl;

    rotation_matrix = rotation_vector.toRotationMatrix();
    cout << "rotation_matrix= \n" << rotation_matrix << endl;

    // use unit_vector to get coordinate_transform
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1, 0, 0) after rotation (by angle axis) = \n" << v_rotated.transpose() << endl;

    v_rotated = rotation_matrix * v;
    cout << "(1, 0, 0) after rotation (by matrix) = \n" << v_rotated.transpose() << endl;

    // Euler angle
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);   // ZYX
    cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

    // Euler transform
    Isometry3d T = Isometry3d::Identity();  // 4*4
    T.rotate(rotation_vector);              // rotation
    T.pretranslate(Vector3d(1, 3, 4));      // translation
    cout << "Transform matrix = \n" << T.matrix() << endl;

    // trans_matrix to corrd_trans
    Vector3d v_transformed = T * v;     // R * v + t
    cout << "v transformed = " << v_transformed.transpose() << endl;

    // quaternion
    Quaterniond q = Quaterniond(rotation_vector);       // AngleAxisd <==> Quaterniond
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;     // (x,y,z,w)

    // AngleAxisd <== Rotation_matrix
    q = Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = " << q.coeffs().transpose() << endl;

    // use quaternion to rotate an vector
    v_rotated = q * v;      // q * v * q^{-1}
    cout << "(1, 0, 0) after rotation = " << v_rotated.transpose() << endl;

    // use normal vector_mutiple 
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

    return 0;
}