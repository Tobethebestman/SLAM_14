#ifndef ROTATION_H
#define ROTATION_H

#include <algorithm>
#include <cmath>
#include <limits>

// math functions
template<typename T>
inline T dotProduct(const T x[3], const T y[3]) {   // dot production
    return (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]);
}

template<typename T>
inline T crossProduct(const T x[3], const T y[3], T result[3]) {     // cross production
    result[0] = x[1] * y[2] - x[2] * y[1];
    result[1] = x[2] * y[0] - x[0] * y[2];
    result[2] = x[0] * y[1] - x[1] * y[0]; 
}

// convert
template<typename T>
inline void angleAxisToQuaternion(const T *angle_axis, T *quaternion) {     // angle axis[a0, a1, a2] --> quaternion[q0, [q1, q2, q3]]
    const T &a0 = angle_axis[0];
    const T &a1 = angle_axis[1];
    const T &a2 = angle_axis[2];
    const T theta_squared = a0 * a0 + a1 * a1 + a2 * a2;

    if (theta_squared > T(std::numeric_limits<double>::epsilon())) {
        // for rotation vector, rotation_angle is the absolute_length of vector
        const T theta = sqrt(theta_squared);
        const T half_theta = theta * T(0.5);
        const T k = sin(half_theta) / theta;
        quaternion[0] = cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else {    // in case if theta_squared is zero, theta --> 0, k --> 0.5
        const T k(0.5);
        quaternion[0] = T(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

template<typename T>
inline void quaternionToAngleAxis(const T *quaternion, T *angle_axis) {   // quaternion --> angle axis
    const T &q1 = quaternion[1];    // normalized quaternion 
    const T &q2 = quaternion[2];
    const T &q3 = quaternion[3];
    const T sin_squared_half_theta = q1 * q1 + q2 * q2 + q3 * q3;

    if(sin_squared_half_theta > T(std::numeric_limits<double>::epsilon())) {
        const T sin_half_theta = sqrt(sin_squared_half_theta);
        const T &cos_half_theta = quaternion[0];

        const T theta = T(2.0) * ((cos_half_theta < 0.0) ? atan2(-sin_half_theta, -cos_half_theta) : atan2(sin_half_theta, cos_half_theta));
        const T k = theta / sin_half_theta;

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {    // zero rotation, theta --> 0, k --> 2
        const T k(2.0);

        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}

template <typename T>
inline void angleAxisRotatePoint(const T angle_axis[3], const T pt[3], T result[3]) {   // compute point after rotation
    const T theta2 = dotProduct(angle_axis, angle_axis);
    if (theta2 > T(std::numeric_limits<double>::epsilon())) {
        const T theta = sqrt(theta2);
        const T costheta = cos(theta);
        const T sintheta = sin(theta);
        const T theta_inverse = 1.0 / theta;

        const T w[3] = {angle_axis[0] * theta_inverse, angle_axis[1] * theta_inverse, angle_axis[2] * theta_inverse};   // w is vector<n>

        // p' = R * p = (cos_theta * I + (1 - cos_theta) * n * n_T + sin_theta * n^) * p
        //    = cos_theta * p + (1 - cos_theta) * n * (n . p) + sin_theta * (n x p)

        T w_cross_pt[3];
        crossProduct(w, pt, w_cross_pt);

        const T tmp = dotProduct(w, pt) * (T(1.0) - costheta);
        result[0] = pt[0] * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
        result[1] = pt[1] * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
        result[2] = pt[2] * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;
    } else {

        // rotation --> 0, theta --> 0.
        // cos(theta) --> 1, sin(theta) -->theta
        // R --> I + theta * n^ = I + (theta * n)^ = I + angle_axis^
        // result = p' = R * p = p + angle_axis^ * p = p + angle_axis x p

        T w_cross_pt[3];
        crossProduct(angle_axis, pt, w_cross_pt);
        result[0] = pt[0] + w_cross_pt[0];
        result[1] = pt[1] + w_cross_pt[1];
        result[2] = pt[2] + w_cross_pt[2];
    }
}

#endif