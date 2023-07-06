#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "common.h"
#include "rotation.h"
#include "random.h"

// DEFINE MAP
typedef Eigen::Map<Eigen::VectorXd> VectorRef;
typedef Eigen::Map<const Eigen::VectorXd> ConstVectorRef;

template<typename T>
// read file
void fscanfFile(FILE *fptr, const char *format, T *value) {
    int num_scanned = fscanf(fptr, format, value);
    if (num_scanned != 1) {
        std::cerr<< "Invalid UW data file.";
    }
}

// add noise(~N(0, sigma ** 2))
void perturbPoint3(const double sigma, double *point) {
    for (int i = 0; i < 3; i++) {
        point[i] += randNormal() * sigma;
    }
}

// find list's median value's address
double findMedian(std::vector<double> *data) {
    int n = data->size();
    std::vector<double>::iterator mid_point = data->begin() + n / 2;
    std::nth_element(data->begin(), mid_point, data->end());
    return *mid_point;
}

// BAL dataset:
// Line 1:              16 22106 83718 (cam_cnt, landmark_cnt, observation_cnt)
// Line 2 - 83719:      6  18595 3.775000e+01 4.703003e+01 (cam_index, landmark_index, observation_data)
// Line 83720 - 83864:  16 cameras' 9-dim parameters: -R(3-dim), t(3-dim), f, k1, k2
// Line 83865 - 150182: 22106 landmarks' 3-dim coordinates
BALProblem::BALProblem(const std::string &filename, bool use_quaternions) {
    FILE *fptr = fopen(filename.c_str(), "r");  // read-only

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file" << filename;
        return;
    };

    // read Line 1
    fscanfFile(fptr, "%d", &num_cameras_);
    fscanfFile(fptr, "%d", &num_points_);
    fscanfFile(fptr, "%d", &num_observations_);

    std::cout << "Header: " << num_cameras_ << " " << num_points_ << " " << num_observations_ << std::endl;

    camera_index_ = new int[num_observations_];
    point_index_ = new int[num_observations_];
    observations_ = new double[2 * num_observations_];

    num_parameters_ = 9 * num_cameras_ + 3 * num_points_;
    parameters_ = new double[num_parameters_];

    // read Line 2 - 83719
    for (int i = 0; i < num_observations_; i++) {
        fscanfFile(fptr, "%d", camera_index_ + i);
        fscanfFile(fptr, "%d", point_index_ + i);
        for (int j = 0; j < 2; j++) {
            fscanfFile(fptr, "%lf", observations_ + 2 * i + j);
        }
    }

    // read Line 83720 - 150182
    for (int i = 0; i < num_parameters_; i++) {
        fscanfFile(fptr, "%lf", parameters_ + i);
    }

    fclose(fptr);

    use_quaternions_ = use_quaternions;
    if(use_quaternions) {
        // angle_axis --> quaternion
        num_parameters_ = 10 * num_cameras_ + 3 * num_points_;
        double *quaternion_parameters = new double[num_parameters_];
        double *original_cursor = parameters_;              // old parameters with angle_axis
        double *quaternion_cursor = quaternion_parameters;  // new parameters with quaternion

        // cam_params change
        for (int i = 0; i < num_cameras_; i++) {
            // original former 4 --> quaternion former 3
            angleAxisToQuaternion(original_cursor, quaternion_cursor);
            quaternion_cursor += 4;
            original_cursor += 3;
            // other cam_param, just copy
            for (int j = 4; j < 10; j++) {
                *quaternion_cursor++ = *original_cursor++;
            }
        }

        // point_params remain
        for (int i = 0; i < 3 * num_points_; i++) {
            *quaternion_cursor++ = *original_cursor++;
        }

        // release memory
        delete[]parameters_;
        parameters_ = quaternion_parameters;
    }
}

void BALProblem::writeToFile(const std::string &filename) const {
    FILE *fptr = fopen(filename.c_str(), "w");

    if (fptr == NULL) {
        std::cerr << "Error: unable to open file" << filename;
        return;
    }

    // output Line 1
    fprintf(fptr, "%d %d %d\n", num_cameras_, num_points_, num_observations_);

    // output Line 2 - 83719
    for (int i = 0; i < num_observations_; i++) {
        fprintf(fptr, "%d %d", camera_index_[i], point_index_[i]);
        for (int j = 0; j < 2; j++) {
            fprintf(fptr, "%g", observations_[2 * i + j]);
        }
        fprintf(fptr, "\n");
    }

    // outpue Line 83720 - 83864
    for (int i = 0; i < num_cameras(); i++) {
        // save as angle-axis
        double angleaxis[9];
        if (use_quaternions_) {
            quaternionToAngleAxis(parameters_ + 10 * i, angleaxis);     // transfrom quaternion(4) --> angleaxis(3) 
            memcpy(angleaxis + 3, parameters_ + 10 * i + 4, 6 * sizeof(double));    // copy the rest cam-params
        } else {    // copy all
            memcpy(angleaxis, parameters_ + 9 * i, 9 * sizeof(double));
        }

        for (int j = 0; j < 9; j ++) {
            fprintf(fptr, "%.16g\n", angleaxis[j]);
        }
    }

    // output Line 83865 - 150182
    const double *points = parameters_ + camera_block_size() * num_cameras_;  // landmark's head address
    for (int i = 0; i < num_points(); i++) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); j++) {
            fprintf(fptr, "%.16g\n", point[j]);
        }
    }

    fclose(fptr);
}

void BALProblem::writeToPLYFile(const std::string &filename) const {    // only cam-position & landmark-position
    std::ofstream of (filename.c_str());

    of << "ply"
       << '\n' << "format ascii 1.0"
       << '\n' << "element vertex " << num_cameras_ + num_points_
       << '\n' << "property float x"
       << '\n' << "property float y"
       << '\n' << "property float z"
       << '\n' << "property uchar red"
       << '\n' << "property uchar green"
       << '\n' << "property uchar blue"
       << '\n' << "end_header" << std::endl;
    
    // export extrinsic data (i.e. camera centers) as green points
    double angle_axis[3];   // save cam-rotation pose
    double center[3];       // save cam-center position
    for (int i = 0; i < num_cameras(); i++) {
        const double *camera = cameras() + camera_block_size() * i;     // camera's head address
        cameraToAngleAxisAndCenter(camera, angle_axis, center);
        of << center[0] << ' ' << center[1] << ' ' << center[2] << "0 255 0" << '\n';
    }

    // export the structure (i.e. 3D points) as white points
    const double *points = parameters_ + camera_block_size() * num_cameras_;
    for (int i = 0; i < num_points(); i++) {
        const double *point = points + i * point_block_size();
        for (int j = 0; j < point_block_size(); j++) {
            of << point[j] << ' ';
        }

        of << "255 255 255\n";
    }
    of.close();
}

// camera's data --> camera-center-position
void BALProblem::cameraToAngleAxisAndCenter(const double *camera, double *angle_axis, double *center) const {
    VectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        quaternionToAngleAxis(camera, angle_axis);
    } else {
        angle_axis_ref = ConstVectorRef(camera, 3);
    }

    // cam_center_world = -R' * t
    Eigen::VectorXd inverse_rotation = -angle_axis_ref;
    angleAxisRotatePoint(inverse_rotation.data(), camera + camera_block_size() - 6, center);
    VectorRef(center, 3) *= -1.0;
}

// get cam-translation
void BALProblem::angleAxisAndCenterToCamera(const double *angle_axis, const double *center, double *camera) const {
    ConstVectorRef angle_axis_ref(angle_axis, 3);
    if (use_quaternions_) {
        angleAxisToQuaternion(angle_axis, camera);
    } else {
        VectorRef(camera, 3) = angle_axis_ref;
    }

    // t = -R * c
    angleAxisRotatePoint(angle_axis, center, camera + camera_block_size() - 6);
    VectorRef(camera + camera_block_size() - 6, 3) *= 1.0;
}

// normalization
void BALProblem::normalize() {
    std::vector<double> tmp(num_points_);
    Eigen::Vector3d median;
    double *points = mutable_points();  // first landmark's address
    for (int i = 0; i < 3; i++) {       // get points' coords in tmp(x,y,z)
        for (int j = 0; j < num_points_; j++) {
            tmp[j] = points[3 * j + i];
        }
        median(i) = findMedian(&tmp);   // return x,y,z's median value
    }

    for (int i = 0; i < num_points(); i++) {  // return LP in tmp
        VectorRef point(points + 3 * i, 3);
        tmp[i] = (point - median).lpNorm<1>();  // (coords - median)'s LP
    }

    const double median_absolute_deviation = findMedian(&tmp);  // return LP's median

    const double scale = 100.0 / median_absolute_deviation;

    // X = (X - median) * scale
    for (int i = 0; i < num_points_; i++) {
        VectorRef point(points + 3 * i, 3);
        point = scale * (point - median);
    }

    double *cameras = mutable_cameras();
    double angle_axis[3];
    double center[3];
    
    // old cam --> old center --> normalization --> new center --> new cam
    for(int i = 0; i < num_cameras_; i++) {
        double *camera = cameras + camera_block_size() * i;
        cameraToAngleAxisAndCenter(camera, angle_axis, center);
        VectorRef(center, 3) = scale * (VectorRef(center, 3) - median);
        angleAxisAndCenterToCamera(angle_axis, center, camera);
    }
}

// add noise
void BALProblem::perturb(const double rotation_sigma, const double translation_sigma, const double point_sigma) {
    assert(rotation_sigma >= 0.0 && translation_sigma >= 0.0 && point_sigma >= 0.0);

    // add noise to landmark
    double *points = mutable_points();
    if (point_sigma > 0.0) {
        for (int i = 0; i < num_points_; i ++) {
            perturbPoint3(point_sigma, points + 3 * i);
        }
    }

    for (int i = 0; i < num_cameras_; i++) {
        double *camera = mutable_cameras() + camera_block_size() * i;
        double angle_axis[3];
        double center[3];

        // for R: quaternion --> angle_axis, then add noise
        cameraToAngleAxisAndCenter(camera, angle_axis, center);
        if (rotation_sigma > 0.0) {
            perturbPoint3(rotation_sigma, angle_axis);
        } 
        angleAxisAndCenterToCamera(angle_axis, center, camera);

        // for t: directly
        if (translation_sigma > 0.0) {
            perturbPoint3(translation_sigma, camera + camera_block_size() - 6);
        }
    }
}

