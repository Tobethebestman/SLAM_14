// define class<BALProblem> to read BAL datasets

#pragma once

class BALProblem {
public:
    // load BAL data from text file
    explicit BALProblem(const std::string &filename, bool use_quaternions = false);

    // destructor, delete some pointers
    ~BALProblem() {
        delete[] point_index_;
        delete[] camera_index_;
        delete[] observations_;
        delete[] parameters_;
    }

    // save results to text file
    void writeToFile(const std::string &filename) const;

    // save results to ply pointcloud
    void writeToPLYFile(const std::string &filename) const;

    // data normalization
    void normalize();

    // noise addation
    void perturb(const double rotation_sigma,
                 const double translation_sigma,
                 const double point_sigma);
    
    int camera_block_size() const {return use_quaternions_ ? 10 : 9;}    // cam_pose_param_size

    int point_block_size() const {return 3;}                            // observed_point_param_size

    int num_cameras() const {return num_cameras_;}                      // camera_cnt

    int num_points() const {return num_points_;}                        // point_cnt

    int num_observations() const {return num_observations_;}            // observation_cnt

    int num_parameters() const {return num_parameters_;}                // cam_pose & landmark cnt

    const int *point_index() const {return point_index_;}               // landmark index corresponding to the observation

    const int *camera_index() const {return camera_index_;}             // camera index corresponding to the observation

    const double *observations() const {return observations_;}          // observation's head address

    const double *parameters() const {return parameters_;}              // parameter's head address

    const double *cameras() const {return parameters_;}                 // cam_param's head address

    const double *points() const {return parameters_ + camera_block_size() * num_cameras_;} // landmark's first address

    // mutable: will be modified initial value
    double *mutable_cameras() {return parameters_;}

    double *mutable_points() {return parameters_ + camera_block_size() * num_cameras_;}

    double *mutable_camera_for_observation(int i) {
        return mutable_cameras() + camera_index()[i] * camera_block_size();
    }

    double *mutable_point_for_observation(int i) {
        return mutable_points() + point_index()[i] * point_block_size();
    }

    const double *camera_for_observation(int i) const {
        return cameras() + camera_index()[i] * camera_block_size();
    }


    const double *point_for_observation(int i) const {
        return points() + point_index()[i] * point_block_size();
    }

private:
    void cameraToAngleAxisAndCenter(const double *camera, double *angle_axis, double *center) const;

    void angleAxisAndCenterToCamera(const double *angle_axis, const double *center, double *camera) const;

    int num_cameras_;
    int num_points_;
    int num_observations_;
    int num_parameters_;
    bool use_quaternions_;

    int *point_index_;
    int *camera_index_;
    double *observations_;
    double *parameters_;
};
