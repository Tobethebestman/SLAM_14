#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <eigen3/Eigen/Core>
#include "sophus/se3.hpp"
#include <chrono>

using namespace std;
using namespace cv;
using namespace Eigen;

#define VecVector2d vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
#define VecVector3d vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>

// BA
// typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
// typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void findFeatureMatches(
    const Mat &img1, const Mat &img2,
    vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches
);

Point2d pixel2cam(
    const Point2d &p,
    const Mat &K
);

void bundleAdjustmentByGaussNewton(
    const VecVector3d &points_3d, const VecVector2d &points_2d,
    const Mat &K,
    Sophus::SE3d &pose
);

// **********************************************************************//
// ************************** //
int main(int argc, char **argv) {
    // check
    if (argc != 5) {
        // ./hpnp ../1.png ../2.png ../depth1.png ../depth2.png
        cout << "usage: epnp img1 img2 depth1 depth2" << endl;
        return 1;
    }

    // read image 
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img1.data && img2.data && "Can't load images!");

    // match feature points
    vector<KeyPoint> keypoints1, keypoints2;    // pixel-system
    vector<DMatch> matches;
    findFeatureMatches(img1, img2, keypoints1, keypoints2, matches);
    cout << "matched pairs: " << matches.size() << endl;

    // build 3d points
    Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);      // depth image
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  // camera intrinsic parameters
    vector<Point3f> pts_3d;     // former camera-system,  the matched left point's location in left camera system with valid depth
    vector<Point2f> pts_2d;     // latter pixel-system, the matched right point's pixel coordinate in right pixel-system
    for (DMatch m:matches) {
        // get matched point's depth
        ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        // check
        if (d == 0) continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);   // p1: normalized former camera-system 
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    // epnp solve R&t
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;           // r: rotation vector
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    // cout << "pts_3d: " << pts_3d << endl;
    Mat R;
    Rodrigues(r, R);    // vector --> matrix   
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "use EPnP solve cost : " << time_used.count() << " seconds." << endl;

    cout << "R = " << R << endl;
    cout << "t = " << t << endl;

    // handmade GaussNewton solve R&t
    VecVector3d pts_3d_eigen;           // former camera-system
    VecVector2d pts_2d_eigen;           // latter pixel-system
    for (size_t i = 0; i < pts_3d.size(); ++i) {                                    // WARNING:type i = 0, not type i.
        pts_3d_eigen.push_back(Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    cout << "dealing bundle adjustment by gauss newton" << endl;
    Sophus::SE3d pose_gn;
    t1 = chrono::steady_clock::now();
    bundleAdjustmentByGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "solve pnp by gauss-newton cost time: " << time_used.count() << " seconds." << endl;

    return 0;
}
// ************************** //
// **********************************************************************//

void findFeatureMatches(const Mat &img1, const Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches) {

    // initialization
    Mat descriptor1, descriptor2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // detect corner point's location
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // generate descriptor
    descriptor->compute(img1, keypoints1, descriptor1);
    descriptor->compute(img2, keypoints2, descriptor2);

    // match features
    vector<DMatch> match;
    matcher->match(descriptor1, descriptor2, match);
    
    // select matching
    double min_dist = 1000, max_dist = 0;
    for (int i = 0; i < descriptor1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("Max dist: %f \n", max_dist);
    printf("Min dist: %f \n", min_dist);

    for (int i = 0; i < descriptor1.rows; i++) {
        if(match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d 
    (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void bundleAdjustmentByGaussNewton(const VecVector3d &points_3d, const VecVector2d &points_2d, const Mat &K, Sophus::SE3d &pose) {
    // initialization
    typedef Matrix<double, 6, 1> Vector6d;

    const int iterations = 10;
    double cost = 0, lastCost = 0;

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);

    for (int iter = 0; iter < iterations; iter++) {
        // initialization
        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < points_3d.size(); i++) {
            Vector3d pc = pose * points_3d[i];          // latter camera-system
            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;

            Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);    // theoretical value
            Vector2d e = points_2d[i] - proj;                                   // points_2d[i]: measurements, latter pixel system

            Matrix<double, 2, 6> J;     // Jacobian matrix
            J << -fx * inv_z, 0, fx * pc[0] * inv_z2, fx * pc[0] * pc[1] * inv_z2, -fx - fx * pc[0] * pc[0] * inv_z2, fx * pc[1] * inv_z,
                0, -fy * inv_z, fy * pc[1] * inv_z2, fy + fy * pc[1] * pc[1] * inv_z2, -fy * pc[0] * pc[1] * inv_z2, -fy * pc[0] * inv_z;
            
            // iteration
            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += e.squaredNorm();
        }

        // solve dx
        Vector6d dx;
        dx = H.ldlt().solve(b);

        // check whether result is nan
        if (isnan(dx[0])) {
            cout << "result is nan" << endl;
            break;
        }

        // check whether updation is valid
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration: " << iter << " cost: " << cost << endl;
        if (dx.norm() < 1e-6) {
            cout << "result has converged" << endl;
            break;
        }
    }

    cout << "pose by g-n: " << pose.matrix() << endl;
}