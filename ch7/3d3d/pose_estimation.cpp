#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Geometry>
#include <eigen3/Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include "sophus/se3.hpp"

using namespace std;
using namespace cv;
using namespace Eigen;

void findFeatureMatches(
    const Mat &img1, const Mat &img2,
    vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches
);

Point2d pixel2cam(const Point2d &p, const Mat &K);

void poseEstimationByICP(
    const vector<Point3f> &pts1, const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

void poseEstimationByBA(
    const vector<Point3f> &pts1, const vector<Point3f> &pts2,
    Mat &R, Mat &t
);

class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    virtual void oplusImpl(const double *update) override {
        Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}

    virtual bool write(ostream &out) const override {}
};

class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Vector3d, VertexPose> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjectXYZRGBDPoseOnly(const Vector3d &point) : _point(point) {}

    virtual void computeError() override {
        const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
        _error = _measurement - pose->estimate() * _point;
    }

    virtual void linearizeOplus() override {
        const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
        Sophus::SE3d T = pose->estimate();
        Vector3d xyz_trans = T * _point;
        _jacobianOplusXi.block<3, 3>(0, 0) = -Matrix3d::Identity();
        _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
    }

    bool read(istream &in) override {}

    bool write(ostream &out) const override {}

protected:
    Vector3d _point;
};

/************************************************************************/
/***********************************************/
int main(int argc, char **argv) {
    if (argc != 5) {
        // ./3d3d ../1.png ../2.png ../depth1.png ../depth2.png
        cout << "usage: pose_estimation.cpp img1 img2 depth1 depth2" << endl;
        return 1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints1, keypoints2;    // left-cam-system & right-cam-system
    vector<DMatch> matches;
    findFeatureMatches(img1, img2, keypoints1, keypoints2, matches);
    cout << "find feature matches: " << matches.size() << endl;

    // build 3d points: pts1 & pts2
    Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED);
    Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249,7, 0, 0, 1);
    vector<Point3f> pts1, pts2;

    for(DMatch m:matches) {
        ushort d1 = depth1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        ushort d2 = depth2.ptr<unsigned short>(int(keypoints2[m.trainIdx].pt.y))[int(keypoints2[m.trainIdx].pt.x)];
        if (d1 == 0 || d2 == 0) continue;   // check, MIND: opreator is "||" not "&&"

        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);   // matched points in former-cam-system
        Point2d p2 = pixel2cam(keypoints2[m.trainIdx].pt, K);   // matched points in latter-cam-system

        float dd1 = float(d1) / 5000.0;
        float dd2 = float(d2) / 5000.0;

        pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
        pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
    }

    cout << "3d-3d pairs: " << pts1.size() << endl;

    Mat R, t;
    poseEstimationByICP(pts1, pts2, R, t);
    cout << "ICP via SVD results: " << endl;
    cout << "R = " << R << endl;
    cout << "t = " << t << endl;
    cout << "inv_R = " << R.t() << endl;
    cout << "inv_t = " << -R.t() * t << endl;

    cout << "calling BA solve ICP" << endl;
    poseEstimationByBA(pts1, pts2, R, t);

    // verify p1 = R * p2 + t
    for (int i = 0; i < 5; i++) {
        cout << "p1 = " << pts1[i] << endl;
        cout << "p2 = " << pts2[i] << endl;
        cout << "R * p2 + t = " << 
            R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
            << endl;
        cout << endl;
    }

    return 0;
}
/***********************************************/
/************************************************************************/

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

void poseEstimationByICP(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t) {
    // 1. compute centroid coordinates
    Point3f p1, p2;
    int N = pts1.size();
    for (int i = 0; i < N; i++) {
        p1 += pts1[i];
        p2 += pts2[i];
    }
    p1 = Point3f(Vec3f(p1) / N);    // p1 = p1 / N
    p2 = Point3f(Vec3f(p2) / N);
    
    // 2. compute decentroid coordinates
    vector<Point3f> q1(N), q2(N);
    for (int i = 0; i < N; i++) {
        q1[i] = pts1[i] - p1;
        q2[i] = pts2[i] - p2;
    }

    // 3. compute W_matrix
    Matrix3d W = Matrix3d::Zero();
    for (int i = 0; i < N; i++) {
        W += Vector3d(q1[i].x, q1[i].y, q1[i].z) * Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose();
    }
    cout << "W = " << W << endl;

    // 4. SVD on W
    JacobiSVD<Matrix3d> svd(W, ComputeFullU | ComputeFullV);
    Matrix3d U = svd.matrixU();
    Matrix3d V = svd.matrixV();
    cout << "U = " << U << endl;
    cout << "V = " << V << endl;

    // 5. get R & t
    Matrix3d R_ = U * (V.transpose());
    if (R_.determinant() < 0) {     // ensure R's determinant > 0
        R_ = -R_;
    }
    Vector3d t_ = Vector3d(p1.x, p1.y, p1.z) - R_ * Vector3d(p2.x, p2.y, p2.z);

    // 6. convert to cv::Mat
    R = (Mat_<double>(3, 3) << 
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}   

void poseEstimationByBA(const vector<Point3f> &pts1, const vector<Point3f> &pts2, Mat &R, Mat &t) {
    typedef g2o::BlockSolverX Block;
    unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverDense<Block::PoseMatrixType>());
    unique_ptr<Block> solver_ptr(new Block(move(linearSolver)));
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(move(solver_ptr));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // vertex
    VertexPose *pose = new VertexPose();
    pose->setEstimate(Sophus::SE3d());
    pose->setId(0);
    optimizer.addVertex(pose);

    // edge
    for (size_t i = 0; i < pts1.size(); i++) {
        EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
        edge->setVertex(0, pose);
        edge->setMeasurement(Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));
        edge->setInformation(Matrix3d::Identity());
        optimizer.addEdge(edge);
    }

    // optimize
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "optimization cost time: " << time_used.count() << " seconds." << endl;
    cout << "T =\n " << pose->estimate().matrix() << endl;

    // convert to cv::Mat
    Matrix3d R_ = pose->estimate().rotationMatrix();
    Vector3d t_ = pose->estimate().translation();
    R = (Mat_<double>(3, 3) << 
        R_(0, 0), R_(0, 1), R_(0, 2),
        R_(1, 0), R_(1, 1), R_(1, 2),
        R_(2, 0), R_(2, 1), R_(2, 2));
    t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}