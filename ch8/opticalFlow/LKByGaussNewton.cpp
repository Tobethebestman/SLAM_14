#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace cv;
using namespace Eigen;

class OpticalFlowTracker{
public:
    OpticalFlowTracker(
        const Mat &img1_, const Mat &img2_, const vector<KeyPoint> &kp1_,
        vector<KeyPoint> &kp2_,
        vector<bool> &success_,
        bool inverse_ = true, bool has_initial_ = false
        ) : img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}

    void calculateOpticalFlow(const Range &range);

private:
    const Mat &img1;
    const Mat &img2;
    const vector<KeyPoint> &kp1;
    vector<KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};

void OpticalFlowSingleLevel(
    const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1,  // @param [in]
    vector<KeyPoint> &kp2,      // @param [in|out]
    vector<bool> &success,      // @param [out]     if a keypoint[i] is tracked successfully, success[i] = 1.
    bool inverse = false,       // @param [in]      whether use inverse formulation. 
    bool has_initial = false    // @param [in]      whether has initialized
);

void OpticalFlowMultiLevel(
    const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1,  // @param [in]
    vector<KeyPoint> &kp2,      // @param [in|out]
    vector<bool> &success,      // @param [out]     if a keypoint[i] is tracked successfully, success[i] = 1.
    bool inverse = false        // @param [in]      whether use inverse formulation.   
);

// get gray value in diff-scales from the referenced image(bi-linear interpolated)
inline float GetPixelValue(const Mat &img, float x, float y);

int main(int argc, char **argv) {
    // check
    if (argc != 3) {
        cout << "usage: LKByGaussNewton.cpp LK1.png LK2.png" << endl;
        return 1;
    }

    // read image data
    Mat img1 = imread(argv[1], 0);
    Mat img2 = imread(argv[2], 0);

    // get img1's keypoints
    vector<KeyPoint> kp1;
    Ptr<GFTTDetector> detcetor = GFTTDetector::create(500, 0.01, 20);
    detcetor->detect(img1, kp1);

    // track img1's keypoints in img2 with single level
    vector<KeyPoint> kp2_single;
    vector<bool> success_single;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 -t1);
    cout << "use single level track cost: " << time_used.count() << " seconds." << endl;

    // track img1's keypoints in img2 with multi levels
    vector<KeyPoint> kp2_multi;
    vector<bool> success_multi;
    t1 = chrono::steady_clock::now();
    OpticalFlowMultiLevel(img1, img2, kp1, kp2_multi, success_multi);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>> (t2 -t1);
    cout << "use multi levels track cost: " << time_used.count() << " seconds." << endl;

    // draw tracking and show
    Mat img2_single;
    cvtColor(img2, img2_single, CV_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if(success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    Mat img2_multi;
    cvtColor(img2, img2_multi, CV_GRAY2BGR);
    for (int i = 0; i < kp2_multi.size(); i++) {
        if(success_multi[i]) {
            cv::circle(img2_multi, kp2_multi[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_multi, kp1[i].pt, kp2_multi[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    imshow("Single Level", img2_single);
    imshow("Multi Levels", img2_multi);
    waitKey(0);
    imwrite("/home/yynn/SLAM_14/ch8/opticalFlow/result/LK_Single.png", img2_single);
    imwrite("/home/yynn/SLAM_14/ch8/opticalFlow/result/LK_Multi.png", img2_multi);
    
    return 0;
}

inline float GetPixelValue(const Mat &img, float x, float y) {  
    // check boundary
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > img.cols - 1) x = img.cols - 2;
    if (y > img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x);
    float yy = y - floor(y);
    int x_a1 = min(img.cols - 1, int(x) + 1);
    int y_a1 = min(img.rows - 1, int(y) + 1);
    int x_a0 = int(x);
    int y_a0 = int(y);

    return (1 - xx) * (1- yy) * img.at<uchar>(y_a0, x_a0)
    + xx * (1 - yy) * img.at<uchar>(y_a0, x_a1)
    + (1 - xx) * yy * img.at<uchar>(y_a1, x_a0)
    + xx * yy * img.at<uchar>(y_a1, x_a1);
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // initialize parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto kp = kp1[i];
        double dx = 0, dy = 0;
        if (has_initial) {
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true;   // if kp1[i] is tracked successfully

        // Gauss-Newton iterations
        Matrix2d H = Matrix2d::Zero();  // Hessian matrix
        Vector2d b = Vector2d::Zero();  // bias vector
        Vector2d J;                     // Jacobian
        for(int iter = 0; iter < iterations; iter++) {
            // reset
            if (inverse == false) {
                H = Matrix2d::Zero(); 
                b = Vector2d::Zero();
            } else {
                b = Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++) {
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) - GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);
                    if (inverse == false) {
                        J = -1.0 * Vector2d(
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) - 
                                   GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) - 
                                   GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))       
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same
                        J = -1.0 * Vector2d(
                            0.5 * (GetPixelValue(img1, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) - 
                                   GetPixelValue(img1, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                            0.5 * (GetPixelValue(img1, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) - 
                                   GetPixelValue(img1, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))       
                        );
                    }
                    // compute H, b; set cost
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        H += J * J.transpose();     // also update H
                    }
                }
            }

            // compute update
            Vector2d update = H.ldlt().solve(b);

            if (isnan(update[0])) {     // check
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {  // check
                break;
            }

            // update dx & dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            // converge
            if (update.norm() < 1e-2) {
                break;
            }
        }

        // set succ & kp2[i].pt
        success[i] = succ;
        kp2[i].pt = kp.pt + Point2f(dx, dy);
    }
}

void OpticalFlowSingleLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<bool> &success, bool inverse, bool has_initial) {
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1, img2, kp1, kp2, success, inverse, has_initial);
    // calls function "calculateOpticalFlow" in parallel
    parallel_for_(Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowMultiLevel(const Mat &img1, const Mat &img2, const vector<KeyPoint> &kp1, vector<KeyPoint> &kp2, vector<bool> &success, bool inverse) {
    // initialize parameters
    int pyramids = 4;
    double pyramids_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    vector<Mat> pyr1, pyr2;
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr, cv::Size(pyr1[i - 1].cols * pyramids_scale, pyr1[i - 1].rows * pyramids_scale));
            cv::resize(pyr2[i - 1], img2_pyr, cv::Size(pyr2[i - 1].cols * pyramids_scale, pyr2[i - 1].rows * pyramids_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "build pyramid cost: " << time_used.count() << " seconds." << endl;

    // coarse-to-fine LK tracking in pyramids
    vector<KeyPoint> kp1_pyr, kp2_pyr;
    for (auto &kp: kp1) {
        auto kp_top = kp;
        kp_top.pt *= scales[pyramids - 1];
        kp1_pyr.push_back(kp_top);
        kp2_pyr.push_back(kp_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        t1 = chrono::steady_clock::now();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], kp1_pyr, kp2_pyr, success, inverse, true);
        t2 = chrono::steady_clock::now();
        time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
        cout << "track pyramid " << level << " cost: " << time_used.count() << " seconds." << endl;

        if (level > 0) {
            for (auto &kp: kp1_pyr) kp.pt /= pyramids_scale;
            for (auto &kp: kp2_pyr) kp.pt /= pyramids_scale;
        }
    }

    // get fine result
    for (auto &kp: kp2_pyr) kp2.push_back(kp);
}
