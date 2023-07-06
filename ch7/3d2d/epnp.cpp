#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void findFeatureMatches(
    const Mat &img1, const Mat &img2,
    vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches
);

Point2d pixel2cam(
    const Point2d &p,
    const Mat &K
);

int main(int argc, char **argv) {
    // check
    if (argc != 4) {
        cout << "usage: epnp img1 img2 depth1" << endl;
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
    vector<Point3f> pts_3d;     // camera-system,  the matched left point's location in left camera system with valid depth
    vector<Point2f> pts_2d;     // pixel-system, the matched right point's pixel coordinate in right pixel-system
    for (DMatch m:matches) {
        // get matched point's depth
        ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
        // check
        if (d == 0) continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);   // p1: normalized former-camera-system 
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(keypoints2[m.trainIdx].pt);
    }
    cout << "3d-2d pairs: " << pts_3d.size() << endl;

    // epnp solve R&t
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Mat r, t;           // r: rotation vector
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    Rodrigues(r, R);    // vector --> matrix   
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> (t2 - t1);
    cout << "use EPnP solve cost : " << time_used.count() << " seconds." << endl;

    cout << "R = " << R << endl;
    cout << "t = " << t << endl;

    return 0;
}

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