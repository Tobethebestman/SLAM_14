#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void findFeatureMatches(
    const Mat &img1, const Mat &img2,
    vector<KeyPoint> &keypoints1,
    vector<KeyPoint> &keypoints2,
    vector<DMatch> &matches
);

void poseEstimation(
    vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2,
    vector<DMatch> matches,
    Mat &R, Mat &t,
    const Mat &K
);

void triangulation(
    const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2, 
    const vector<DMatch> matches, 
    const Mat &R, const Mat &t, const Mat &K,
    vector<Point3d> &points
);

inline Scalar getColor(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    // Scalar():BGR. the feature point nearer, the color more red
    return Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

Point2d pixel2cam(const Point2d &p, const Mat &K);

// *******************************************************************************************************//
// *****************************************//
int main(int argc, char **argv) {
    // check
    if (argc != 3) {
        cout << "usage: pose_estimation img1 img2" << endl;
        return 1;
    }

    // read image
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    assert(img1.data && img2.data && "Can not load images!");

    // camera intrinsic
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    // feature matching
    vector<KeyPoint> keypoints1;    // pixel coordinate system
    vector<KeyPoint> keypoints2;
    vector<DMatch> matches;
    findFeatureMatches(img1, img2, keypoints1, keypoints2, matches);
    cout << "find matches: " << matches.size() << endl;

    // pose estimation
    Mat R, t;
    poseEstimation(keypoints1, keypoints2, matches, R, t, K);

    // triangulation
    vector<Point3d> points;         // camera coordinate system
    triangulation(keypoints1, keypoints2, matches, R, t, K, points);

    // plot
    Mat img1_plot = img1.clone();
    Mat img2_plot = img2.clone();
    for (int i = 0; i < matches.size(); i++) {
        // plot1
        float depth1 = points[i].z;
        cout << "feature point " << i << "'s depth in img1: " << depth1 << endl;
        circle(img1_plot, keypoints1[matches[i].queryIdx].pt, 2, getColor(depth1), 2);

        // plot2
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cout << "feature point " << i << "'s depth in img2: " << depth2 << endl;
        cout << "\n" << endl;
        circle(img2_plot, keypoints2[matches[i].trainIdx].pt, 2, getColor(depth2), 2);
    }

    imshow("img_1", img1_plot);
    imshow("img_2", img2_plot);
    waitKey(0);

    return 0;
}
// *****************************************//
// *******************************************************************************************************//

void findFeatureMatches(const Mat &img1, const Mat &img2, vector<KeyPoint> &keypoints1, vector<KeyPoint> &keypoints2, vector<DMatch> &matches ) {
    // initialization
    Mat descriptors1, descriptors2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    // 1. detect corner points' location
    detector->detect(img1, keypoints1);
    detector->detect(img2, keypoints2);

    // 2. compute BRIEF descriptor
    descriptor->compute(img1, keypoints1, descriptors1);
    descriptor->compute(img2, keypoints2, descriptors2);

    // 3. matching
    vector<DMatch> match;
    matcher->match(descriptors1, descriptors2, match);

    // 4. select matches
    double min_dist = 1000, max_dist = 0;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("--Max dist: %f \n", max_dist);
    printf("--Min dist: %f \n", min_dist);

    for (int i = 0; i < descriptors1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d ((p.x - K.at<double>(0, 2)) / K.at<double>(0, 0), (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

void poseEstimation(vector<KeyPoint> keypoints1, vector<KeyPoint> keypoints2, vector<DMatch> matches, Mat &R, Mat &t, const Mat &K) {
    // vector<Keypoint> ----> vector<Point2f>
    vector<Point2f> points1, points2;   // pixel system

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);  // queryIdx: index for one set
        points2.push_back(keypoints2[matches[i].trainIdx].pt);  // trainIdx: idnex for another set
    }

    // compute matrix F
    Mat F_matrix;
    F_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "F_matrix: " << endl << F_matrix << endl;

    // compute matrix E
    Point2d principal_point(325.1, 249.7);  // camera light center
    double focal_length = 521;  // camera focal length
    Mat E_matrix;
    E_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "E_matrix: " << endl << E_matrix << endl;

    // compute matrix H
    Mat H_matrix;
    H_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "H_matrix: " << endl << H_matrix << endl;

    // recover R&t from E_matrix
    recoverPose(E_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}

void triangulation(const vector<KeyPoint> &keypoints1, const vector<KeyPoint> &keypoints2,  const vector<DMatch> matches,  const Mat &R, const Mat &t, const Mat &K, vector<Point3d> &points) {
    // T1: projection matrix of left camera, T2: projection matrix of right camera
    Mat T1 = (Mat_<float>(3, 4) << 
        1, 0, 0, 0,
        0, 1, 0, 0, 
        0, 0, 1, 0);
    
    Mat T2 = (Mat_<float>(3, 4) << 
        R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
        R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
        R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0));
    
    // pixel-->cam(normalized)
    vector<Point2f> pts1, pts2;
    for (DMatch m:matches) {
        pts1.push_back(pixel2cam(keypoints1[m.queryIdx].pt, K));
        pts2.push_back(pixel2cam(keypoints2[m.trainIdx].pt, K));
    }

    Mat pts4d;  // result, cam system
    triangulatePoints(T1, T2, pts1, pts2, pts4d);

    // cam(normalized)-->cam 
    // column-by-column
    for (int i = 0; i < pts4d.cols; i++) {
        Mat x = pts4d.col(i);
        x /= x.at<float>(3, 0);
        Point3d P(
            x.at<float>(0, 0),
            x.at<float>(1, 0),
            x.at<float>(2, 0)
        );
        points.push_back(P);
    }
}