#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <boost/format.hpp>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud);

int main(int argc, char **argv) {

    // initialize
    vector<cv::Mat> colorImgs, depthImgs;
    TrajectoryType poses;

    // check pose information
    ifstream fin("../pose.txt");
    if (!fin) {
        cerr << "can't find pose information!" << endl;
        return 1;
    }

    // get image&pose information
    for (int i = 0; i < 5; i++) {
        // example: "../color/1.png": "color" --> %s; "1" --> %d; "png" --> %s.
        boost::format fmt("../%s/%d.%s") ;   
        colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
        depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "pgm").str(), -1));

        double data[7] = {0};
        for (auto &d:data) {
            fin >> d;
        }

        // data:[x,y,z,qx,qy,qz,qw]; SE(3):[q, t], q=[qw,qx,qy,qz], t=[x,y,z]
        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]), Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
    }

    // camera intrinsic parameters
    double cx = 325.5;
    double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;

    double depthScale = 1000.0;

    // initialize pointcloud
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    // request memory space, increase efficiency
    pointcloud.reserve(1000000);    

    // get pointcloud information
    for (int i = 0; i < 5; i++) {

        cout << "process image_" << (i + 1) << endl;

        // get data
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];

        // 
        for (int v = 0; v < color.rows; v++) {
            for (int u = 0; u < color.cols; u++) {
                // get depth data
                unsigned int d = depth.ptr<unsigned short>(v)[u];

                // check depth
                if (d == 0) continue;   

                // get coordinates in camera coordinate system
                Eigen::Vector3d point;  
                point[2] = double(d) / depthScale;
                point[1] = (v - cy) * point[2] / fy;
                point[0] = (u - cx) * point[2] / fx;

                // camera ----> world
                Eigen::Vector3d pointWorld = T * point;

                // zip to pointcloud
                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];       // green
                p[4] = color.data[v * color.step + u * color.channels() + 1];   // blue
                p[3] = color.data[v * color.step + u * color.channels() + 2];   // red
                pointcloud.push_back(p);
            }
        }
    }

    cout << "pointcloud has " << pointcloud.size() << " dots totally" << endl;
    showPointCloud(pointcloud);

    return 0;
}

void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) {

    // check pointcloud
    if (pointcloud.empty()) {
        cerr << "Point Cloud is EMPTY" << endl;
        return;
    }

    // draw
    // initialize pangolin
    pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_SRC_ALPHA);
    glEnable(GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
        .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
        .SetHandler(new pangolin::Handler3D(s_cam));
    
    while(pangolin::ShouldQuit() == false) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

        glPointSize(2);
        glBegin(GL_POINTS);
        for (auto &p: pointcloud) {
            glColor3f(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
            glVertex3d(p[0], p[1], p[2]);
        }
        glEnd();
        pangolin::FinishFrame();
        usleep(5000);
    }

    return;
}