#include <pangolin/pangolin.h>
#include <eigen3/Eigen/Core>
#include <unistd.h>

using namespace std;
using namespace Eigen;

// path to trajectory file(!!absolute path!!)
string trajectory_file = "/home/yynn/SLAM_14/ch3/examples/trajectory.txt";

// if function"drawTrajectory" is front of function"main", don't use the following sentence.
void drawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {

    vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;

    ifstream fin(trajectory_file);  

    // ensure we have data
    if (!fin) {

        cout << "can't find trajectory file at " << trajectory_file << endl;
        return 1;

    }

    // when read data, package to pose
    // fin.eof(): get the rare data
    while (!fin.eof()) {

        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;  // assign values by storage format
        Isometry3d Twr(Quaterniond(qw, qx, qy, qz));            // get robot --> world rotation
        Twr.pretranslate(Vector3d(tx, ty, tz));                 // get robot --> world transform
        poses.push_back(Twr);

    }
    cout << "read total " << poses.size() << " pose entires" << endl;

    // draw trajectory in pangolin
    drawTrajectory(poses);

    return 0;
}

void drawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {

    // create pangolin window and plot the trajectory
    pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);

    // glEnable: OpenGL's function, control function
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    pangolin::View &d_cam = pangolin::CreateDisplay()
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

    while (pangolin::ShouldQuit() == false) {

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);

        // draw corrd_axis of poses
        for(size_t i = 0; i < poses.size(); i++) {

            Vector3d Ow = poses[i].translation();
            Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
            Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
            Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));

            glBegin(GL_LINES);
            glColor3f(1.0, 0.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Xw[0], Xw[1], Xw[2]);
            glColor3f(0.0, 1.0, 0.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Yw[0], Yw[1], Yw[2]);
            glColor3f(0.0, 0.0, 1.0);
            glVertex3d(Ow[0], Ow[1], Ow[2]);
            glVertex3d(Zw[0], Zw[1], Zw[2]);
            glEnd();

        }

        // draw lines
        for (size_t i = 0; i < poses.size(); i++) {

            glColor3f(0.0, 0.0, 0.0);
            glBegin(GL_LINES);
            auto p1 = poses[i];
            auto p2 = poses[i + 1];
            glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
            glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
            glEnd();

        }

        pangolin::FinishFrame();
        usleep(5000);   // sleep 5 ms
    }
}