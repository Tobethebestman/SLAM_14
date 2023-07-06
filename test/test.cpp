#include <iostream>
#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/features2d/features2d.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <opencv4/opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

int a() const {return 3;}

int main(int agrc, char **argv) {
    int t = a();
    cout << t << endl;
    return 0;
}