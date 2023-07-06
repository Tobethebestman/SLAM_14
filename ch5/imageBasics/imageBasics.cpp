#include <iostream>
#include <chrono>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {

    // read image from argv[1]
    // ATTENTION: after "make", use "(build)./imageBasics (absolute_path)'/home/yynn/SLAM_14/ch5/imageBasics/ubuntu.png'"
    //                           or "(build)./imageBasics (relative_path)'../ubuntu.png'" ---relative to "make" in build_file
    cv::Mat image;
    image = cv::imread(argv[1]);

    // check image
    if(image.data == nullptr) {
        cerr << "image " << argv[1] << " not exist" << endl;
        return 0;
    }

    // get basic information
    cout << "image's column = " << image.cols << ", row = " << image.rows << ", channel = " << image.channels() << endl;
    cv::imshow("image", image);
    cv::waitKey(0);

    // check type
    if (image.type() != CV_8UC1 && image.type() != CV_8UC3) {
        // neither gray nor RGB
        cout << "please input a grayscale chart or a rgb chart" << endl;
        return 0;
    }

    // traverse image
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();  // use chorno to calculate the cost time
    for (size_t y = 0; y < image.rows; y++) {
        // use cv::Mat::ptr to get row's pointer ---> row_ptr
        unsigned char *row_ptr = image.ptr<unsigned char>(y);
        for (size_t x = 0; x < image.cols; x++) {
            // get(x,y)'s pixel ----> data_ptr
            unsigned char *data_ptr = &row_ptr[x * image.channels()];
            // output pixel's channel data
            for (int c = 0; c < image.channels(); c++) {
                // data: row->y; col->x; channel->c
                unsigned char data = data_ptr[c];
            }
        }
    }
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast <chrono::duration <double>> (t2 - t1);
    cout << "traversal cost: " << time_used.count() << "s" << endl;

    // copy image: direct assignment
    cv::Mat image_another = image;  
    // change image_another, image also change
    image_another(cv::Rect(0, 0, 100, 100)).setTo(0);
    cv::imshow("image", image);
    cv::waitKey(0);

    // copy image: use function "clone"
    cv::Mat image_clone = image.clone();
    // change image_clone, image dosen't change
    image_clone(cv::Rect(0, 0, 100, 100)).setTo(255);
    cv::imshow("image", image);
    cv::imshow("image_clone", image_clone);
    cv::waitKey(0);

    cv::destroyAllWindows();
    return 0;
}