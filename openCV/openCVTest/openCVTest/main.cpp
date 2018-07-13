#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;


int main() {
	Mat test = imread("E:\\pic\\coin.jpg", CV_LOAD_IMAGE_UNCHANGED); // 和 fopen 类似
	imshow("test", test);
	waitKey();	// 没有会直接退出
}