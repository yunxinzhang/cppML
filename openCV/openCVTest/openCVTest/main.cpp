#include <opencv2\opencv.hpp>
#include <iostream>

using namespace cv;


int main() {
	Mat test = imread("E:\\pic\\coin.jpg", CV_LOAD_IMAGE_UNCHANGED); // �� fopen ����
	imshow("test", test);
	waitKey();	// û�л�ֱ���˳�
}