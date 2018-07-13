#include <opencv2\opencv.hpp>
#include <string>
#include <cstdint>
using namespace cv;
using namespace std;

string dir = "e:/pic/";

void test01_read_color() {
	Mat readColor = imread("e:/pic/coin.jpg", CV_LOAD_IMAGE_COLOR);
	imshow("color", readColor);
	Mat readGray = imread("e:/pic/coin.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	imshow("gray", readGray);
	imwrite(dir + "gray.jpg", readGray);
	waitKey();
}

void test02_window() {
	Mat fit = imread(dir + "coin.jpg", CV_LOAD_IMAGE_COLOR);
	Mat fixed = imread(dir + "coin.jpg", CV_LOAD_IMAGE_UNCHANGED);
	namedWindow("fit", CV_WINDOW_FREERATIO);
	namedWindow("fixed", CV_WINDOW_AUTOSIZE);
	imshow("fit", fit);
	imshow("fixed", fixed);

	resizeWindow("fit", fit.cols / 2, fit.rows / 2);
	resizeWindow("fixed", fixed.cols / 2, fit.rows / 2);

	moveWindow("fit", 200, 400);
	moveWindow("fixed", 200 + fit.cols / 2, 400);

	waitKey();
}

void test03_pixels_gray() {
	Mat original = imread(dir + "coin.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat modified = imread(dir + "coin.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	for (size_t r = 0; r < modified.rows; ++r) {
		for (size_t c = 0; c < modified.cols; ++c) {
			modified.at<uint8_t>(r, c) *= 0.5;
		}
	}

	imshow("original", original);
	imshow("modifed", modified);
	waitKey();
}

void test04_pixels_color() {
	Mat original = imread(dir + "coin.jpg", CV_LOAD_IMAGE_COLOR);
	Mat modified = imread(dir + "coin.jpg", CV_LOAD_IMAGE_COLOR);

	for (size_t r = 0; r < modified.rows; ++r) {
		for (size_t c = 0; c < modified.cols; ++c) {
			modified.at<cv::Vec3b>(r, c)[0] = 255 - modified.at<cv::Vec3b>(r, c)[0];
			modified.at<cv::Vec3b>(r, c)[1] = 255 - modified.at<cv::Vec3b>(r, c)[1];
			modified.at<cv::Vec3b>(r, c)[2] = 255 - modified.at<cv::Vec3b>(r, c)[2];
		}
	}

	imshow("original", original);
	imshow("modifed", modified);
	waitKey();
}

void test05_split_merge() {
	Mat original = imread(dir + "coin.jpg", CV_LOAD_IMAGE_COLOR);
	Mat splitChannels[3];
	split(original, splitChannels);
	imshow("B", splitChannels[0]);
	imshow("G", splitChannels[1]);
	imshow("R", splitChannels[2]);

	splitChannels[2] = Mat::zeros(splitChannels[2].size(), CV_8UC1);
	Mat output;
	merge(splitChannels, 3, output);
	imshow("merged", output);
	waitKey();
}

int main() {
	
}