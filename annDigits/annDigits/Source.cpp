#include "cv.h"
#include "highgui.h"
#include <iostream>
#include "cv.h"                             //  OpenCV �ļ�ͷ
#include <fstream>
#include "highgui.h"

#include "cvaux.h"

#include "cxcore.h"

#include "opencv2/opencv.hpp"

#include "opencv2/imgproc.hpp"
#include <cmath>
#include <Eigen>
#include <iostream>

#include <string>
struct thchar {
	uchar a;
	uchar b;
	uchar c;
};
using namespace Eigen;
using namespace std;
using namespace cv;
void test_read_pixel() {
	Mat grayim = imread("pic/1.jpg");
	VectorXf x(1729, 1);// ����ά�ȵ�����
						/*for (auto i = 0; i < x.size(); ++i) {
						x[i] = 0;
						}*/
						//cout << x << endl;

	std::cout << grayim.rows << std::endl;
	std::cout << grayim.cols << std::endl;
	for (size_t i = 0; i < grayim.rows; ++i) {
		///��ȡi����Ԫ�ص�ַ
		//short *p = grayim.ptr<short>(i);
		for (size_t j = 0; j<grayim.cols; ++j)
		{
			///ͼ���ֵ��
			/*if (p[j]>100)p[j] = 255;///����i�е�ÿһ������
			else p[j] = 0;*/
			//std::cout << (int)p[j]<<"\t" ;
			/*if (grayim.at<ushort>(i,j) > 125) {
			x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
			}
			else {
			x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
			}
			cout << x[i*grayim.cols + j]<<"\t";*/
			//cout <<(unsigned short) grayim.at<thchar>(i, j).a<<","<< (unsigned short)grayim.at<thchar>(i, j).b<<","<< (unsigned short)grayim.at<thchar>(i, j).c<<" ";
			if ((uchar)grayim.at<thchar>(i, j).a > 122)cout << 1;
			else cout << 0;
		}
		std::cout << std::endl;
	}
}
double fun(double x) {
	return 1.0 / (1 + exp(-x));
}
void train() {
	/*IplImage* test;
	test = cvLoadImage("1.jpg", 1);
	cvGet2D(test, 0, 1);
	cvNamedWindow("opencv_demo", 1);
	cvShowImage("opencv_demo", test);
	cvWaitKey(0);
	cvDestroyWindow("opencv_demo");*/
	VectorXf test[36];
	//VectorXf x(193, 1);// ����ά�ȵ�����

	for (size_t k = 0; k < 36; ++k) {
		string path = "pic/" + to_string(k + 1) + ".png";
		Mat grayim = imread(path);
		VectorXf x(193, 1);// ����ά�ȵ�����

						   //cout << x << endl;

		std::cout << grayim.rows << std::endl;
		std::cout << grayim.cols << std::endl;
		size_t ind = 0;
		x[ind++] = 1;
		for (size_t i = 0; i < grayim.rows; i = i + 3) {
			///��ȡi����Ԫ�ص�ַ
			//short *p = grayim.ptr<short>(i);

			for (size_t j = 0; j<grayim.cols; j = j + 3)
			{
				///ͼ���ֵ��
				/*if (p[j]>100)p[j] = 255;///����i�е�ÿһ������
				else p[j] = 0;*/
				//std::cout << (int)p[j]<<"\t" ;
				/*if (grayim.at<ushort>(i,j) > 125) {
				x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
				}
				else {
				x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
				}
				cout << x[i*grayim.cols + j]<<"\t";*/
				//cout <<(unsigned short) grayim.at<thchar>(i, j).a<<","<< (unsigned short)grayim.at<thchar>(i, j).b<<","<< (unsigned short)grayim.at<thchar>(i, j).c<<" ";
				int n = 0;
				if ((uchar)grayim.at<thchar>(i, j).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i, j + 1).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i, j + 2).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 1, j).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 1, j + 1).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 1, j + 2).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 2, j).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 2, j + 1).a > 122)++n;
				if ((uchar)grayim.at<thchar>(i + 2, j + 2).a > 122)++n;
				if (n > 4)x[ind++] = 1; else x[ind++] = 0;
				//if (n > 4)cout << 1; else cout << 0;
			}
			//std::cout << std::endl;
		}
		test[k] = x;
	}

	VectorXf w(193, 1);
	for (auto i = 0; i < w.size(); ++i) {
		w[i] = 0;
	}
	for (int j = 0; j < 30000000; ++j) {
		int i = j % 36;
		//if (i == 11||i==23)continue;
		double in = w.dot(test[i]);
		double out = fun(in);
		VectorXf adj(193, 1);
		if (i < 18) {
			adj = test[i] * (1 - out)*out*(1 - out);
			//	cout << i << "==" << 1 << endl;
		}

		else {
			adj = test[i] * (1 - out)*out*(0 - out);
			//	cout << i << "==" << 0 << endl;
		}

		w += adj;
	}
	cout << fun(w.dot(test[11])) << endl;
	cout << fun(w.dot(test[23])) << endl;
	ofstream out("tr.txt");
	for (size_t i = 0; i < 193; ++i) {
		out << w[i] << "\n";
	}
	out.close();
}
void test() {
	string path = "test.png";
	Mat grayim = imread(path);
	VectorXf x(193, 1);// ����ά�ȵ�����

					   //cout << x << endl;

	std::cout << grayim.rows << std::endl;
	std::cout << grayim.cols << std::endl;
	size_t ind = 0;
	x[ind++] = 1;
	for (size_t i = 0; i < grayim.rows; i = i + 3) {
		///��ȡi����Ԫ�ص�ַ
		//short *p = grayim.ptr<short>(i);

		for (size_t j = 0; j<grayim.cols; j = j + 3)
		{
			///ͼ���ֵ��
			/*if (p[j]>100)p[j] = 255;///����i�е�ÿһ������
			else p[j] = 0;*/
			//std::cout << (int)p[j]<<"\t" ;
			/*if (grayim.at<ushort>(i,j) > 125) {
			x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
			}
			else {
			x[i*grayim.cols + j] = grayim.at<ushort>(i, j);
			}
			cout << x[i*grayim.cols + j]<<"\t";*/
			//cout <<(unsigned short) grayim.at<thchar>(i, j).a<<","<< (unsigned short)grayim.at<thchar>(i, j).b<<","<< (unsigned short)grayim.at<thchar>(i, j).c<<" ";
			int n = 0;
			if ((uchar)grayim.at<thchar>(i, j).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i, j + 1).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i, j + 2).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 1, j).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 1, j + 1).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 1, j + 2).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 2, j).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 2, j + 1).a > 122)++n;
			if ((uchar)grayim.at<thchar>(i + 2, j + 2).a > 122)++n;
			if (n > 4)x[ind++] = 1; else x[ind++] = 0;
			//if (n > 4)cout << 1; else cout << 0;
		}
		//std::cout << std::endl;
	}
	VectorXf w(193, 1);
	ifstream fin("tr.txt");
	int cnt = 0;
	double kk;
	cout << "-----" << endl;
	while (!fin.eof()) {
		fin >> kk;
		//cout << cnt << endl;
		if (cnt == 193)break;
		w[cnt] = kk;
		//cout << cnt << endl;
		++cnt;
	}
	fin.close();
	cout << fun(w.dot(x));
	getchar();
}
int main()
{
	test();
	getchar();
}