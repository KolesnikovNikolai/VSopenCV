// kurs.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <stdio.h>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;


int main()
{
	double minHessian = 250.0;
	double max_diff = 0; 
	double min_diff = 100;
	FlannBasedMatcher matcher;
	Ptr<SURF> detector = SURF::create(minHessian);
	Mat des_object, des_image;
	Mat img = imread("big.jpg", IMREAD_GRAYSCALE);
	Mat img_object = imread("object.jpg", IMREAD_GRAYSCALE);
	Mat img_matches;
	std::vector<KeyPoint> p_object, p_image;
	std::vector< DMatch > matches;
	std::vector< DMatch > good_matches;
	std::vector<Point2f> obj;
	std::vector<Point2f> im;
	std::vector<Point2f> obj_angle(4);
	std::vector<Point2f> im_angle(4);

	
	if (!img_object.data || !img.data)
	{
		std::cout << "Image not found." << std::endl; 
		system("pause");
		return 0;
	}
	//дескрипторы с использованием SURF
	detector->detectAndCompute(img, Mat(), p_image, des_image);
	detector->detectAndCompute(img_object, Mat(), p_object, des_object);
	//FLANN алгоритм
	matcher.match(des_object, des_image, matches);
	//Поиск расстояний
	for (int i = 0; i < des_object.rows; i++)
	{
		double diff = matches[i].distance;
		if (diff < min_diff) 
			min_diff = diff;
		if (diff > max_diff) 
			max_diff = diff;
	}
	//Отрисовка хороших
	for (int i = 0; i < des_object.rows; i++)
	{
		if (min_diff * 3 > matches[i].distance)
		{
			good_matches.push_back(matches[i]);
		}
	}
	drawMatches(img_object, p_object, img, p_image,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//Отрисовка объекта
	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//Точки из хороших совпадений
		obj.push_back(p_object[good_matches[i].queryIdx].pt);
		im.push_back(p_image[good_matches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, im, RANSAC);
	//Получаем углы
	obj_angle[0] = cvPoint(0, 0); 
	obj_angle[1] = cvPoint(img_object.cols, 0);
	obj_angle[2] = cvPoint(img_object.cols, img_object.rows); 
	obj_angle[3] = cvPoint(0, img_object.rows);
	perspectiveTransform(obj_angle, im_angle, H);
	//Рисуем линии
	line(img_matches, im_angle[0] + Point2f(img_object.cols, 0), im_angle[1] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, im_angle[1] + Point2f(img_object.cols, 0), im_angle[2] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, im_angle[2] + Point2f(img_object.cols, 0), im_angle[3] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	line(img_matches, im_angle[3] + Point2f(img_object.cols, 0), im_angle[0] + Point2f(img_object.cols, 0), Scalar(0, 255, 0), 4);
	imshow("Результат", img_matches);
	waitKey(0);
	system("pause");
	return 0;
}
