#ifndef F_H_
#define F_H_

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <Shlwapi.h>
#pragma comment(lib, "Shlwapi.lib")

class ImgEnhance
{
public:
	ImgEnhance() {};
	~ImgEnhance() {};

	bool Test(const cv::Mat srcImg, cv::Mat &dstImg);   //entrance

private:
	bool CLHE(const cv::Mat CtdL, cv::Mat &HdL, const float fClipRatio = 0.5f);
	bool ModifyYComponent(const cv::Mat srcImg, const cv::Mat oriL, const cv::Mat dstL, cv::Mat &dstImg);
	
	bool SqrtAutoColorLevel(const cv::Mat srcImg, cv::Mat &dstImg);

	bool LogTrans(const cv::Mat srcImg, cv::Mat &dstImg, const float fRatio = 0.01f);
	bool NormLogL(const cv::Mat oriLogL, cv::Mat &dstL);

};

#endif

