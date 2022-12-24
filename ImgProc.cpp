#include "ImgProc.h"

bool HUGHIE::ImgEnhance::Test(const cv::Mat srcImg, cv::Mat &dstImg)
{
	CV_Assert(!srcImg.empty());

	cv::Mat yuv_img, clhe_img, y_clhe;
	cv::cvtColor(srcImg, yuv_img, cv::COLOR_BGR2YUV);
	std::vector<cv::Mat>imgs;
	cv::split(yuv_img, imgs);
	CLHE(imgs[0], y_clhe, 0.5f);
	ModifyYComponent(srcImg, imgs[0], y_clhe, clhe_img);

	cv::Mat sqrt_img;
	SqrtAutoColorLevel(clhe_img, sqrt_img);

	cv::Mat log_img;
	LogTrans(sqrt_img, log_img);

	return true;
}

bool HUGHIE::ImgEnhance::CLHE(const cv::Mat CtdL, cv::Mat &HdL, const float fClipRatio)
{
	std::vector<int> vnH;
	vnH.resize(256);
	for (int i = 0; i < CtdL.rows; i++)
	{
		for (int j = 0; j < CtdL.cols; j++)
		{
			vnH[CtdL.at<uchar>(i, j)]++;
		}
	}

	int nMax = *std::max_element(vnH.begin(), vnH.end());
	int nClipThd = int(fClipRatio * nMax);

	//直方图剪枝
	int nUpperSum = 0;
	for (int i = 0; i < 256; i++)
	{
		if (vnH.at(i) > nClipThd)
		{
			nUpperSum += (vnH.at(i) - nClipThd);
			vnH.at(i) = nClipThd;
		}
	}
	int nIncrement = nUpperSum / 256;

	std::vector<float> vfP(256, 0.0f);				//PDF
	std::vector<uchar> vucLUT(256, 0);
	int nSize = int(CtdL.total());
	for (int i = 0; i < 256; i++)
	{
		vnH.at(i) += nIncrement;
		vfP.at(i) = (float)vnH.at(i) / (float)nSize;

		float fCP = std::accumulate(vfP.begin(), vfP.begin() + i, 0.0f);	//CDF
		vucLUT.at(i) = uchar(255 * fCP + 0.5f);
	}

	HdL = cv::Mat::zeros(CtdL.size(), CV_8UC1);
	for (int i = 0; i < CtdL.rows; i++)
	{
		for (int j = 0; j < CtdL.cols; j++)
		{
			HdL.at<uchar>(i, j) = vucLUT.at(CtdL.at<uchar>(i, j));
		}
	}

	vnH.clear(); vfP.clear(); vucLUT.clear();
	return true;
}

bool HUGHIE::ImgEnhance::ModifyYComponent(const cv::Mat srcImg,	const cv::Mat oriL, const cv::Mat dstL, cv::Mat &dstImg)
{
	CV_Assert(!srcImg.empty());
	CV_Assert(!oriL.empty());
	CV_Assert(!dstL.empty());

	if (srcImg.channels() == 1)
		dstImg = dstL.clone();
	else if (srcImg.channels() == 3)
	{
		cv::Mat srcImgTmp, oriLTmp, dstLTmp;
		srcImg.convertTo(srcImgTmp, CV_32F);
		oriL.convertTo(oriLTmp, CV_32F);
		dstL.convertTo(dstLTmp, CV_32F);

		std::vector<cv::Mat> vsrcImgTmp;
		cv::split(srcImgTmp, vsrcImgTmp);

		cv::Mat BY = vsrcImgTmp[0] + oriLTmp;
		cv::Mat GY = vsrcImgTmp[1] + oriLTmp;
		cv::Mat RY = vsrcImgTmp[2] + oriLTmp;

		cv::Mat B_Y = vsrcImgTmp[0] - oriLTmp;
		cv::Mat G_Y = vsrcImgTmp[1] - oriLTmp;
		cv::Mat R_Y = vsrcImgTmp[2] - oriLTmp;

		dstImg = cv::Mat::zeros(srcImg.size(), CV_8UC3);
		for (int i = 0; i < srcImg.rows; i++)
		{
			for (int j = 0; j < srcImg.cols; j++)
			{
				float fTmp = dstLTmp.at<float>(i, j) / oriLTmp.at<float>(i, j);
				int nB = min(255, int((fTmp*BY.at<float>(i, j) + B_Y.at<float>(i, j)) / 2.0f));
				int nG = min(255, int((fTmp*GY.at<float>(i, j) + G_Y.at<float>(i, j)) / 2.0f));
				int nR = min(255, int((fTmp*RY.at<float>(i, j) + R_Y.at<float>(i, j)) / 2.0f));

				nB = max(0, nB);
				nG = max(0, nG);
				nR = max(0, nR);

				dstImg.at<cv::Vec3b>(i, j) = cv::Vec3b(nB, nG, nR);
			}
		}
		srcImgTmp.release(); oriLTmp.release(); dstLTmp.release();
		BY.release(); GY.release(); RY.release();
		B_Y.release(); G_Y.release(); R_Y.release();
		vsrcImgTmp.clear();
	}
	else return false;

	return true;
}

bool HUGHIE::ImgEnhance::SqrtAutoColorLevel(const cv::Mat srcImg, cv::Mat &dstImg)
{
	CV_Assert(!srcImg.empty());
	std::vector<cv::Mat>split_imgs(3), merge_imgs(3);
	split(srcImg, split_imgs);

	for (int i = 0; i < split_imgs.size(); i++)
	{
		std::vector<int> vnHist, lutHist;
		vnHist.resize(256);
		lutHist.resize(256);
		cv::Mat vsrcImg = split_imgs[i];
		cv::Mat vdstImg = cv::Mat::zeros(vsrcImg.size(), CV_8UC1);
		double sum = 0.0;
		for (int row = 0; row < vsrcImg.rows; row++)
		{
			for (int col = 0; col < vsrcImg.cols; col++)
			{
				vnHist[vsrcImg.at<uchar>(row, col)]++;
			}
		}

		sum = vnHist[0];
		for (int k = 1; k < vnHist.size() - 1; k++)
		{
			if (k == 1)
			{
				double delta = 2 * vnHist[k];
				sum += delta;
			}
			else
			{
				double delta = 2 * sqrt((double)vnHist[k]);
				sum += delta;
			}
		}
		sum += sqrt((double)vnHist[vnHist.size() - 1]);
		double scale = 255 / sum;
		lutHist[0] = 0;
		sum = vnHist[0];
		for (int j = 1; j < vnHist.size() - 1; j++)
		{
			if (j == 1)
			{
				double delta = vnHist[j];
				sum += delta;
				lutHist[j] = (int)round(sum * scale);
				sum += delta;
			}
			else
			{
				double delta = sqrt((double)vnHist[j]);
				sum += delta;
				lutHist[j] = (int)round(sum * scale);
				sum += delta;
			}
		}
		lutHist[lutHist.size() - 1] = 255;

		for (int x = 0; x < vsrcImg.rows; x++)
		{
			for (int y = 0; y < vsrcImg.cols; y++)
			{
				vdstImg.at<uchar>(x, y) = lutHist[vsrcImg.at<uchar>(x, y)];

			}
		}
		merge_imgs[i] = vdstImg;
	}
	merge(merge_imgs, dstImg);
	merge_imgs.clear();

	return true;
}

bool HUGHIE::ImgEnhance::LogTrans(const cv::Mat srcImg, cv::Mat &dstImg, const float fRatio)
{
	CV_Assert(!srcImg.empty());
	cv::Mat yuv_img, y_img;
	cvtColor(srcImg, yuv_img, cv::COLOR_BGR2YUV);
	std::vector<cv::Mat>split_imgs;
	split(yuv_img, split_imgs);
	y_img = split_imgs[0];

	std::vector<int> vnHist;
	vnHist.resize(256);
	uchar max_pixel = 0;
	for (int i = 0; i < y_img.rows; i++)
	{
		for (int j = 0; j < y_img.cols; j++)
		{
			if (y_img.at<uchar>(i, j) > max_pixel) max_pixel = y_img.at<uchar>(i, j);
			vnHist[y_img.at<uchar>(i, j)]++;
		}
	}

	int nLevelSum = 0, nMinLevel = 0, nMaxLevel = 255;
	for (int i = 255; i >= 0; i--)
	{
		nLevelSum += vnHist[i];
		if (nLevelSum >= y_img.total()*fRatio) { nMaxLevel = i; break; }	//fRatio对应色阶上限
	}
	cv::Scalar Mean = cv::mean(y_img);
	float fMean = Mean.val[0];

	double dMin = 0.0, dMax = 0.0;
	cv::Point minPt, MaxPt;
	cv::minMaxLoc(y_img, &dMin, &dMax, &minPt, &MaxPt);

	std::vector<float> vfLUTLog;
	vfLUTLog.resize(256);
	for (int i = 0; i < 256; i++)	vfLUTLog[i] = log(0.000001f + float(i) / 255.0f);

	float fLogSum = 0.0f;
	for (int i = 0; i < y_img.rows; i++)
	{
		for (int j = 0; j < y_img.cols; j++)
		{
			fLogSum += vfLUTLog[y_img.at<uchar>(i, j)];
		}
	}
	float fAvg = exp(fLogSum / float(y_img.total()));

	std::vector<float> vfLUTG;
	vfLUTG.resize(256);
	for (int i = 0; i < 256; i++)
	{
		if (i <= fMean) vfLUTG[i] = log((float(i) / 255.0f) / fAvg + 1) / log((float(dMax) / 255.0f) / fAvg + 1);
		else
		{
			float c = 0.2*(i - fMean) / (max_pixel - fMean) + 0.3;
			vfLUTG[i] = c*log10(i + 1.0);
		}
	}

	cv::Mat dstL = cv::Mat::zeros(y_img.size(), CV_32FC1);
	for (int i = 0; i < y_img.rows; i++)
	{
		for (int j = 0; j < y_img.cols; j++)
		{
			dstL.at<float>(i, j) = vfLUTG[y_img.at<uchar>(i, j)];
		}
	}
	cv::Mat dstLTmp;
	NormLogL(dstL, dstLTmp);
	split_imgs[0] = dstLTmp;
	merge(split_imgs, dstImg);
	cvtColor(dstImg, dstImg, cv::COLOR_YUV2BGR);

	return true;
}

bool HUGHIE::ImgEnhance::NormLogL(const cv::Mat oriLogL, cv::Mat &dstL)
{
	CV_Assert(!oriLogL.empty());

	double dMin = 0.0, dMax = 0.0;
	cv::Point minPt, MaxPt;
	cv::minMaxLoc(oriLogL, &dMin, &dMax, &minPt, &MaxPt);

	dstL = cv::Mat::zeros(oriLogL.size(), CV_8UC1);
	float fInv = 255.0f / float(dMax - dMin);
	for (int i = 0; i < oriLogL.rows; i++)
	{
		for (int j = 0; j < oriLogL.cols; j++)
		{
			dstL.at<uchar>(i, j) = (uchar)((oriLogL.at<float>(i, j) - (float)dMin)*fInv + 0.5f);
		}
	}

	return true;
}
