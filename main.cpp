#include <baseapi.h>
// #include <allheaders.h>
//#include "sys/time.h"
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/core/core.hpp>
#include "putText.h"


using namespace cv;
using namespace std;
using namespace cv::dnn;



void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
	std::vector<RotatedRect>& detections, std::vector<float>& confidences);

string UTF8ToGB(const char* str);
wchar_t * Utf_8ToUnicode(const char* szU8);
char* UnicodeToAnsi(const wchar_t* szStr);

int main(int argc, char** argv)
{


	float confThreshold = 0.5;    ///置信度阈值
	float nmsThreshold = 0.4;  ///非最大抑制阈值
	int inpWidth = 320;   ///通过调整大小到特定宽度来预处理输入图像。它应该是32的倍数
	int inpHeight = 320;  ///通过调整大小到特定高度来预处理输入图像。它应该是32的倍数
	String model = "frozen_east_text_detection.pb";
	const char *text = NULL;
	CV_Assert(!model.empty());

	// 加载网络模型
	Net net = readNet(model);

/////////////////OCR
// 初始化 tesseract OCR 
	tesseract::TessBaseAPI *myOCR =
		new tesseract::TessBaseAPI();

	printf("Tesseract-ocr version: %s\n",
		myOCR->Version());  //输出Tesseract-ocr 版本

	const char* datapath = "D:\\opencv实践\\tesseract_4.0\\tessdata";  //设置字库模型的调用地址
	if (myOCR->Init(datapath, "chi_sim", tesseract::OEM_TESSERACT_ONLY))  //加载中文检测模型
	{
		fprintf(stderr, "Could not initialize tesseract.\n");
		exit(1);
	}

	tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(7); // treat the image as a single text line
	myOCR->SetPageSegMode(pagesegmode);


	//定义显示参数
	int fontFace = FONT_HERSHEY_PLAIN;
	double fontScale = 2;
	int thickness = 2;
/////////////////////////////////////////

	VideoCapture cap;

	cap.open(0);

	static const std::string kWinName = "EAST文本检测";
	//namedWindow(kWinName, WINDOW_NORMAL);

	std::vector<Mat> outs;
	std::vector<String> outNames(2);
	outNames[0] = "feature_fusion/Conv_7/Sigmoid";
	outNames[1] = "feature_fusion/concat_3";

	Mat frame, blob,frameclone;
	
	while (waitKey(1) < 0)
	{
		
		cap >> frame;
		if (frame.empty())
		{
			waitKey();
			break;
		}
		frameclone = frame.clone();
		blobFromImage(frame, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
		net.setInput(blob);
		net.forward(outs, outNames);

		Mat scores = outs[0];
		Mat geometry = outs[1];

		// 定义预测框及其置信度
		std::vector<RotatedRect> boxes;
		std::vector<float> confidences;
		decode(scores, geometry, confThreshold, boxes, confidences);

		// 应用非最大抑制程序
		std::vector<int> indices;
		NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

		Mat image;
		cvtColor(frameclone, image, CV_BGR2GRAY);

		// 渲染检测
		//ratio是获取原图和文本检测图在长和宽上的缩放比例
		Point2f ratio((float)frame.cols / inpWidth, (float)frame.rows / inpHeight);
		for (size_t i = 0; i < indices.size(); ++i)
		{
			RotatedRect& box = boxes[indices[i]];

			Point2f vertices[4];
			box.points(vertices);
			for (int j = 0; j < 4; ++j)
			{
				vertices[j].x *= ratio.x;
				vertices[j].y *= ratio.y;
			}
			//在原图上绘制检测框,
			//检测框编号的顺序： 1 ------ 2
			//                   |        |
			//                   0 ------ 3
			for (int j = 0; j < 4; ++j)
				line(frame, vertices[j], vertices[(j + 1) % 4], Scalar(0, 255, 0), 1);

	///////////////对检测框中的文字用Tesseract-ocr 进行识别

		   // 识别文本
			int Width = (vertices[2].x - vertices[1].x)*1.1;
			int Height = (vertices[0].y - vertices[1].y)*1.1;
			myOCR->TesseractRect(image.data, 1, image.step1(), vertices[1].x, vertices[1].y, Width, Height);
			text = myOCR->GetUTF8Text();
		
			wchar_t *tempchar;
			const char * resulttemp;
			tempchar = Utf_8ToUnicode(text);
			resulttemp = UnicodeToAnsi(tempchar);
			


			/*string ss;
			ss = UTF8ToGB(text);

			printf("text: \n");
			printf(ss.c_str());
			printf("\n");
*/

			
		    //putText 只能在图片显示英文
			//putText(frame, ss, Point(vertices[1].x, vertices[1].y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
		
		 //在图片显示中文
			putTextZH(frame, resulttemp, Point(vertices[1].x-10, vertices[1].y-10), Scalar(0, 255, 0), 10, "微软雅黑", true, true);
	
			delete[] tempchar;
			delete[] resulttemp;
		}

		// 显示帧率.
		std::vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		/*std::string label = format("Inference time: %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));*/

		Mat frameclone;
		resize(frame, frameclone, Size(frame.cols * 2, frame.rows * 2));
		imshow(kWinName, frameclone);
	}

	delete[] text;
	
	myOCR->Clear();
	myOCR->End();
	return 0;
}


////预测框及其置信度
void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
	std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
	detections.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < scoreThresh)
				continue;

			// 解码预测.
			// 多个4，因为要素图比输入图像少4倍.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
				offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			detections.push_back(r);
			confidences.push_back(score);
		}
	}
}


string UTF8ToGB(const char* str)
{
	string result;
	WCHAR *strSrc;
	//TCHAR *szRes;
	LPSTR szRes;

	//获得临时变量的大小
	int i = MultiByteToWideChar(CP_UTF8, 0, str, -1, NULL, 0);
	strSrc = new WCHAR[i + 1];
	MultiByteToWideChar(CP_UTF8, 0, str, -1, strSrc, i);

	//获得临时变量的大小
	i = WideCharToMultiByte(CP_ACP, 0, strSrc, -1, NULL, 0, NULL, NULL);
	szRes = new CHAR[i + 1];
	WideCharToMultiByte(CP_ACP, 0, strSrc, -1, szRes, i, NULL, NULL);

	result = szRes;
	delete[]strSrc;
	delete[]szRes;

	return result;
}

//utf-8转unicode
wchar_t * Utf_8ToUnicode(const char* szU8)
{
	//UTF8 to Unicode
	//由于中文直接复制过来会成乱码，编译器有时会报错，故采用16进制形式

	//预转换，得到所需空间的大小
	int wcsLen = ::MultiByteToWideChar(CP_UTF8, NULL, szU8, strlen(szU8), NULL, 0);
	//分配空间要给'\0'留个空间，MultiByteToWideChar不会给'\0'空间
	wchar_t* wszString = new wchar_t[wcsLen + 1];
	//转换
	::MultiByteToWideChar(CP_UTF8, NULL, szU8, strlen(szU8), wszString, wcsLen);
	//最后加上'\0'
	wszString[wcsLen] = '\0';
	return wszString;
}

//将宽字节wchar_t*转化为单字节char*  
char* UnicodeToAnsi(const wchar_t* szStr)
{
	int nLen = WideCharToMultiByte(CP_ACP, 0, szStr, -1, NULL, 0, NULL, NULL);
	if (nLen == 0)
	{
		return NULL;
	}
	char* pResult = new char[nLen];

	WideCharToMultiByte(CP_ACP, 0, szStr, -1, pResult, nLen, NULL, NULL);

	return pResult;

}




//int main(int argc, char* argv[]) {
//
//	// 初始化 tesseract OCR 
//	tesseract::TessBaseAPI *myOCR =
//		new tesseract::TessBaseAPI();
//
//	printf("Tesseract-ocr version: %s\n",
//		myOCR->Version());  //输出Tesseract-ocr 版本
//	// printf("Leptonica version: %s\n",
//	//        getLeptonicaVersion());
//
//	const char* datapath = "D:\\opencv实践\\tesseract_4.0\\tessdata";  //设置字库模型的调用地址
//	if (myOCR->Init(datapath, "eng")) {
//		fprintf(stderr, "Could not initialize tesseract.\n");
//		exit(1);
//	}
//
//	tesseract::PageSegMode pagesegmode = static_cast<tesseract::PageSegMode>(7); // treat the image as a single text line
//	myOCR->SetPageSegMode(pagesegmode);
//
//	// read iamge
//	namedWindow("tesseract-opencv", 0);
//	Mat image = imread("testCarID.jpg", 0);
//
//	// set region of interest (ROI), i.e. regions that contain text
//	Rect text1ROI(80, 50, 800, 110);
//	Rect text2ROI(190, 200, 550, 50);
//
//	// recognize text
//	myOCR->TesseractRect(image.data, 1, image.step1(), text1ROI.x, text1ROI.y, text1ROI.width, text1ROI.height);
//	const char *text1 = myOCR->GetUTF8Text();
//
//	myOCR->TesseractRect(image.data, 1, image.step1(), text2ROI.x, text2ROI.y, text2ROI.width, text2ROI.height);
//	const char *text2 = myOCR->GetUTF8Text();
//
//	// remove "newline"
//	string t1(text1);
//	t1.erase(std::remove(t1.begin(), t1.end(), '\n'), t1.end());
//
//	string t2(text2);
//	t2.erase(std::remove(t2.begin(), t2.end(), '\n'), t2.end());
//
//	// print found text
//	printf("found text1: \n");
//	printf(t1.c_str());
//	printf("\n");
//
//	printf("found text2: \n");
//	printf(t2.c_str());
//	printf("\n");
//
//	// draw text on original image
//	Mat scratch = imread("sample.png");
//
//	int fontFace = FONT_HERSHEY_PLAIN;
//	double fontScale = 2;
//	int thickness = 2;
//	putText(scratch, t1, Point(text1ROI.x, text1ROI.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
//	putText(scratch, t2, Point(text2ROI.x, text2ROI.y), fontFace, fontScale, Scalar(0, 255, 0), thickness, 8);
//
//	rectangle(scratch, text1ROI, Scalar(0, 0, 255), 2, 8, 0);
//	rectangle(scratch, text2ROI, Scalar(0, 0, 255), 2, 8, 0);
//
//	imshow("tesseract-opencv", scratch);
//	waitKey(0);
//
//	delete[] text1;
//	delete[] text2;
//
//	// destroy tesseract OCR engine
//	myOCR->Clear();
//	myOCR->End();
//
//	return 0;
//}