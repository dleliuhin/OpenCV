/*
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
using namespace cv;

int main()
{

Mat img_gray;
Mat img = imread("circles1.jpg");

cvtColor(img, img_gray, CV_BGR2GRAY);

namedWindow("image", WINDOW_NORMAL);
imshow("image", img_gray);

GaussianBlur(img_gray, img_gray, Size(9, 9), 2, 2);
//namedWindow("image2", WINDOW_NORMAL);
//imshow("image2", img_gray);

vector<Vec3f> circles;
HoughCircles(img_gray, circles, CV_HOUGH_GRADIENT, 1, img_gray.rows / 8, 200, 100, 0, 0);

/// Draw the circles detected
for (size_t i = 0; i < circles.size(); i++)
{
Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
int radius = cvRound(circles[i][2]);
// circle center
circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
// circle outline
circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
}
/// Show your results
namedWindow("image3", WINDOW_NORMAL);
imshow("image3", img);

waitKey(0);
return 0;

CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY);
assert(capture);


cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, 640);
cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, 480);

double width = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH);
double height = cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT);

IplImage* frame = 0;

namedWindow("capture", CV_WINDOW_AUTOSIZE);

while (true)
{
frame = cvQueryFrame(capture);
cvShowImage("capture", frame);

char c = cvWaitKey(33);
if (c == 27)
{
break;
}
}
cvReleaseCapture(&capture);
cvDestroyWindow("capture");
return 0;
}
*/

// нахождение кругов на изображении, 
// с использованием трансформации Хафа cvHoughCircles()
/*
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <math.h>
#include <ctime>

int main()
{
	unsigned int start_time = clock();//Начальное время

	IplImage* image = 0;
	//char* imgName = argv[1];
	const char* imgName = "circles.jpg";
	// получаем картинку (в градациях серого)
	image = cvLoadImage(imgName, CV_LOAD_IMAGE_GRAYSCALE);

	assert(image != 0);

	// загрузим оригинальное изображении
	IplImage* src = cvLoadImage(imgName);

	// хранилище памяти для кругов
	CvMemStorage* storage = cvCreateMemStorage(0);
	// сглаживаем изображение
	cvSmooth(image, image, CV_GAUSSIAN, 5, 5);
	// поиск кругов
	CvSeq* results = cvHoughCircles(image, storage,	CV_HOUGH_GRADIENT, 10, image->width / 5);

	// пробегаемся по кругам и рисуем их на оригинальном изображении
	for (int i = 0; i < results->total; i++) {
		float* p = (float*)cvGetSeqElem(results, i);
		CvPoint pt = cvPoint(cvRound(p[0]), cvRound(p[1]));
		cvCircle(src, pt, cvRound(p[2]), CV_RGB(0xff, 0, 0));
	}
	
	unsigned int end_time = clock();//Конечное время
	unsigned int search_time = end_time - start_time; // искомое время
	printf("Processing time: %f", search_time + 0.005);
	// показываем
	cvNamedWindow("cvHoughCircles", 1);
	cvShowImage("cvHoughCircles", src);

	// ждём нажатия клавиши
	cvWaitKey(0);

	// освобождаем ресурсы
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&image);
	cvReleaseImage(&src);
	cvDestroyAllWindows();

	
	return 0;
}
*/

/*
//Выделение контура кругов на видео
//по распределению цветовой гаммы
//цвет объекта отличается от фона

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <ctype.h>

void FindBall(IplImage* Img);
void Counter(IplImage* img);

CvPoint2D32f center;
float radius;
long pointer = 0;
int main()
{
//Имеем входной поток изображения
	CvCapture* capture = 0;
	//Поток в виде видео
	capture = cvCaptureFromAVI("5.avi");
	cvNamedWindow("Demo", 1);
	//На каждом кадре видео определяем наличие объекта
	for (;;)
	{
		IplImage* frame = 0;
		int i, k, c;
		frame = cvQueryFrame(capture);
		if (!frame) break;
		//Выделяем круг окружностью
		FindBall(frame);
		cvShowImage("Demo", frame);
		c = cvWaitKey(50);
		if ((char)c==27) break;
		pointer++;
	}
	cvWaitKey(0);
	cvReleaseCapture(&capture);
	cvDestroyWindow("Demo");
	
	return 0;
}

void FindBall(IplImage* Img)
{
	IplImage* Image = cvCreateImage(cvGetSize(Img), 8, 3);
	cvCopy(Img, Image);
	uchar* ptr1;
	ptr1 = (uchar*)(Image->imageData);
	int i, j;
	//Объекту соответствуют оранжевые пиксели
	for (i = 0; i < Img->height; i++)
		for (j = 0; j < Img->width; j++)
		{
			uchar R = ptr1[j * 3 + 2 + i * Image->widthStep];
			uchar G = ptr1[j * 3 + 1 + i * Image->widthStep];
			uchar B = ptr1[j * 3 + i * Image->widthStep];
			int del = Image->widthStep;
			//Для выделения оранжевых пикселей
			//достаточно условия R > 1.5*G. R > 2*B
			//R G B - компонеты пикселя
			if (R==G && B==G)
			{
				ptr1[j*3+i*Image->widthStep] = 255;
				ptr1[j*3+1+i*Image->widthStep] = 255;
				ptr1[j*3+2+i*Image->widthStep] = 255;
			}
			else
			{
				ptr1[j*3+i*Image->widthStep] = 0;
				ptr1[j*3+1+i*Image->widthStep] = 0;
				ptr1[j*3+2+i*Image->widthStep] = 0;
			}
		}
	Counter(Image);
	if (center.x > -1)
	{
		CvPoint p;
		p.x = center.x;
		p.y = center.y;
		cvCircle(Img, p, radius, CV_RGB(255, 0, 0), 3, 8, 0);
	}
	cvReleaseImage(&Image);
}

void Counter(IplImage* img)
{
	//Из найденных контуров необходимо выделения 
	//максимального, который в данном примере будет 
	//соответствовать мячу
	IplImage* img_gray = cvCreateImage(cvSize(img->width, img->height), 8, 1);
	CvSeq* contours = 0;
	CvMemStorage* storage = cvCreateMemStorage(0);
	cvCvtColor(img, img_gray, CV_BGR2GRAY);
	cvFindContours(img_gray, storage, &contours, sizeof(CvContour),
		CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	CvSeq* h_next = 0;
	//Поиск максимального контура
	for (CvSeq* c = contours; c != NULL; c = c->h_next)
	{
		if (c != contours)
		{
			//Проверяем какой контур больше
			if (h_next->total >= c->total)
			{
				h_next->h_next = h_next->h_next->h_next;
				continue;
			}
		}
		h_next = c;
	}

	center.x = -1;
	if (h_next->total<200) return;//нет  мяча – слишком маленькие контуры
	cvDrawContours(img, h_next, CV_RGB(255, 0, 0), CV_RGB(0, 255, 0), 2, 2, CV_AA, cvPoint(0, 0));
	//Минимальная окружность
	//Функция ищет минимальную окружность, включающую 
	//в себя все точки последовательности. 
	//При невозможности найти окружность функция 
	//возвращает 0.
	cvMinEnclosingCircle(h_next, &center, &radius);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img_gray);
}
*/

//
// позволяет подобрать параметры
// Rmin, Rmax, Gmin, Gmax, Bmin, Bmax
// для выделения нужного объекта по цвету
//
// robocraft.ru
//

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdlib.h>
#include <stdio.h>

IplImage* image = 0;
IplImage* dst = 0;

// для хранения каналов RGB
IplImage* rgb = 0;
IplImage* r_plane = 0;
IplImage* g_plane = 0;
IplImage* b_plane = 0;
// для хранения каналов RGB после преобразования
IplImage* r_range = 0;
IplImage* g_range = 0;
IplImage* b_range = 0;
// для хранения суммарной картинки
IplImage* rgb_and = 0;

int Rmin = 0;
int Rmax = 256;

int Gmin = 0;
int Gmax = 256;

int Bmin = 0;
int Bmax = 256;

int RGBmax = 256;

//
// функции-обработчики ползунка
//
void myTrackbarRmin(int pos) {
	Rmin = pos;
	cvInRangeS(r_plane, cvScalar(Rmin), cvScalar(Rmax), r_range);
}

void myTrackbarRmax(int pos) {
	Rmax = pos;
	cvInRangeS(r_plane, cvScalar(Rmin), cvScalar(Rmax), r_range);
}

void myTrackbarGmin(int pos) {
	Gmin = pos;
	cvInRangeS(g_plane, cvScalar(Gmin), cvScalar(Gmax), g_range);
}

void myTrackbarGmax(int pos) {
	Gmax = pos;
	cvInRangeS(g_plane, cvScalar(Gmin), cvScalar(Gmax), g_range);
}

void myTrackbarBmin(int pos) {
	Bmin = pos;
	cvInRangeS(b_plane, cvScalar(Bmin), cvScalar(Bmax), b_range);
}

void myTrackbarBmax(int pos) {
	Bmax = pos;
	cvInRangeS(b_plane, cvScalar(Bmin), cvScalar(Bmax), b_range);
}

int main()
{
	const char* imgName = "circles.jpg";
	// получаем картинку (в градациях серого)
	image = cvLoadImage(imgName, 1);

	//
	// определяем минимальное и максимальное значение
	// у каналов HSV
	double framemin = 0;
	double framemax = 0;

	cvMinMaxLoc(r_plane, &framemin, &framemax);
	printf("[R] %f x %f\n", framemin, framemax);
	Rmin = framemin;
	Rmax = framemax;
	cvMinMaxLoc(g_plane, &framemin, &framemax);
	printf("[G] %f x %f\n", framemin, framemax);
	Gmin = framemin;
	Gmax = framemax;
	cvMinMaxLoc(b_plane, &framemin, &framemax);
	printf("[B] %f x %f\n", framemin, framemax);
	Bmin = framemin;
	Bmax = framemax;

	// окна для отображения картинки
	cvNamedWindow("original", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("R", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("G", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("B", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("R range", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("G range", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("B range", CV_WINDOW_AUTOSIZE);
	cvNamedWindow("rgb and", CV_WINDOW_AUTOSIZE);

	cvCreateTrackbar("Rmin", "R range", &Rmin, RGBmax, myTrackbarRmin);
	cvCreateTrackbar("Rmax", "R range", &Rmax, RGBmax, myTrackbarRmax);
	cvCreateTrackbar("Gmin", "G range", &Gmin, RGBmax, myTrackbarGmin);
	cvCreateTrackbar("Gmax", "G range", &Gmax, RGBmax, myTrackbarGmax);
	cvCreateTrackbar("Bmin", "B range", &Gmin, RGBmax, myTrackbarBmin);
	cvCreateTrackbar("Bmax", "B range", &Gmax, RGBmax, myTrackbarBmax);

	//
	// разместим окна по рабочему столу
	//
	if (image->width <1920 / 4 && image->height<1080 / 2) {
		cvMoveWindow("original", 0, 0);
		cvMoveWindow("R", image->width + 10, 0);
		cvMoveWindow("G", (image->width + 10) * 2, 0);
		cvMoveWindow("B", (image->width + 10) * 3, 0);
		cvMoveWindow("rgb and", 0, image->height + 30);
		cvMoveWindow("R range", image->width + 10, image->height + 30);
		cvMoveWindow("G range", (image->width + 10) * 2, image->height + 30);
		cvMoveWindow("B range", (image->width + 10) * 3, image->height + 30);
	}

	while (true) {

		// показываем картинку
		cvShowImage("original", image);
/*
		// показываем слои
		cvShowImage("R", r_plane);
		cvShowImage("G", g_plane);
		cvShowImage("B", b_plane);

		// показываем результат порогового преобразования
		cvShowImage("R range", r_range);
		cvShowImage("G range", g_range);
		cvShowImage("B range", b_range);

		// складываем 
		cvAnd(r_range, g_range, rgb_and);
		cvAnd(rgb_and, b_range, rgb_and);

		// показываем результат
		cvShowImage("rgb and", rgb_and);
*/
		char c = cvWaitKey(33);
		if (c == 27) { // если нажата ESC - выходим
			break;
		}
	}
	printf("\n[i] Results:\n");
	printf("[i][R] %d : %d\n", Rmin, Rmax);
	printf("[i][G] %d : %d\n", Gmin, Gmax);
	printf("[i][B] %d : %d\n", Bmin, Bmax);

	// освобождаем ресурсы
	cvReleaseImage(&image);
	cvReleaseImage(&rgb);
	cvReleaseImage(&r_plane);
	cvReleaseImage(&g_plane);
	cvReleaseImage(&b_plane);
	cvReleaseImage(&r_range);
	cvReleaseImage(&g_range);
	cvReleaseImage(&b_range);
	cvReleaseImage(&rgb_and);
	// удаляем окна
	cvDestroyAllWindows();
	return 0;
}