// header inclusion
#include <stdio.h>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
//void detectAndDisplay( Mat frame );
//vector<Rect> facesByHand();
void sobel(Mat src, char* imageName);
void hough(Mat imgMag, Mat imgGrad, int minRad, int maxRad, char* imageName);

/** Global variables */
//String cascade_name = "dartcascade/cascade.xml";
//CascadeClassifier cascade;

/** @function main */
int main( int argc, char** argv )
{

	// LOADING THE IMAGE
	 char* imageName = argv[1];

	 for(int i = 0; i<=15; i++){
		 sprintf(imageName, "dart%d.jpg", i);

		 Mat image;
		 image = imread( imageName, 1 );
		 Mat gray_image, image_blured;
		 GaussianBlur(image, image_blured, Size(5,5), 0, 0, BORDER_DEFAULT);
		 cvtColor(image_blured, gray_image, CV_BGR2GRAY);

		 // if( argc != 2 || !image.data )
		 // {
			//  printf( " No image data \n " );
			//  return -1;
		 // }

		 sobel(gray_image, imageName);
	 }



	 // Mat magnitude, orientation;
   // magnitude = imread( "gradientMag.png", 1 );
   // orientation = imread( "gradientOrt.png", 1);
   // cvtColor(magnitude, magnitude, CV_BGR2GRAY);
   // cvtColor(orientation, orientation, CV_BGR2GRAY);

   //hough(magnitude, orientation, 10, 60);

	return 0;
}

void sobel(Mat src, char* imageName){

	Mat imageDX(Size(src.cols, src.rows), CV_32FC1, Scalar(0));
  Mat imageDY(Size(src.cols, src.rows), CV_32FC1, Scalar(0));
  Mat imageMagnitude(Size(src.cols, src.rows), CV_32FC1, Scalar(0));
  Mat imageOrientation(Size(src.cols, src.rows), CV_32FC1, Scalar(0));

	//initialise Kernels
	Mat kerGRX = (Mat_<float>(3,3) << 1, 0, -1,
                                    2, 0, -2,
                                    1, 0, -1);
  Mat kerGRY = (Mat_<float>(3,3) << 1, 2, 1,
                                    0, 0, 0,
                                    -1, -2, -1);
   Mat kernel = (Mat_<float>(3,3) << 1, 1, 1,
                                     1, 1, 1,
                                     1, 1, 1);

	 //compute x and y gradient images
	 for(int y = 1; y<src.rows-1; y++){
   	for(int x = 1; x<src.cols-1; x++){

     for(int j = 0; j <= 2; j++){
       for(int i = 0; i <= 2; i++){
         imageDX.at<float>(y,x) += (float)( ( (float)src.at<uchar>(y+(j-1), x+(i-1)) ) * (kerGRX.at<float>(j, i)) );
         imageDY.at<float>(y,x) += (float)( ( (float)src.at<uchar>(y+(j-1), x+(i-1)) ) * (kerGRY.at<float>(j, i)) );
       }
     }
		 //find image gradient magnitude and gradient orientation
     imageMagnitude.at<float>(y,x) = (float)sqrt( pow(imageDX.at<float>(y,x), 2) + pow(imageDY.at<float>(y,x), 2) );
     imageOrientation.at<float>(y,x) = (float)atan2( imageDY.at<float>(y,x), imageDX.at<float>(y,x) );

		 //set threshold for image magnitude
     if(imageMagnitude.at<float>(y,x) > 40)
     {
       imageMagnitude.at<float>(y,x) = 255;
     }
     else
     {
       imageMagnitude.at<float>(y,x) = 0;
     }

   	}
	}

	//normalize values to between 0 and 255
	normalize(imageDX, imageDX, 0, 255, CV_MINMAX);
  normalize(imageDY, imageDY, 0, 255, CV_MINMAX);
  normalize(imageMagnitude, imageMagnitude, 0, 255, CV_MINMAX);
  //normalize(imageOrientation, imageOrientation, 0, 255, CV_MINMAX);

	//convert images to uchars
	imageDX.convertTo(imageDX, CV_8UC1);
	imageDY.convertTo(imageDY, CV_8UC1);
	imageMagnitude.convertTo(imageMagnitude, CV_8UC1);
	//imageOrientation.convertTo(imageOrientation, CV_8UC1);

	//output images
	imwrite("gradX.png", imageDX);
	imwrite("gradY.png", imageDY);
	imwrite("gradientMag.png", imageMagnitude);
	imwrite("gradientOrt.png", imageOrientation);

	hough(imageMagnitude, imageOrientation, 50, 200, imageName);
}

int ***malloc3dArray(int dim1, int dim2, int dim3)
{
    int i, j, k;

    int ***array = (int ***) malloc(dim1 * sizeof(int **));

    for (i = 0; i < dim1; i++) {

		array[i] = (int **) malloc(dim2 * sizeof(int *));

		for (j = 0; j < dim2; j++) {

			array[i][j] = (int *) malloc(dim3 * sizeof(int));
		}

    }
    return array;

}

void hough(Mat imgMag, Mat imgGrad, int minRad, int maxRad, char* imageName)
{
  //initialise 3D array
  int ***ipppArr;
  int dim1 = imgMag.rows, dim2 = imgMag.cols, dim3 = maxRad;
  int i, j, k;

  ipppArr = malloc3dArray(dim1, dim2, dim3);

  for (i = 0; i < dim1; ++i)
    for (j = 0; j < dim2; ++j)
      for (k = minRad; k < dim3; ++k)
        ipppArr[i][j][k] = 0;



    for(int x = 0; x < imgMag.rows; x++){
      for(int y = 0; y < imgMag.cols; y++){

      if(imgMag.at<uchar>(x,y) == 255){

        for(int r = minRad; r<maxRad; r++){

            float angle = imgGrad.at<float>(x,y);
						//std::cout << angle << "("  << ")";

            for (int spread = -5; spread <= 5; spread++) {
							float spreadRad = spread * (CV_PI/180);
							float ang = angle + spreadRad;

              // int a = x + r*sin(ang* (CV_PI/180));
              // int b = y + r*cos(ang* (CV_PI/180));
							int a = x + r*sin(ang);
							int b = y + r*cos(ang);

              if( a >= 0 && b >= 0 && a < imgMag.rows && b < imgMag.cols){
                  ipppArr[a][b][r] += 1;
              }

              // int a1 = x - r*sin(ang* (CV_PI/180));
              // int b1 = y - r*cos(ang* (CV_PI/180));
							a = x - r*sin(ang);
							b = y - r*cos(ang);

              if(a >= 0 && b >= 0 && a < imgMag.rows && b < imgMag.cols){
                  ipppArr[a][b][r] += 1;
              }
            }
        }
      }
    }
  }

  //create houghSpace
  Mat houghSpace = Mat(imgMag.rows, imgMag.cols, CV_32SC1, float(0));

  cvtColor(imgMag, imgMag, CV_GRAY2BGR);
  //draw circles
  for (i = 0; i < dim1; ++i){
    for (j = 0; j < dim2; ++j){
      int houghSum = 0;
      for (k = minRad; k < dim3; ++k){

        houghSum += ipppArr[i][j][k];

        if(ipppArr[i][j][k] > 150){
          circle(imgMag, Point(j, i), k, Scalar(0, 0, 255), 2, 8, 0);
					circle(imgMag, Point(j, i), 1, Scalar(0, 255, 0), 2, 8, 0);
        }

      }
      houghSpace.at<int>(i,j) = houghSum;
    }
  }

  double min, max;
  cv::minMaxLoc(houghSpace, &min, &max);
  Mat newHough;
  houghSpace.convertTo(newHough, CV_8U, 255.0/(max-min), -255.0*min/(max-min));

  //imshow("Hough Space", newHough);
  //waitKey(0);

  //imshow("hough", imgMag);
  //waitKey(0);

	char finalImage[20];
	sprintf(finalImage, "hough%s", imageName);
  imwrite(finalImage, imgMag);
	sprintf(finalImage, "houghSpace%s", imageName);
	imwrite(finalImage, newHough);

}
