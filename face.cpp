/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
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
void detectAndDisplay( Mat frame );
vector<Rect> facesByHand();

/** Global variables */
String cascade_name = "frontalface.xml";
CascadeClassifier cascade;


/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	// 3. Detect Faces and Display Result
	detectAndDisplay( frame );

	// 4. Save Result Image
	imwrite( "detected15.jpg", frame );

	return 0;
}

vector<Rect> facesByHand(){
	std::vector<Rect> groundFaces;
	groundFaces.push_back(Rect(65, 142, 55, 60));
	groundFaces.push_back(Rect(252, 168, 51, 60));
	groundFaces.push_back(Rect(384, 195, 54, 50));
	groundFaces.push_back(Rect(518, 177, 50, 61));
	groundFaces.push_back(Rect(649, 189, 52, 57));
	groundFaces.push_back(Rect(54, 253, 63, 65));
	groundFaces.push_back(Rect(195, 209, 53, 74));
	groundFaces.push_back(Rect(296, 235, 51, 75));
	groundFaces.push_back(Rect(428, 230, 58, 73));
  groundFaces.push_back(Rect(562, 238, 52, 76));
	groundFaces.push_back(Rect(680, 239, 53, 70));

	return groundFaces;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.1, 1, 0|CV_HAAR_SCALE_IMAGE, Size(50, 50), Size(500,500) );

       // 3. Print number of Faces found
	std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}

	vector<Rect> groundFaces = facesByHand();
	for (int i = 0; i < groundFaces.size(); i++){
		rectangle(frame, Point(groundFaces[i].x, groundFaces[i].y), Point(groundFaces[i].x + groundFaces[i].width, groundFaces[i].y + groundFaces[i].height), Scalar( 255, 0, 0 ), 2);
	}


}
