#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc,char**argv)
{
  
  VideoCapture c(0);
  if(!c.isOpened())
    return -1;
  std::cout << "frame: " <<
    c.get(CV_CAP_PROP_FRAME_WIDTH) << " " <<
    c.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
  /*  c.set(CV_CAP_PROP_FRAME_WIDTH,352);
      c.set(CV_CAP_PROP_FRAME_HEIGHT,288);
  
      std::cout << "frame: " <<
      c.get(CV_CAP_PROP_FRAME_WIDTH) << " " <<
      c.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
  */
  CascadeClassifier face;
  face.load("haarcascade_frontalface_default.xml");
  namedWindow("bla",CV_WINDOW_AUTOSIZE);
  namedWindow("left",CV_WINDOW_AUTOSIZE);
  namedWindow("right",CV_WINDOW_AUTOSIZE);
  Mat image;
  while(1){
    c >> image;
    //GaussianBlur(image,image,Size(5,5),3,3);
    Mat g;
    cvtColor(image,g,CV_BGR2GRAY);
    double scale=2;
    Mat small( cvRound (g.rows/scale), cvRound(g.cols/scale), CV_8UC1 );
    resize(g,small,small.size(),0,0,INTER_NEAREST);
    //    equalizeHist(small,small);
    std::vector<Rect>faces;
    face.detectMultiScale(small,faces,1.2,12,
			  0
			  |CV_HAAR_FIND_BIGGEST_OBJECT
			  |CV_HAAR_DO_ROUGH_SEARCH
			  |CV_HAAR_SCALE_IMAGE,Size(30,30));
    if(faces.size()){
      //      rectangle(small,faces[0],Scalar(255));
      Rect left;
      Point
	rc=faces[0].tl()+Point(.7*faces[0].width,.4*faces[0].height),
	lc=faces[0].tl()+Point(.3*faces[0].width,.4*faces[0].height),
	s=Size(.3*.35*faces[0].width,.3*.3*faces[0].height);
      Mat left_eye(g,Rect(scale*(lc-s),scale*(lc+s))),
	right_eye(g,Rect(scale*(rc-s),scale*(rc+s)));
      
      imshow("left",left_eye);
      imshow("right",right_eye);
      
      rectangle(small,Rect(lc-s,lc+s),Scalar(255)); 
      rectangle(small,faces[0],Scalar(255),3); 
      rectangle(small,Rect(rc-s,rc+s),Scalar(255)); 
      imshow("bla",small);
   } else
      imshow("bla",small);
    
    if(waitKey(10) >= 0) 
      ;
  }
}

