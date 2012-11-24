#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

int main(int argc,char**argv)
{
  
  VideoCapture c(0);
  std::cout << "frame: " <<
    c.get(CV_CAP_PROP_FRAME_WIDTH) << " " <<
    c.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
  c.set(CV_CAP_PROP_FRAME_WIDTH,352);
  c.set(CV_CAP_PROP_FRAME_HEIGHT,288);
  
  std::cout << "frame: " <<
    c.get(CV_CAP_PROP_FRAME_WIDTH) << " " <<
    c.get(CV_CAP_PROP_FRAME_HEIGHT) << std::endl;
  
  CascadeClassifier face;
  face.load("haarcascade_frontalface_default.xml");
  namedWindow("bla",0); //CV_WINDOW_AUTOSIZE);
    
  while(1){
    Mat image;
    c >> image;
    //  GaussianBlur(image,image,Size(9,9),2,2);
    resize(image,image,Size(),.25,.25,INTER_NEAREST);
    Mat g;
    cvtColor(image,g,CV_BGR2GRAY);
    std::vector<Rect>faces;
    if(0)face.detectMultiScale(g,faces,1.1,2,
			  0|CV_HAAR_SCALE_IMAGE,Size(30,30));
    if(0)for(int i=0;i<faces.size();i++){
      float w=faces[i].width*.5,
	h= faces[i].height*.5;
      Point c(faces[i].x+w,faces[i].y+h);
      ellipse(g,c,Size(w,h),0,0,360,Scalar(255,0,255),
	      4,8,0);
    }
    imshow("bla",g);
  }
 
  
  waitKey(0);
}

