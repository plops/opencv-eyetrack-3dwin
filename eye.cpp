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


  Ptr<FilterEngine> fx = createDerivFilter(CV_8UC1,CV_32FC1,1,0,5);
  Ptr<FilterEngine> fy = createDerivFilter(CV_8UC1,CV_32FC1,0,1,5);

  Mat frame;
  while(1){
    c >> frame;
    //GaussianBlur(image,image,Size(5,5),3,3);
    Mat g;
    cvtColor(frame,g,CV_BGR2GRAY);
    std::vector<Rect>faces;
    face.detectMultiScale(frame,faces,1.2,12,
			  0
			  |CV_HAAR_FIND_BIGGEST_OBJECT
			  |CV_HAAR_DO_ROUGH_SEARCH
			  |CV_HAAR_SCALE_IMAGE,Size(30,30));
    if(faces.size()){
      Rect left;
      Point
	rc=faces[0].tl()+Point(.7*faces[0].width,.4*faces[0].height),
	lc=faces[0].tl()+Point(.3*faces[0].width,.4*faces[0].height),
	s=Size(.3*.35*faces[0].width,.3*.3*faces[0].height);
      //Mat left_eye;
      Rect roi(lc-s,lc+s);
      Mat 
	lfx(roi.size(),CV_32FC1),
	lfy(roi.size(),CV_32FC1),
	rfx(roi.size(),CV_32FC1),
	rfy(roi.size(),CV_32FC1);
      
      fx->apply(g,lfx,roi);
      fy->apply(g,lfy,roi);
      fx->apply(g,rfx,roi);
      fy->apply(g,rfy,roi);
      Mat
	lg,
	rg;
      magnitude(lfx,lfy,lg);
      magnitude(rfx,rfy,rg);
      imshow("left",.01*lg);
      imshow("right",.01*rg);
      
      rectangle(frame,Rect(lc-s,lc+s),Scalar(255)); 
      rectangle(frame,faces[0],Scalar(255),3); 
      rectangle(frame,Rect(rc-s,rc+s),Scalar(255)); 
      imshow("bla",frame);
   } else
      imshow("bla",g);
    
    if(waitKey(10) >= 0) 
      ;
  }
}

