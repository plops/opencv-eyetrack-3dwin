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
    GaussianBlur(frame,frame,Size(5,5),3,3);
    Mat g;
    cvtColor(frame,g,CV_BGR2GRAY);
    std::vector<Rect>faces;
    face.detectMultiScale(g,faces,1.2,12,
			  0
			  |CV_HAAR_FIND_BIGGEST_OBJECT
			  |CV_HAAR_DO_ROUGH_SEARCH
			  |CV_HAAR_SCALE_IMAGE,Size(30,30));
    if(faces.size()){
      Rect left;
      Point
	rc=faces[0].tl()+Point(.7*faces[0].width,.4*faces[0].height),
	lc=faces[0].tl()+Point(.3*faces[0].width,.4*faces[0].height),
	s=Size(.5*.35*faces[0].width,.5*.3*faces[0].height);
      //Mat left_eye;
      Rect roir(rc-s,rc+s),roil(lc-s,lc+s);;
      // Mat 
      // 	lfx(roil.size(),CV_32FC1),
      // 	lfy(roil.size(),CV_32FC1),
      // 	rfx(roir.size(),CV_32FC1),
      // 	rfy(roir.size(),CV_32FC1);

      Mat
	lfx_u(g,roil);

      Mat lfx;
      lfx_u.convertTo(lfx,CV_32FC1);
      
      
      fx->apply(lfx_u,lfx);
      //fy->apply(g,lfy,roil);
      //   fx->apply(g,rfx,roil,Point(lc-s));
      //fy->apply(g,rfy,roil,Point(lc-s));
      Mat lg, rg;
      //magnitude(lfx,lfy,lg);
      //magnitude(rfx,rfy,rg);
      imshow("left",lfx/255.);
      // imshow("right",.005*rg);
      
      //rectangle(frame,Rect(lc-s,lc+s),Scalar(255)); 
      rectangle(frame,roil,Scalar(255)); 
      rectangle(frame,faces[0],Scalar(255),3); 
      rectangle(frame,roir,Scalar(255)); 
      imshow("bla",frame);
   } else
      imshow("bla",g);
    
    if(waitKey(10) >= 0) 
      ;
  }
}

