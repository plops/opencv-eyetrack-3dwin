#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

Mat
computeMerrit(const Mat &gx,const Mat &gy,const Mat &g,const Mat &Im, double threshold)
{
  int h=gx.rows, w=gx.cols;
  Mat res(h,w,CV_32FC1,Scalar(0));
  for(int jj=0;jj<h;jj++){
    float*R=res.ptr<float>(jj);
    for(int ii=0;ii<w;ii++){
      int count =0;
      for(int j=0;j<h;j++){
	const float*Gx=gx.ptr<float>(j);
	const float*G=g.ptr<float>(j);
	const uchar*I=g.ptr<uchar>(j);
	int j0=jj-j;
	float doty=j0*gy.at<float>(j,0);
	for(int i=0;i<w;i++){
	  if(G[i]>threshold){
	    int i0=ii-i;
	    float
	      dotx=i0*Gx[i],
	      dot = (dotx+doty)/(sqrt(i0*i0+j0*j0)*G[i]);
	    if(dot>.1)
	      {
	      count++;
	      R[ii]+=(1+((255-I[i])/256.0))*dot*dot;
	    }
	  }
	}
	if(count>0)
	  R[ii]/=count;
      }
    }
  }
  return res;
}

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
  namedWindow("lmerrit",CV_WINDOW_AUTOSIZE);
  namedWindow("right",CV_WINDOW_AUTOSIZE);

  moveWindow("bla",0,100);
  moveWindow("lmerrit",200,0);
  moveWindow("right",400,0);

  Ptr<FilterEngine> fx = createDerivFilter(CV_8UC1,CV_32FC1,1,0,1);
  Ptr<FilterEngine> fy = createDerivFilter(CV_8UC1,CV_32FC1,0,1,1);

  Mat frame;

  c >> frame;
  //GaussianBlur(frame,frame,Size(5,5),3,3);
  Mat g;
  cvtColor(frame,g,CV_BGR2GRAY);
  std::vector<Rect>faces;
  face.detectMultiScale(g,faces,1.2,12,
			0
			|CV_HAAR_FIND_BIGGEST_OBJECT
			|CV_HAAR_DO_ROUGH_SEARCH
			|CV_HAAR_SCALE_IMAGE,Size(30,30));
  
  
  while(1){
    c >> frame;
  
    Mat g;
    cvtColor(frame,g,CV_BGR2GRAY);
  
    if(faces.size()){
      Rect left;
      double eye_s=.5;
      Point
	rc=faces[0].tl()+Point(.7*faces[0].width,.4*faces[0].height),
	lc=faces[0].tl()+Point(.3*faces[0].width,.4*faces[0].height),
	s=Size(eye_s*.5*.35*faces[0].width,eye_s*.8*.5*.3*faces[0].height);
      //Mat left_eye;
      Rect roir(rc-s,rc+s),roil(lc-s,lc+s);;
      // Mat 
      // 	lfx(roil.size(),CV_32FC1),
      // 	lfy(roil.size(),CV_32FC1),
      // 	rfx(roir.size(),CV_32FC1),
      // 	rfy(roir.size(),CV_32FC1);

      Mat 
	lsel(g,roil),
	rsel(g,roir);
      Mat 
	lblur,
	rblur;
      double sig1=2.2,sig2=7;
      GaussianBlur(lsel,lsel,Size(3,3),sig1,sig1);
      GaussianBlur(rsel,rsel,Size(3,3),sig1,sig1);

      GaussianBlur(lsel,lblur,Size(7,7),sig2,sig2);
      GaussianBlur(rsel,rblur,Size(7,7),sig2,sig2);

      Mat lfx,lfy,rfx,rfy;
      lblur.convertTo(lfx,CV_32FC1);
      lfy=lfx.clone();
      rblur.convertTo(rfx,CV_32FC1);
      rfy=rfx.clone();
      
      fx->apply(lsel,lfx);
      fy->apply(lsel,lfy);
      fx->apply(rsel,rfx);
      fy->apply(rsel,rfy);
      
      Mat lg, rg;
      magnitude(lfx,lfy,lg);
      magnitude(rfx,rfy,rg);
      Mat lsm,lss;
      meanStdDev(lg,lsm,lss);
      Mat rsm,rss;
      meanStdDev(rg,rsm,rss);
      

      //float threshold_r=30,threshold_l=threshold_r;
      Mat 
	threshold_r=rsm+.1*rss,
	threshold_l=lsm+.1*lss;
      
      std::cout << threshold_r.at<double>(0) << " "
		<< threshold_l.at<double>(0) << std::endl;
      
      //imshow("left",.04*(lg-threshold_l));
      Mat lmer = computeMerrit(rfx,rfy,rg,rblur,threshold_r.at<double>(0));
      double ma,mi;
      Point Ma;
      minMaxLoc(lmer,&mi,&ma,0,&Ma);
      ;;      lmer.at<float>(Ma.y,Ma.x)=0;
      rg.at<float>(Ma.y,Ma.x)=100;
      double rat=.0;
      imshow("lmerrit",((lmer-mi)/(ma-mi)-rat)/(1-rat));
      imshow("left",255*rsel/norm(rsel,NORM_INF));
      
      imshow("right",rg/norm(rg,NORM_INF));

      //rectangle(frame,Rect(lc-s,lc+s),Scalar(255)); 
      rectangle(g,roil,Scalar(255)); 
      circle(g,rc-s+Ma,4,Scalar(255));
      rectangle(g,faces[0],Scalar(255),3); 
      rectangle(g,roir,Scalar(255)); 
      imshow("bla",g);
   } else
      imshow("bla",g);
    
    if(waitKey(10) >= 0) 
      ;
  }
}

