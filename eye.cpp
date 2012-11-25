#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;


void
computeMerrit(const Mat &gx,const Mat &gy,const Mat &g,
	      const Mat &Im, Mat &res,double threshold)
{
  int h=gx.rows, w=gx.cols;
  //Mat res(h,w,CV_32FC1,Scalar(0));
  int o=2;
  for(int jj=o;jj<h-o;jj++){
    float*R=res.ptr<float>(jj);
    const uchar*I=Im.ptr<uchar>(jj);
    for(int ii=o;ii<w-o;ii++){
      int count =0;
      float wc=(255-I[ii])/256.0;
      for(int j=o;j<h-o;j++){
	const float*Gx=gx.ptr<float>(j);
      	const float*Gy=gy.ptr<float>(j);
	const float*G=g.ptr<float>(j);
	
	int j0=jj-j;
	
	for(int i=o;i<w-o;i++){
	  if(G[i]>threshold)  {
	    int i0=ii-i;
	    float
	      doty=j0*Gy[i],
	      dotx=i0*Gx[i],
	      dot = (dotx+doty)*1./G[i];
      	    if(dot<0){
	      count++;
	      R[ii]+=wc*dot*dot/(i0*i0+j0*j0);
	    }
	  }
	}
      }
      if(count>0)
	R[ii]/=count;
    }
  }
}

void
on_opengl(void*parm)
{
  Point**Ma=(Point**)parm;
  Point lMa=*(Ma[0]), rMa=*(Ma[1]);
}


int main(int argc,char**argv)
{
  
  VideoCapture c(0);
  if(!c.isOpened())
    return -1;
  std::cerr << "frame: " <<
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
  namedWindow("rmerrit",CV_WINDOW_AUTOSIZE);
  
  //namedWindow("gl",1);
  //Mat gl(Size(1350,700),CV_8UC1);
  //imshow("gl",gl);
  Point rMa,lMa;
  //Point*parm[2]={&lMa,&rMa};
  //setOpenGlDrawCallback("gl",on_opengl,(void*)&parm);

  int g_thr=30/4,g_thr_n=100;
  createTrackbar("gradient-threshold","bla",&g_thr,g_thr_n);
  int g_size=40/2,g_size_n=100;
  createTrackbar("eye-size","bla",&g_size,g_size_n);

  moveWindow("bla",0,100);
  moveWindow("lmerrit",200,0);
  moveWindow("rmerrit",600,0);
  moveWindow("right",400,0);

  Ptr<FilterEngine> fx = createDerivFilter(CV_8UC1,CV_32FC1,1,0,1);
  Ptr<FilterEngine> fy = createDerivFilter(CV_8UC1,CV_32FC1,0,1,1);
  
  Mat frame;
  
  //  Mat oldframe,ooldframe;
  while(1){
    // ooldframe=oldframe.clone();
    //oldframe=frame.clone();
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
    

  // Mat g;
    //cvtColor((frame+oldframe+ooldframe)/3,g,CV_BGR2GRAY);
  
    if(faces.size()){
      Rect left;
      double eye_s=g_size*2./g_size_n;
      float eye_x=.7;
      Point
	rc=faces[0].tl()+Point(eye_x*faces[0].width,.4*faces[0].height),
	lc=faces[0].tl()+Point((1-eye_x)*faces[0].width,.4*faces[0].height),
	s=Size(eye_s*.5*.35*faces[0].width,eye_s*.8*.5*.3*faces[0].height);
      //Mat left_eye;
      Rect roir(rc-s,rc+s),roil(lc-s,lc+s);;

      Mat 
	lsel(g,roil),
	rsel(g,roir);
      Mat 
	lblur,
	rblur;
      double sig1=1.2,sig2=3;
      GaussianBlur(lsel,lsel,Size(5,5),sig1,sig1);
      GaussianBlur(rsel,rsel,Size(5,5),sig1,sig1);

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
	threshold_r=rsm+g_thr*4./g_thr_n*rss,
	threshold_l=lsm+g_thr*4./g_thr_n*lss;
      
      //std::cout << g_thr << std::endl;

      //std::cout << threshold_r.at<double>(0) << " "
      //	<< threshold_l.at<double>(0) << std::endl;
      
      //imshow("left",.04*(lg-threshold_l));
      Mat rmer(rblur.size(),CV_32FC1,Scalar(0)),
	lmer(lblur.size(),CV_32FC1,Scalar(0));
      //std::cout << rblur.type() << std::endl;
      computeMerrit(rfx,rfy,rg,rblur,rmer,threshold_r.at<double>(0));
      computeMerrit(lfx,lfy,lg,lblur,lmer,threshold_l.at<double>(0));
      double rma,lma;
      // Point rMa,lMa;
      minMaxLoc(lmer,0,&lma,0,&lMa);
      minMaxLoc(rmer,0,&rma,0,&rMa);
      
      Point l= lc-s+lMa, r= rc-s+rMa, a = .5*(r+l);
      double d=norm(r-l);
      std::cout << "(" << (a.x-320)*1./320 << " " << (a.y-240)*1./240 << " " << d <<")"<< std::endl;
      
      //      lmer.at<float>(Ma.y,Ma.x)=0;
      //rg.at<float>(Ma.y,Ma.x)=100;
      {double rat=.9;
	imshow("lmerrit",(lmer/lma-rat)/(1-rat));}
      {double rat=.9;
	imshow("rmerrit",(rmer/rma-rat)/(1-rat));}

      imshow("left",4*(lg-threshold_l)/norm((lg-threshold_l),NORM_INF));
      imshow("right",4*(rg-threshold_r)/norm((rg-threshold_r),NORM_INF));

      //rectangle(frame,Rect(lc-s,lc+s),Scalar(255)); 

      circle(frame,rc-s+rMa,8,Scalar(20));
      circle(frame,lc-s+lMa,8,Scalar(20));
      line(frame,l,r,4);
      circle(frame,a,3,Scalar(255,255,255));
      line(frame,a-Point(.5*d,0),a+Point(.5*d,0),Scalar(255,255,255),3);
      rectangle(frame,faces[0],Scalar(0),3); 
      rectangle(frame,roil,Scalar(0)); 
      rectangle(frame,roir,Scalar(0)); 
      imshow("bla",frame);
   } else
      imshow("bla",g);
    
    if(waitKey(5) >= 0) 
      ;
  }
}

