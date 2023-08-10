#include <glog/logging.h>

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/lanedetect.hpp>

#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

/*****************************************************************************
** Namespaces
*****************************************************************************/
using namespace std;
using namespace cv;



#define FONT_FACE FONT_HERSHEY_COMPLEX
#define FONT_SCALE 0.5
#define COLOR  Scalar(0, 191, 255)
#define THICKNESS 1
#define LINE_THICKNESS 2


bool isLeftLine(const Point& startPoint){
    if(startPoint.x < 320) 
      return true;
    return false;
}

double calculateDistancePx(const Point& point1, const Point& point2) {
    int dx = point2.x - point1.x;
    int dy = point2.y - point1.y;
    return sqrt(dx * dx + dy * dy);
}

double calculateAngleWithBottomEdge(const Point& point1, const Point& point2) {
    int dx = point2.x - point1.x;
    int dy = point2.y - point1.y;
    double angle = atan2(dy, dx); // Tính góc trong radian
    angle *= 180 / CV_PI; // Chuyển đổi sang độ
    if(point1.x < point2.x) // tính góc bên trái
        return -angle; 
    else 
        return angle + 180; // tính góc bên phải
}

namespace vitis {
namespace ai {

struct LaneDetect{
    static std::unique_ptr<LaneDetect> create();
    LaneDetect();
    std::vector<vitis::ai::RoadLineResult> run(const cv::Mat &input_image);
    int getInputWidth();
    int getInputHeight();
    size_t get_input_batch();
private:
  std::unique_ptr<vitis::ai::RoadLine> lane_detect_;

  
  bool debug;
};
std::unique_ptr<LaneDetect> LaneDetect::create(){
  return std::unique_ptr<LaneDetect>(new LaneDetect());
}

int LaneDetect::getInputWidth() { return lane_detect_->getInputWidth(); }
int LaneDetect::getInputHeight() { return lane_detect_->getInputHeight(); }
size_t LaneDetect::get_input_batch() { return lane_detect_->get_input_batch(); }

LaneDetect::LaneDetect():lane_detect_{vitis::ai::RoadLine::create("vpgnet_pruned_0_99")}
{
  const char * val = std::getenv( "LANE_DETECTION_DEBUG" );
  if ( val == nullptr ) {
    debug = true;
  }
  else {
    cout << "[INFO] LANE_DETECTION_DEBUG" << endl;
    debug = true;
  }
}



std::vector<vitis::ai::RoadLineResult>
LaneDetect::run(const cv::Mat &input_image) {
  std::vector<vitis::ai::RoadLineResult> results;
  cv::Mat image;
  image = input_image;
  //string char_labels[2] = {"Active","Fatique"};
  static int frame_number = 0;
  frame_number++;
  
  if ( debug == true ) {
    cout << "Frame " << frame_number << endl;
  }
  
  float model_time = 0.0;
  float total_time = 0.0;
  //run lane detection
  auto t_start = chrono::high_resolution_clock::now();
  auto lane_detect_results = lane_detect_->run(image);
  auto t_model_stop = chrono::high_resolution_clock::now();
  model_time = (float) chrono::duration_cast<chrono::milliseconds>(t_model_stop - t_start).count();
  cout << "MODEL lane detect:" << model_time << " ms" << endl;
  cout << "NUM OF LANES: " << lane_detect_results.lines.size() << endl;
  
  //process
  std::vector<int> color1 = {0, 255, 0, 0, 100, 255};
  std::vector<int> color2 = {0, 0, 255, 0, 100, 255};
  std::vector<int> color3 = {0, 0, 0, 255, 100, 255};
  
  string curveDirection = "Curve Direction: Straight";
  //lệch tâm đường(đơn vị m)
  string offCenter = "Off Center:";
  
  // khoảng cách line trái và phải tới giũa
  double distanceLeftLine;
  double distanceRightLine;
  double angleLeftLine;
  double angleRightLine;
  
  // điểm bắt đầu và kết thúc của line trái và phải
  Point startPointLeft(1,1);
  Point startPointRight(640,480);
  Point endPointLeft;
  Point endPointRight;
  double lengthLeftLine = 0;
  double lengthRightLine = 0;
  
  // Đường kẻ chính giữa màn hình
   //int LINE_THICKNESS = 2; // Độ dày của đường thẳng
  int x = image.cols / 2; // Vị trí y của đường thẳng
  Point startPointCenter(x, image.rows/2); // Điểm bắt đầu của đường thẳng
  Point endPointCenter(x, image.rows); // Điểm kết thúc của đường thẳng
  line(image, startPointCenter, endPointCenter, Scalar(0, 191, 255), LINE_THICKNESS);
  
  for (auto& line : lane_detect_results.lines) {
    vector<Point> points_poly = line.points_cluster;
    // điểm đầu và điểm cuối của line
    Point startPoint = points_poly.front().y > points_poly.back().y ? points_poly.front():points_poly.back() ;
    Point endPoint = points_poly.front().y < points_poly.back().y ? points_poly.front():points_poly.back();
    if(points_poly.size() < 15) continue;
    if(isLeftLine(startPoint) && startPoint.x > endPoint.x) continue;
    if(!isLeftLine(startPoint) && startPoint.x < endPoint.x) continue;
    
    int type = line.type < 5 ? line.type : 5;
    if (type == 2 && points_poly[0].x < image.rows * 0.5)
      continue;
    polylines(image, points_poly, false,
                  Scalar(color1[type], color2[type], color3[type]), LINE_THICKNESS,
                  CV_AA, 0);
    
    if(isLeftLine(startPoint) && startPoint.x >startPointLeft.x){

      startPointLeft = startPoint ;
      endPointLeft = endPoint;
      lengthLeftLine = calculateDistancePx(points_poly.front(),points_poly.back());
    }
    if(!isLeftLine(startPoint) && startPoint.x < startPointRight.x){
      startPointRight = startPoint ;
      endPointRight = endPoint;
      lengthRightLine = calculateDistancePx(points_poly.front(),points_poly.back());
    }
  }
  
  if(lengthLeftLine > 0)  {
    double slope = static_cast<double>(endPointLeft.y - startPointLeft.y) / (endPointLeft.x - startPointLeft.x);
    int x_intersection = startPointLeft.x + (image.rows - startPointLeft.y) / slope;
    Point intersectionPoint(x_intersection, image.rows);
    line(image, startPointLeft, intersectionPoint, Scalar(50,127, 205),2);
    distanceLeftLine = calculateDistancePx(intersectionPoint, Point(image.cols/2,image.rows));
    angleLeftLine = calculateAngleWithBottomEdge(startPointLeft, endPointLeft);
  }
  if(lengthRightLine > 0){
    double slope = static_cast<double>(endPointRight.y - startPointRight.y) / (endPointRight.x - startPointRight.x);
    int x_intersection = startPointRight.x + (image.rows - startPointRight.y) / slope;
    Point intersectionPoint(x_intersection, image.rows);
    line(image, startPointRight, intersectionPoint, Scalar(50,127, 205),2);
    distanceRightLine = calculateDistancePx(intersectionPoint, Point(image.cols/2,image.rows));
    angleRightLine = calculateAngleWithBottomEdge(startPointRight, endPointRight);
  }
  
  if(lengthRightLine> 0 && lengthLeftLine > 0){
    double totalDistance = distanceLeftLine + distanceRightLine;
    if(distanceLeftLine / totalDistance <= 0.4){
      offCenter = "Off Center: To the left";
    }else if(distanceLeftLine / totalDistance >= 0.6){
      offCenter = "Off Center: To the right";
    }else{
      offCenter = "Off Center: Center";
      
      // khi xe ở giữa làn đường thì xác định hướng đường đi 
      if(angleLeftLine - angleRightLine > 15){
      curveDirection = "Curve Direction: Left Curve";
      }else if(angleLeftLine - angleRightLine < -15){
        curveDirection = "Curve Direction: Right Curve";
      }else{
        curveDirection = "Curve Direction: Straight";
      }
    }
    
    
    // khi quá gần vạch kẻ đường bên trái hoặc phải
    if(distanceLeftLine / totalDistance <= 0.25){
      putText(image, "!!Near Left!", endPointLeft, FONT_FACE, FONT_SCALE, Scalar(58, 30, 196), 1.2);
    }else if(distanceLeftLine / totalDistance >= 0.75){
      putText(image, "!!Near Right!", endPointRight, FONT_FACE, FONT_SCALE, Scalar(58, 30, 196), 1.2);
    }
  
  }
  
  // hiện thị cảnh báo xe lệch tâm đường và hướng của đường đi
  putText(image, offCenter, Point(280, 30), FONT_FACE, FONT_SCALE, COLOR, THICKNESS);
  putText(image, curveDirection, Point(280, 50), FONT_FACE, FONT_SCALE, COLOR, THICKNESS);
  
  //tính thời gian và FPS xử lý trên từng ảnh
  auto t_stop = chrono::high_resolution_clock::now();
  total_time = (float) chrono::duration_cast<chrono::milliseconds>(t_stop - t_start).count();
  float FPS = 1000/total_time;
  cout << "Total:" << total_time << " ms" << endl;
  cout << "FPS:" << FPS << endl;
  
  results.emplace_back(lane_detect_results);
  return results;
}

}//ai
}//vitis









