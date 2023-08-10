#include <glog/logging.h>

#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo.hpp>
#include <vitis/ai/nnpp/yolov3.hpp>
#include <vitis/ai/yolov3.hpp>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>

using namespace std;
namespace vitis {
namespace ai {

struct StatusDetection{
    static std::unique_ptr<StatusDetection> create();
    StatusDetection();
    std::vector<vitis::ai::YOLOv3Result> run(const cv::Mat &input_image);
    int getInputWidth();
    int getInputHeight();
    size_t get_input_batch();
    static string model;
private:
  std::unique_ptr<vitis::ai::YOLOv3> status_detect_;
  
  bool debug;
};
std::unique_ptr<StatusDetection> StatusDetection::create(){
  return std::unique_ptr<StatusDetection>(new StatusDetection());
}

int StatusDetection::getInputWidth() { return status_detect_->getInputWidth(); }
int StatusDetection::getInputHeight() { return status_detect_->getInputHeight(); }
size_t StatusDetection::get_input_batch() { return status_detect_->get_input_batch(); }

StatusDetection::StatusDetection():status_detect_{vitis::ai::YOLOv3::create(model)}
//StatusDetection::StatusDetection():status_detect_{vitis::ai::YOLOv3::create("yolov3_tiny")}
//StatusDetection::StatusDetection():status_detect_{vitis::ai::YOLOv3::create("yolov3_tiny_3l")}
{
  const char * val = std::getenv( "STATUS_DETECTION_DEBUG" );
  if ( val == nullptr ) {
    debug = true;
  }
  else {
    cout << "[INFO] STATUS_DETECTION_DEBUG" << endl;
    debug = true;
  }
}

std::vector<vitis::ai::YOLOv3Result>
StatusDetection::run(const cv::Mat &input_image) {
  std::vector<vitis::ai::YOLOv3Result> results;
  cv::Mat image;
  image = input_image;
  string char_labels[2] = {"Active","Fatique"};
  static int frame_number = 0;
  frame_number++;
  
  if ( debug == true ) {
    cout << "Frame " << frame_number << endl;
  }
  
  float model_time = 0.0;
  float total_time = 0.0;
  //run lane detection
  auto t_start = chrono::high_resolution_clock::now();
  auto status_detect_results = status_detect_->run(image);
  auto t_model_stop = chrono::high_resolution_clock::now();
  model_time = (float) chrono::duration_cast<chrono::milliseconds>(t_model_stop - t_start).count();
  cout << "MODEL :" << model_time << " ms" << endl;
  cout << "NUM OF OBJECTS: " << status_detect_results.bboxes.size() << endl;
  
  //process
  
  
  for (const auto bbox : status_detect_results.bboxes) {
    int label = bbox.label;
    std::string status = (label == 1) ? "fatique":"active";
    cv::Scalar color;
    
    if(label == 0){
      color = cv::Scalar(0, 255, 0);
    }else {
      color = cv::Scalar(0, 0, 255);
    }

    
    float xmin = bbox.x * image.cols + 1;
    float ymin = bbox.y * image.rows + 1;
    float xmax = xmin + bbox.width * image.cols;
    float ymax = ymin + bbox.height * image.rows;
    float confidence = bbox.score;
    if (xmax > image.cols) xmax = image.cols;
    if (ymax > image.rows) ymax = image.rows;
    LOG_IF(INFO, true) << "RESULT: " << label << "\t" << xmin << "\t" << ymin
                          << "\t" << xmax << "\t" << ymax << "\t" << confidence
                          << "\n";
    int thickness;
    int fontsize;
    if(bbox.width* image.cols > 600){
      thickness = 4;
      fontsize = 4;
    }else {
      thickness = 2;
      fontsize = 1;
    }
    
    if(confidence > 0.5){
      cv::rectangle(image, cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                  color, thickness, 1, 0);
      cv::putText(image, status, cv::Point(xmin + 5, ymax - 5), cv::FONT_HERSHEY_COMPLEX, fontsize, color, thickness);
    }
    
  }
  
  auto t_stop = chrono::high_resolution_clock::now();
  total_time = (float) chrono::duration_cast<chrono::milliseconds>(t_stop - t_start).count();
  float FPS = 1000/total_time;
  cout << "Total:" << total_time << " ms" << endl;
  cout << "FPS:" << FPS << endl;
  
  results.emplace_back(status_detect_results);
  return results;
}

}//ai
}//vitis









