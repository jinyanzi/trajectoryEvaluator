#ifndef DETECTION_DEBUGGER_H
#define DETECTION_DEBUGGER_H

#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>

struct DetUnit {
	DetUnit(const cv::Rect& r, float s, std::string& cls):box(r),score(s), cls(cls), obj_id(-1){}
	cv::Rect box;
	float score;
	std::string cls;
	int obj_id;
};

typedef std::vector<DetUnit> DetVec;

class DetectionDebugger {
public:
	void drawDetections(cv::Mat& img, int frame_num, const cv::Scalar& color, int thickness = 1, double fx = 1, double fy = 1)	const;	
	void printDetectionSummary()	const;
	bool readDetectionFile(const std::string& filename, int to_frame_num = -1);
	void readLine(std::string& line, int& frame_num, cv::Rect& r, float& score, std::string& cls)	const;
	int getDetectionSum()	const;
	int getMaxDetectionArea()	const;
	int getMinDetectionArea()	const;
protected:
	// key frame number, value vector of detection boxes with scores
	std::map<int, DetVec> detections;
	cv::Rect scale_rect(const cv::Rect& box, double r)	const {	return cv::Rect(box.tl()*r, box.br()*r);	}
};

#endif
