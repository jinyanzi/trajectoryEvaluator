#include "detectionDebugger.hpp"
#include <fstream>

using namespace std;
using namespace cv;

void DetectionDebugger::drawDetections(cv::Mat& img, int frame_num, const cv::Scalar& color, int thickness, double fx, double fy)	const {
	Scalar c = color;
	std::map<int, DetVec>::const_iterator det_it = detections.find(frame_num);
	if(det_it != detections.end()){
		for(const auto &det: det_it->second) {
			if(det.obj_id > 0)
				c = Scalar(128, 0, 128);
			
			stringstream ss;
			ss << det.cls << " " << det.score;
			Point p = det.box.tl()*fx;
			p.y -= 2*fy;
			p.x += 5*fy;

			string label = ss.str();
			int fontface = FONT_HERSHEY_SIMPLEX;
			double fontscale = fx >= 1 ? 0.4: 1.0;
			int offset = std::min(5.0, fx*5.0);
			double alpha = 0.4;

			cv::Size text = cv::getTextSize(label, fontface, fontscale, thickness, &offset);
			cv::Rect text_box = cv::Rect(p + cv::Point(0, offset), p + cv::Point(text.width, -text.height)) & cv::Rect(0, 0, img.cols, img.rows);
			cv::Mat roi = img(text_box);
			cv::Mat color_roi(roi.size(), CV_8UC3, c); 

			addWeighted(cv::Scalar(255, 0, 0), alpha, roi, 1.0 - alpha , 0.0, roi); 
			rectangle(img, scale_rect(det.box, fx), c, thickness*fx);
			putText(img, label, p, fontface, fontscale, cv::Scalar::all(255));
		}
	}
}

void DetectionDebugger::printDetectionSummary()	const {
	cout << getDetectionSum() << " detections in total\nMaximal area: " 
		 << getMinDetectionArea() << " minimal area: " << getMinDetectionArea()  << endl;
}


int DetectionDebugger::getDetectionSum()	const {
	int sum = 0;
	for(const auto &dets : detections) 
		sum += dets.second.size();

	return sum;
}

int DetectionDebugger::getMaxDetectionArea()	const {
	int max_r = INT_MIN;
	for(const auto &dets : detections) {
		for(const auto &det: dets.second){
			if(det.box.area() > max_r)
				max_r = det.box.area();
		}
	}
	return max_r;
}

int DetectionDebugger::getMinDetectionArea()	const {
	int min_r = INT_MAX;
	for(const auto &dets : detections) {
		for(const auto &det: dets.second){
			if(det.box.area() < min_r)
				min_r = det.box.area();
		}
	}

	return min_r;
}


bool DetectionDebugger::readDetectionFile(const std::string& filename, int to_frame_num) {
	ifstream infile;
	infile.open(filename.c_str());
	if(!infile.is_open()){
		cerr << "Failed to open file " << filename << endl;
		return false;
	}

	// read detection file
	// the format is frame_num, x, y, width, height, score
	std::string line;
	int last_frame_num = -1;
	int tmp_frame_num;
	Rect tmp_box;
	float tmp_score;
	string tmp_cls;
	DetVec tmp_detections;

	while(!infile.eof()) {
		std::getline(infile, line);
		readLine(line, tmp_frame_num, tmp_box, tmp_score, tmp_cls);
		if(last_frame_num < 0)
			last_frame_num = tmp_frame_num;

		// new frame
		if(tmp_frame_num != last_frame_num) {
			if(!tmp_detections.empty()) {
				detections.insert(std::make_pair(last_frame_num, tmp_detections));
				tmp_detections.clear();
			}
		}

		if(tmp_frame_num <= to_frame_num || to_frame_num == -1)
			tmp_detections.push_back(DetUnit(tmp_box, tmp_score, tmp_cls));
		
		last_frame_num = tmp_frame_num;
	}

	if(infile.is_open())
		infile.close();

	return true;
}

void DetectionDebugger::readLine(std::string& line, int& frame_num, cv::Rect& r, float& score, string& cls)	const {
	char * tokens = NULL;
	const char* delims = " ";
	int x = 0, y = 0, w = 0, h = 0;

	tokens = strtok(strdup(line.c_str()), delims);
	int j = 0;
	while(tokens != NULL) {
		switch(j) {
			case 0:	frame_num = atoi(tokens);	break;
			case 1: x = atoi(tokens);	break;
			case 2: y = atoi(tokens);	break;
			case 3: w = atoi(tokens);	break;
			case 4: h = atoi(tokens);	break;
			case 5: score = atof(tokens);	break;
			case 6: cls = tokens;	break;
			default: break;
		}

		tokens = strtok(NULL, delims);
		j++;
	}
	r = Rect(x, y, w, h);
}
