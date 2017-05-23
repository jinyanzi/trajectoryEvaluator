#ifndef TRAJ_DEBUGGER_H
#define TRAJ_DEBUGGER_H

#define RED cv::Scalar(0, 0, 255)
#define GREEN cv::Scalar(0, 255, 0)
#define BLUE cv::Scalar(255, 0, 0)
#define YELLOW cv::Scalar(0, 255, 255)
#define PINK cv::Scalar(255, 0, 255)
#define CYAN cv::Scalar(255, 255, 0)

#define RESULT 0
#define GROUND_TRUTH 1
#define DEBUG 1

#define TOP_LEFT 0
#define BOTTOM_RIGHT 1
#define STATIONARY_DIST 8

#include <vector>
#include <map>
#include <string>
#include <opencv2/opencv.hpp>


struct TrajUnit{
	TrajUnit(int f_id = -1, const cv::Rect& r = cv::Rect(), int occluded = 0, const std::string& c = ""):frame_id(f_id), box(r), if_occluded(occluded), cls(c){}
	int frame_id;
	cv::Rect box;
	int if_occluded;
	std::string cls;
};

struct MatchStruct{
	MatchStruct(int id = -1, int n = 0, float overlap = 0, float dist = 0, float f_ratio = 0, float ratio = 0)
		:matched_id(id), overlap_frame_count(n), overlap_area_sum(overlap), center_dist_sum(dist), max_frame_overlap_ratio(f_ratio), overlap_ratio(ratio){}
	int matched_id;
	int overlap_frame_count;
	float overlap_area_sum;	// overlap area along time, in pixels
	float center_dist_sum;	// accumulated center distance along time, in pixels
	float max_frame_overlap_ratio; // maximal overlap ratio on one frame along trajectory
	float overlap_ratio;	// overall overlap ratio along time, sum_t(overlap_area)/sum_t(union_area)
};


typedef std::vector<TrajUnit> TrajVec;

class TrajDebugger{
protected:
	// draw trajectory and box
    // obj_id = -1, draw all the objects
    void drawTrajectory(cv::Mat& img, int result_or_gt, int to_frame_num, const cv::Scalar& color, int thickness, double fx = 1, double fy = 1, int location = TOP_LEFT, int obj_id = -1)	const;
	void drawTrajectoryMatch(cv::Mat& img, int to_frame_num, const cv::Scalar& matched_color, const cv::Scalar& unmatched_color, int thickness, double fx = 1, double fy = 1, int obj_id = -1)	const;

    
	// draw trajectory and box of a certain object
    // return the index of the object at to_frame_num
    int drawObject(cv::Mat& img, const std::map<int, TrajVec>::const_iterator it, int to_frame_num, const cv::Scalar& color, int thicknes = 1,  double fx = 1, double fy = 1, int location = TOP_LEFT)	const;

	int drawObjectMatch(cv::Mat& img, const std::map<int, TrajVec>::const_iterator it, int to_frame_num, const cv::Scalar& matched_color, const cv::Scalar& unmatched_color, int thicknes = 1,  double fx = 1, double fy = 1, int location = TOP_LEFT)	const;
    
	void drawNumBox(cv::Mat& img, const cv::Rect& r, const cv::Scalar& color, int num,  double fx = 1, double fy = 1, int location = TOP_LEFT)	const;

	void readTrajLine(const std::string& line, int& obj_id, int& frame_id, cv::Rect& box, std::string& cls, int& occluded)	const;
	bool readTrajFile(int result_or_gt, const std::string& filename, int to_frame_num = -1);
    
	bool readSQL(int result_or_gt, const std::string& video_name, const std::string& db_name, int to_frame_num);
    
	// update object trajcetory with box
	void updateObject(int obj_id, int frame_id, const cv::Rect& box, std::map<int, TrajVec>& trajectories);

	MatchStruct trajVecMatch(const TrajVec& v1, const TrajVec& v2, int match_id = -1)	const;
	void areaSum(const std::map<int, TrajVec>& trajectory, std::map<int, float>& area_map)	const;
    
	// indicator function
	int indicator( bool f )	const{	return f ? 1:0;		}
	int indicator( int i, int thresh = 0)	const{	return i>thresh ? 1:0;	}
	int indicator( float f, float thresh = 0)	const{	return f > thresh? 1:0;	}

	int getObjectNum()	const {	return ground_truth.size();	}

	// find the start frame of a TrajVec for initialization
	// if 0 <= percent < 1, take the proportation of size
	int getInitStartIndex(const TrajVec& traj_vec, float percent)	const;
	int getTrajUnitSum()	const;
	
	cv::Point scale_point(const cv::Point& p, double rx, double ry)	const {	return cv::Point(p.x*rx, p.y*ry);	}
	cv::Rect scale_rect(const cv::Rect& box, double rx, double ry)	const {	return cv::Rect(scale_point(box.tl(), rx, ry), scale_point(box.br(), rx, ry));	}

    // variables
    std::map<int, TrajVec> result;
	std::map<int, TrajVec> ground_truth;

	std::map<int, MatchStruct> gt_res_match;
	std::map<int, MatchStruct> res_gt_match;
	
	std::map<int, float> gt_area_sum, res_area_sum;

public:

    // read functions
	// both result and ground truth are written in object id order
	// to_frame_num = -1 means read until end of file
	bool readResultFromFile(const std::string& filename, int to_frame_num = -1) {	return readTrajFile(RESULT, filename, to_frame_num);	}
	bool readGroundTruthFromFile(const std::string& filename, int to_frame_num = -1) {	return readTrajFile(GROUND_TRUTH, filename, to_frame_num);	}
	// bool readResultFromSQL(const std::string& filename, const std::string& db_name, int to_frame_num = -1) {	return readSQL(RESULT, filename, db_name, to_frame_num);	}
	// bool readGroundTruthFromSQL(const std::string& filename, const std::string& db_name, int to_frame_num = -1) {	return readSQL(GROUND_TRUTH, filename, db_name, to_frame_num);	}    
    
	// print function
    void printTrajectorySummary(int result_or_gt, int obj_id = -1)	const; 
	void printUnmatchedGroundTruth(const std::map<int, std::vector<MatchStruct> >& match_matrix = std::map<int, std::vector<MatchStruct> >())	const;
	void printMatchSummary()	const;
	// IO functions
	bool writeTrajectoryToFile(int result_or_gt, const std::string& traj_path)	const;
	bool writeGroundTruthTrajMatchStats(std::ofstream& out_file, const std::string& video_path)	const;
	bool writeGroundTruthTrajMatchToFile(const std::string& out_file)	const;
	bool writeInitBoxesToFile(const std::string& out_file, float init_percent)	const;

	void cleanGroundTruth();
	void groundTruthTrajMatch(float thresh = 0);
	void groundTruthTrajMatchByInit();
	
	std::string getFileNameFromPath(const std::string& file_path) const;
	std::string getFileDirPath(const std::string& file_path)	const;
	
	// return the intersection area size in pixel
	int overlappedArea(const cv::Rect &r1, const cv::Rect &r2)	const {	return (r1&r2).area();	};
	// return whether the two rectangles are overlapped
	int isOverlapped(const cv::Rect &r1, const cv::Rect &r2)	const	{	return overlappedArea(r1, r2)>0;	};
	// return the overlapped portion of two rectangles
	float overlapRatio(const cv::Rect& r1, const cv::Rect& r2)	const	{	return overlappedArea(r1, r2)/double(r1.area() + r2.area() - overlappedArea(r1, r2));	}
	float boxDist(const cv::Rect& r1, const cv::Rect& r2)	const {	return cv::norm(center(r1)-center(r2));	}
    cv::Point2f center(const cv::Rect& r) const{  return cv::Point2f(r.x+r.width/2, r.y+r.height/2); }
};


#endif
