#ifndef EVALUATOR_H
#define EVALUATOR_H

#include <iostream>
#include <fstream>
#include "trajDebugger.h"
#include "detectionDebugger.hpp"

#define FIT 0
#define MATCH 1

#define OVERLAP_THRESH 0.0
#define SCORE_THRESH 0.001
#define MATCHED 1
#define MISSED 0
#define UNMATCHED -1


typedef std::pair<int, float> MatchPair;
struct MatchUnit{
    MatchUnit(std::vector<MatchPair>& s, int i = -1)
        :scores(s), id(i){}
    bool operator < (const MatchUnit& u)   const{
        if( !u.scores.empty() && !this->scores.empty() )
            return u.scores.front().second > this->scores.front().second;
        return u.scores.size() > this->scores.size();
    }
    friend std::ostream& operator << (std::ostream& out, const MatchUnit& u);
    std::vector<MatchPair> scores;  // sorted score with all objects in other set
    int id;     // id of the object, either ground_truth id or result id
};


struct CmpMatchPair{
    bool operator()(const MatchPair & p1, const MatchPair& p2)  const{
        return p1.second > p2.second;
    }
};

class Evaluator: public TrajDebugger, public DetectionDebugger{
private:
	bool detectionMatched;
	// score matrix
	// first dimension: ground_truth
	// second dimension: result
	float** fit_matrix;
	float** match_matrix;

    // bool matrix
    bool** fit_bool_matrix;
	bool** match_bool_matrix;

    // matched result
    // matching are performed in both direction
    // ground_truth to result and result to ground_truth
    std::map<int, MatchPair> ground_truth_match;
    std::map<int, MatchPair> result_match;

    static const int ground_truth_match_ref[];
    static const int result_match_ref[];
	
	// compute score for one object in tracked file and ground truth
	// fit computes overlapped portion in terms of size
	// match computes overlaped portion without size
	float compute_score(int gt_id, int result_id, int fit_or_match);
	// align two trajectories, start and end by the same frame_num
	// return true if two trajectories have overlapped frames
	bool align_trajectories(TrajVec &t1, TrajVec &t2);	

    // build heap, sorted by largest matching score
    // Using heap because we want to do greedy matching from the largest score
    void buildHeap( float** matrix, int from_set, std::vector<MatchUnit>& score_heap);

    float finalScore(const std::map<int, MatchPair >& match_result) const;
    // maximum bipartite matching
    // bool bipartitMatch();


	// detection related functions
	int getMatchedDetectionNum()	const;
	// return the number of trajectory that at least overlap with one detection
	int getDetectionTrajOverlapNum()	const;

    template<typename T>
    void freeMatrix(T** matrix, unsigned int rows);

public:
    Evaluator()
        :detectionMatched(false), fit_matrix(NULL), match_matrix(NULL), 
         fit_bool_matrix(NULL), match_bool_matrix(NULL){}
    ~Evaluator();
	// generate evaluate matrix
	// method = 1: fit
	// method = 2: match
	void evaluate(int method);

	void groundTruthDetectionMatch(float thresh = OVERLAP_THRESH);
    float finalScore(int from_set) const;

	// bipartite match between result and ground truth
	// void match();

    // match ground truth and result greedily
    // method indicates which matrix to use in matching
    void greedyMatch(int method, int from_set = -1);

    // if_bool = true, print binary matrix
	void printScoreMatrix(int method, int if_bool = false)	const;
    void printScoreDetail(int gt_id, int result_id);//   const;

    void printMatchDetail( int from_set )   const;

	void printDetectionStats()	const;

	// IO functions
	void writeDetectionStats(std::ofstream& outfile, const std::string& video_path)	const;
	bool writeGroundTruthDetetionMatchToFile(const std::string& outfile)	const;
	// visually show the iesground truth and result
    // obj_id != -1, showies only a certain object
    bool visualize(const std::string& video_name, int start_frame = 0, int obj_id = -1, int result_or_gt = GROUND_TRUTH)	const;
    // save to video file
    bool saveToVideo(const std::string& video_name, std::string& out_video_name, int out_w = 0, int out_h = 0, double resize_factor = 1, int to_frame = -1)	const;

};

#endif
