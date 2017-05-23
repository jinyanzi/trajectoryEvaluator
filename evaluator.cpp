#include "evaluator.h"
#include <algorithm>

using namespace std;
using namespace cv;

const int Evaluator::ground_truth_match_ref[] = {
    7, 5, 13, 1, 0, 2, 3, -1, 6, 8, 
    9, 10, 11, 13, 14, -1, 17, 18, 19, 16, 
    20, 21, 22, 23, 24, 25, 26, 27, 31, 29, 
    28, 35, 32, 33, 34, 36, -1, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    51, 50, -1, 52, 53, 55, 56, 57, -1, 60,
    58, 61, 62
};

const int Evaluator::result_match_ref[] = {
    4, 3, 5, 6, -1, 1, 8, 0, 9, 10,
    11, 12, -1, 13, 14, -1, 19, 16, 17, 18,
    20, 21, 22, 23, 24, 25, 26, 27, 30, 29,
    -1, 28, 32, 33, 34, -1, 35, 37, 38, 39, 
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 
    51, 50, 53, 54, 50, 55, 56, 57, 60, -1,
    59, 61, 62
};

std::ostream& operator << (std::ostream& out, const MatchUnit& u){
    if( u.scores.empty() )
        out << u.id << "\tempty scores" << std::endl;
    else
        out << u.id << "\t(" << u.scores.front().first << "," 
            << u.scores.front().second << ")" << std::endl;
    return out;
}

void Evaluator::evaluate(int method){
    float** matrix = (method == FIT) ? fit_matrix : match_matrix;
    bool** bool_matrix = (method == FIT) ? fit_bool_matrix : match_bool_matrix;

    unsigned int result_num = result.size();
    unsigned int gt_num = ground_truth.size();

    // allocate space
    if( matrix == NULL ){
        matrix = new float*[gt_num];
        for( unsigned int i = 0; i < gt_num; i++)
            matrix[i] = new float[result_num];
    }

    if( bool_matrix == NULL ){
        bool_matrix = new bool*[gt_num];
        for(unsigned int i = 0; i < gt_num; i++)
            bool_matrix[i] = new bool[result_num];
    }

    // write score to table
   for(unsigned int i = 0; i < gt_num; i++)
       for(unsigned int j = 0; j < result_num; j++){
            float score = compute_score(i, j, method);
            matrix[i][j] = score;
            bool_matrix[i][j] = indicator(score, SCORE_THRESH);
        }
   
   // pass back
   if(method == FIT){
       fit_matrix = matrix;
       fit_bool_matrix = bool_matrix;
   }else{
       match_matrix = matrix;
       match_bool_matrix = bool_matrix;
   }
}

Evaluator::~Evaluator(){
    // free matrix space
    freeMatrix(fit_matrix, ground_truth.size());
    freeMatrix(fit_bool_matrix, ground_truth.size());
    freeMatrix(match_matrix, ground_truth.size());
    freeMatrix(match_bool_matrix, ground_truth.size());
}

template<typename T>
void Evaluator::freeMatrix(T** matrix, unsigned int rows){
   if(matrix){
       for(unsigned int i = 0; i < rows; i++){
           delete matrix[i];
       }
       delete[] matrix;
   }
}

void Evaluator::buildHeap(float** matrix, int from_set, std::vector<MatchUnit>& score_heap){
    unsigned int from_set_num, to_set_num;
    from_set_num = (from_set == RESULT) ? result.size() : ground_truth.size();
    to_set_num = (from_set == RESULT) ? ground_truth.size() : result.size();
    score_heap.reserve(from_set_num);

    for( unsigned i = 0; i < from_set_num; i++){
        vector<MatchPair> scores(to_set_num);
        for(unsigned j = 0; j < to_set_num; j++){
            // in the matrix, the rows are ground_truth and columns are results
            // if we build heap for result, the scores vector should be vertical columns
            // if we build heap for ground_truth, the scores vector should be horizontal rows
            scores[j] = (from_set == RESULT) ? MatchPair(j, matrix[j][i]) : MatchPair(j, matrix[i][j]);
        }
        // sort scores by descending order
        std::sort(scores.begin(), scores.end(), CmpMatchPair());
        MatchUnit m(scores, i);
        score_heap.push_back(m);
    }
    // sort heap by max_score
    std::make_heap(score_heap.begin(), score_heap.end());
    std::sort_heap(score_heap.begin(), score_heap.end());

#if DUBUG 
    cout << "heap size " << score_heap.size() << endl;
    for( unsigned int i = 0; i < score_heap.size(); i++ ){
        cout << score_heap[i];
    }
#endif
}

float Evaluator::finalScore(const std::map<int, MatchPair >& match_result)  const{
    float sum_score = 0;
    for(std::map<int, MatchPair >::const_iterator it = match_result.begin();
            it != match_result.end(); it++){
        if( it->second.first != -1 )
            sum_score += it->second.second;
    }
    return 2*sum_score/(ground_truth.size() + result.size() );
}

float Evaluator::finalScore(int from_set)   const{
    if(from_set == RESULT)
        return finalScore(result_match);
    else
        return finalScore(ground_truth_match);
}

float Evaluator::compute_score( int gt_id, int result_id, int fit_or_match){
	std::map<int, TrajVec>::const_iterator it_ground_truth = ground_truth.find(gt_id);
    std::map<int, TrajVec>::const_iterator it_result = result.find(result_id);
	
    if( it_result == result.end() || it_ground_truth == ground_truth.end() )
		return 0.0;

	TrajVec traj_result = it_result->second, traj_ground_truth = it_ground_truth->second;
	// first align two trajectories, start and end by the same frame_num
	
	// if two trajectories have no overlapped frames, return 0.0 as score
	if(!align_trajectories(traj_ground_truth, traj_result))
		return 0.0;
#if DEBUG
	cout << "compute " << (fit_or_match == FIT? "FIT": "MATCH") << " score: result " << result_id 
         << " [" << traj_result.front().frame_id << "-" << traj_result.back().frame_id
		 << " | " << traj_result.back().frame_id - traj_result.front().frame_id + 1 << "]" 
         << " with ground truth " << gt_id 
		 << " [" << traj_ground_truth.front().frame_id << "-" << traj_ground_truth.back().frame_id
		 << " | " << traj_ground_truth.back().frame_id - traj_ground_truth.front().frame_id + 1 << "]" << endl;
#endif
	// fit: sum( overlap_area(ground_truth, result) )/sum( union_area(ground_truth, result) )
	if( fit_or_match == FIT ){
		long int overlapped_area_sum = 0;
		long int total_area_sum = 0, total_area_sum2 = 0;
		for( unsigned int i = 0; i < traj_ground_truth.size(); i++ ){
			int area = overlappedArea(traj_result[i].box, traj_ground_truth[i].box);
			overlapped_area_sum += area;
            if(area > 0)
                total_area_sum2 = total_area_sum2 + (traj_ground_truth[i].box.area() + traj_result[i].box.area() - area);
#if DEBUG 
			cout << i << "\ttotal_area_sum = " << total_area_sum << "\t+\t " << traj_ground_truth[i].box.area() << "\t+\t" 
                 << traj_result[i].box.area() << "\t-\t" << area;
#endif
            total_area_sum = total_area_sum + (traj_ground_truth[i].box.area() + traj_result[i].box.area() - area);
#if DEBUG 
            cout << "\t=\t" << total_area_sum << traj_result[i].box << "\t" << traj_ground_truth[i].box << endl;
#endif
		}
#if DEBUG 
		cout << overlapped_area_sum << "/" << total_area_sum << " = " << (float)overlapped_area_sum/total_area_sum << "\t"
             << overlapped_area_sum << "/" << total_area_sum2 << " = " << (float)overlapped_area_sum/total_area_sum2 << "\t"
             << "total area difference " << total_area_sum - total_area_sum2 << endl;
#endif
		return (float)overlapped_area_sum/total_area_sum;
	}

	// match: count( overlapRatio(ground_truth, result) > thresh )/count( area(ground_truth)>0 + area(result)>0 - overlapRatio(ground_truth, result)>thresh)
	if( fit_or_match == MATCH ){
		int overlap_count = 0;
		int total_count = 0;
		for( unsigned int i = 0; i < traj_result.size(); i++ ){
			int c = indicator(overlapRatio(traj_result[i].box, traj_ground_truth[i].box), OVERLAP_THRESH);
			overlap_count += c;
			total_count += (indicator(traj_ground_truth[i].box.area()) + indicator(traj_result[i].box.area()) - c);
		}
#if DEBUG > 1
		cout << overlap_count << "/" << total_count << " = " << (float)overlap_count/total_count << endl;
#endif
		return (float)overlap_count/total_count;
	}

	return 0.0;
}

bool Evaluator::align_trajectories(TrajVec &t1, TrajVec &t2){
	// return false if two trajectories have no overlapped frames
	if( t1.front().frame_id > t2.back().frame_id 
			|| t2.front().frame_id > t1.back().frame_id)
		return false;
	TrajVec* prev_t = NULL, *after_t = NULL;

	// check beginning
	if( t1.front().frame_id < t2.front().frame_id ){
		prev_t = &t1;
		after_t = &t2;
	}else{
		prev_t = &t2;
		after_t = &t1;
	}
	// insert empty TrajUnit into the beginning of traj_ground_truth
	int fid_front = after_t->front().frame_id;
	int start_fid = prev_t->front().frame_id;
	while(--fid_front >= start_fid)
		after_t->insert(after_t->begin(), TrajUnit(fid_front, Rect()));

	// check back
	if( t1.back().frame_id < t2.back().frame_id ){
		prev_t = &t1;
		after_t = &t2;
	}else{
		prev_t = &t2;
		after_t = &t1;
	}

	int fid_back = prev_t->back().frame_id;
	int end_fid = after_t->back().frame_id;
	while(++fid_back <= end_fid)
		prev_t->push_back(TrajUnit(fid_back, Rect()));

	return true;
}

void Evaluator::groundTruthDetectionMatch(float thresh)	{
	if(ground_truth.empty() || detections.empty())
		return;

	// loop through detections on each frame
	for(auto &dets: detections) {
		// loop through each detection on one frame
		for(auto &det: dets.second) {
			int match_obj_id = -1;
			float max_overlap = 0;
			// loop through each object trajectory
			for(const auto &obj_traj: ground_truth) {
				// loop through each trajectory of one object
				for(const auto &traj: obj_traj.second) {
					if(traj.frame_id == dets.first) {
						float o = overlapRatio(traj.box, det.box);
						if(o > max_overlap) {
							match_obj_id = obj_traj.first;
							max_overlap = o;
						}
						// one object only appears on one frame once
						break;
					}
				}
			}

			if(max_overlap > thresh) 
				det.obj_id = match_obj_id;
			//else
			//	cout << max_overlap << endl;
		}
	}

	detectionMatched = true;
}


void Evaluator::printScoreMatrix( int method, int if_bool ) const{
	float **matrix = (method == FIT) ? fit_matrix: match_matrix;
    bool **bool_matrix = (method == FIT)? fit_bool_matrix : match_bool_matrix;
    if( (matrix == NULL && !if_bool) || (bool_matrix == NULL && if_bool))  return;

    cout << (method == FIT ? "FIT: ": "MATCH: ")
         << "the verticle is ground truth, the horizontal is result"
         << "\n---------------------------------------------------------------------------------" << endl;

	cout << "\t";
	for(unsigned int i = 0; i < result.size(); i++)
		cout << i << " ";
	cout << "\n---------------------------------------------------------------------------------" << endl;

	for(unsigned int i = 0; i < ground_truth.size(); i++){
        int max_match_id = -1;
        float max_match = FLT_MIN;
		cout << i << " |\t";
		for(unsigned int j = 0; j < result.size(); j++){
			float score = matrix[i][j];
            if(score > max_match){
                max_match_id = j;
                max_match = score;
            }
            if(!if_bool)
                cout << (indicator(score, SCORE_THRESH) ? score : 0 ) << " ";
            else
                cout << bool_matrix[i][j] << " ";
		}
        if(max_match > SCORE_THRESH) 
            cout << " --> " << max_match_id;
		cout << endl;
	}
    cout << "---------------------------------------------------------------------------------\n" << endl;

}

void Evaluator::printScoreDetail(int gt_id, int result_id){
    std::map<int, TrajVec>::const_iterator it_ground_truth = ground_truth.find(gt_id);
    std::map<int, TrajVec>::const_iterator it_result = result.find(result_id);
	
    if( it_result == result.end() || it_ground_truth == ground_truth.end() )
		return ;
    
    TrajVec traj_result = it_result->second, traj_ground_truth = it_ground_truth->second;
    if(!align_trajectories(traj_ground_truth, traj_result))
		return ;
#if DEBUG
    cout << "FrameNum\t\tGround truth " << gt_id << "\t\tResult " << result_id << endl;
    cout << traj_ground_truth.size() << "\t" << traj_result.size() << endl;
#endif
    int traj_overlap_num = 0;
    for(unsigned int i = 0; i < traj_ground_truth.size(); i++){
#if DEBUG
        cout << i << "\t\t" << traj_ground_truth[i].box << "\t\t" << traj_result[i].box << endl;
#endif
        if(traj_ground_truth[i].box != Rect() && traj_result[i].box != Rect())
            traj_overlap_num++;
    }
    
    cout << "Ground truth [" << gt_id << "]\tResult [" << result_id << "]"
         << "\n---------------------------------------------------------------------------------\n" 
         << traj_overlap_num << "/" << traj_ground_truth.size() << " frames overlap" << endl;
    cout << "FIT score " << compute_score(gt_id, result_id, FIT) << endl;
    cout << "MATCH score " << compute_score(gt_id, result_id, MATCH) << endl;
    
}


void Evaluator::printMatchDetail(int from_set)  const{
    // match from ground truth to result or from result to ground truth
    const std::map<int, MatchPair >* match_result = (from_set == RESULT) ? &result_match : &ground_truth_match;

    if( match_result->empty() ){
        cout << "No match found" << endl;
        return;
    }

    cout << "Match result from " << (from_set == RESULT ? "result" : "ground truth")
         << " to " << (from_set == RESULT ? "ground truth" : "result") 
         << "\n---------------------------------------------------------------------------------" << endl;
    for( std::map<int, MatchPair >::const_iterator it = match_result->begin();
            it != match_result->end(); it++ ){
        int matched_id_ref = (from_set == RESULT) ? Evaluator::result_match_ref[it->first] : 
                     Evaluator::ground_truth_match_ref[it->first];
        // ATTENTION: object id starts from 0 
        cout << it->first << " to " << it->second.first << "(" << it->second.second << ") | "
             << matched_id_ref << "\t" << (it->second.first != matched_id_ref ? "**" : "") << endl;
    }

    cout << "\nFinal score is " << finalScore(from_set) 
         << "\n---------------------------------------------------------------------------------" << endl;
}


int Evaluator::getMatchedDetectionNum()	const {
	if(!detectionMatched)	return -1;	

	int true_pos = 0;

	// loop through detections on each frame
	for(const auto &dets: detections) {
		// loop through each detection on one frame
		for(const auto &det: dets.second) {
			if(det.obj_id > 0)
				++true_pos;
		}
	}

	return true_pos;
}

int Evaluator::getDetectionTrajOverlapNum()	const {

	if(!detectionMatched)	return false;

	int n = 0;
	for(const auto &obj_traj: ground_truth) {
		for(const auto &traj: obj_traj.second) {
			bool overlapped = false;
			for(const auto& dets: detections) {
				if(dets.first == traj.frame_id){
					for(const auto& det: dets.second) {
						// if the detection is matched to the ground truth
						if(det.obj_id == obj_traj.first) {
							overlapped = true;
							++n;
							break;
						}
					}	
				}
				if(overlapped)	break;
			}
			if(overlapped)	break;
		}
	}

	return n;
}

void Evaluator::printDetectionStats()	const {
	int det_sum = getDetectionSum();
	int gt_sum = getTrajUnitSum();
	int obj_sum = getObjectNum();
	int true_pos = getMatchedDetectionNum();

	float prec = (float)true_pos/det_sum;
	float recall = (float)true_pos/gt_sum;

	cout << "\n" << true_pos << " out of " << det_sum << " detections matched."
		 << "\n" << true_pos << " out of " << gt_sum << " ground truth boxes matched."
		 << "\nPrecision: " << prec << " recall: " << recall 
		 << "\n" << getDetectionTrajOverlapNum() << "/" << obj_sum << " objects at least overlap with one detection\n" << endl;
}

void Evaluator::writeDetectionStats(ofstream& outfile, const string& video_path)	const {
	// output format: true positive, false positive, detection sum, precision, recall, overlapped objects, object number, video_name
	if(outfile.is_open()){
		int det_sum = getDetectionSum();
		int gt_sum = getTrajUnitSum();
		int overlap_num = getDetectionTrajOverlapNum();
		int obj_sum = getObjectNum();
		int true_pos = getMatchedDetectionNum();
		size_t pos = video_path.find_last_of('/');
		string video_name = (pos == string::npos) ? video_path : video_path.substr(pos+1);
	
		float prec = (float)true_pos/det_sum;
		float recall = (float)true_pos/gt_sum;

		outfile << true_pos << " " << det_sum << " " << gt_sum << " " << overlap_num << " " << obj_sum 
				<< " "  << prec << " " << recall << " "<< video_name << endl;
	}
}

bool Evaluator::writeGroundTruthDetetionMatchToFile(const std::string& outfile)	const {
	if(!detectionMatched)	return false;
	ofstream out(outfile);
	if(out.is_open()) {
		for(const auto &obj_traj: ground_truth) {
			for(const auto &traj: obj_traj.second) {
				bool overlapped = false;
				for(const auto& dets: detections) {
					if(dets.first == traj.frame_id){
						for(const auto& det: dets.second) {
							// if the detection is matched to the ground truth
							// output format
							// groundtruthBox, detectionBox, overlapBox, overlapArea, overlapRatio
							if(det.obj_id == obj_traj.first) {
								overlapped = true;
								Rect r = det.box & traj.box;
								out	<< traj.box.x << " " << traj.box.y << " " << traj.box.width << " " << traj.box.height << " "
									<< det.box.x << " " << det.box.y << " " << det.box.width << " " << det.box.height << " "
									<< r.x << " " << r.y << " " << r.width << " " << r.height << " " 
									<< r.area() << " " << overlapRatio(det.box, traj.box) << endl;
								break;
							}
						}	
					}
					if(overlapped)	break;
				}
				if(overlapped)	break;
			}
		}

		out.close();
		return true;
	}
	return false;
}

bool Evaluator::visualize(const string& video_name, int start_frame, int obj_id, int result_or_gt)	const {
    VideoCapture capture(video_name);
    if(!capture.isOpened()){
        cerr << "Failed to open video " << video_name << endl;
        return false;
    }

    int frame_num = capture.get(CAP_PROP_FRAME_COUNT);
    int frame_rate = capture.get(CAP_PROP_FPS);

    bool play = false;
	bool draw_detection = (obj_id == -1);
    int end_frame = frame_num-1;

    Mat img;
	const std::map<int, TrajVec> & trajectory = result_or_gt == RESULT? result : ground_truth;
    std::map<int, TrajVec>::const_iterator it_result = trajectory.find(obj_id);

	// if the object is found, set the start and end frame of the object life
    if(it_result != trajectory.end()) {
        start_frame = it_result->second.front().frame_id;
        end_frame = it_result->second.back().frame_id;
    }else{	// if the object is not found, display all the objects
		cout << "Object " << obj_id << " is not found" << endl;
		obj_id = -1;
	}

	cout << frame_num << " frames, start from " << start_frame << endl;

	// if no object is specified, start from the first frame
	// otherwise, show the object frame by frame
	capture.set(CAP_PROP_POS_FRAMES, start_frame);
    for(int i = start_frame; i <= end_frame;){
        if(play || i == start_frame){
            capture >> img;
		
			resize(img, img, Size(0, 0), 2, 2, 2);
			if(!result.empty() && !ground_truth.empty() && obj_id == -1)
				drawTrajectoryMatch(img, i, GREEN, RED, 1, 1, 1);
			else {
				// draw objects
				if(result_or_gt == GROUND_TRUTH || obj_id == -1)
					drawTrajectory(img, RESULT, i, BLUE, 1, 1, 1, TOP_LEFT);
				if(result_or_gt == RESULT || obj_id == -1)
					drawTrajectory(img, GROUND_TRUTH, i, GREEN, 1, 1, 1, BOTTOM_RIGHT);
			}

			// draw detections
			if(draw_detection)
				drawDetections(img, i, BLUE, 1);
            i++;
        }
	
		imshow("video", img);
		// if only show one object, show it frame by frame
		if(obj_id != -1)
			play = false;

		// keyboard input
		char c = waitKey(frame_rate);
		if(c == ' ')
			play = !play;
		if(c == 'd')
			draw_detection = !draw_detection;
		if(c == 27)
			break;
    }
	cv::destroyWindow("video");
    return true;
}



bool Evaluator::saveToVideo(const string& video_path, string& out_video_name, int out_w, int out_h, double resize_factor, int to_frame)	const {
    VideoCapture capture(video_path);
    if(!capture.isOpened()){
        cerr << "Failed to open video " << video_path << endl;
        return false;
    }

    int frame_num = capture.get(CAP_PROP_FRAME_COUNT);
    double frame_rate = capture.get(CAP_PROP_FPS);
    int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	double fx = resize_factor, fy = resize_factor;
	bool resized = false;

	// if resize required
	if(out_w && out_h) {
		fx = out_w / (double) width;
		fy = out_h / (double) height;
		resized = true;
	}// only output width is given, computer resize factor by it
	else if(out_w){
		fx = out_w / (double) width;
		fy = fx;
		resized = true;
	}// only output width is given, computer resize factor by it
	else if(out_h){
		fy = out_h / (double) height;
		fx = fy;
		resized = true;

	}// resize facrtor given
	else{
		out_w = width*resize_factor;
		out_h = height*resize_factor;

		if(resize_factor != 1)	resized = true;
	}

	if(out_video_name.empty())
		out_video_name = getFileNameFromPath(video_path)+".mp4";

	VideoWriter writer(out_video_name, VideoWriter::fourcc('D', 'I', 'V', 'X'), frame_rate, Size(out_w, out_h), true);
    //VideoWriter writer(out_video_name, capture.get(CAP_PROP_FOURCC), frame_rate, Size(width, height), true);
    if(!writer.isOpened()){
        cerr << "Failed to write to video " << out_video_name << endl;
        return false;
    }

    Mat img;
	to_frame = (to_frame < 0 ? frame_num : to_frame);
//	capture.set(CAP_PROP_POS_FRAMES, 6309);

    for(int i = 0; i < to_frame; ++i){
        capture >> img;

		if(img.empty())	continue;
		if(resized)		resize(img, img, Size(out_w, out_h), fx, fy);
		
		drawDetections(img, i, YELLOW, 1, fx, fy);
		if(!result.empty() && !ground_truth.empty())
			drawTrajectoryMatch(img, i, GREEN, RED, 1, fx, fy);
		else {
			drawTrajectory(img, RESULT, i, BLUE, 1, fx, fy, TOP_LEFT);
			drawTrajectory(img, GROUND_TRUTH, i, GREEN, 1, fx, fy, BOTTOM_RIGHT);
		}
        writer.write(img);
    }
    cout << "Write " << frame_num << " frames to video " << out_video_name << endl;

    return true;
}
