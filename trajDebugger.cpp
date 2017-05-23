#include "trajDebugger.h"
#include <fstream>
#include <vector>
#include <map>
#include <list>
// #include <cppconn/driver.h>
// #include <cppconn/exception.h>
// #include <cppconn/resultset.h>
// #include <cppconn/prepared_statement.h>
// #include <cppconn/statement.h>
// #include <mysql_connection.h>

using namespace std;
using namespace cv;
/*
bool TrajDebugger::readSQL(int result_or_gt, const string& video_name, const string& db_name, int to_frame_num){
    try {
        sql::Driver *driver;
        sql::Connection *con;
        sql::PreparedStatement *pstmt;
        sql::ResultSet *res;

        //  Create a connection 
        driver = get_driver_instance();
        con = driver->connect("tcp://127.0.0.1:3306", "root", "");
        con->setSchema(db_name);
        
        string table_name = ((result_or_gt == RESULT) ? "results": "ground_truth");
	    TrajVec tmp_trajectory;
        int last_obj_id = -1;

        // get video_id 
        pstmt = con->prepareStatement("SELECT video_id FROM videos WHERE video_name = ?");
        pstmt->setString(1, video_name);
        res = pstmt->executeQuery();
        if(res->next()){
            int  video_id = res->getInt(1);
            delete pstmt;
            delete res;

            // if to_frame_num is set, select trajectory up to to_frame_num
            // both result and trajectory and grond_truth has the table named "trajectory"
            // sort by object id at frame order

            string select_statement; 
            if(to_frame_num > 0){
                select_statement = "SELECT * FROM " + table_name + 
                    " WHERE video_id = ? AND frame_id <= ? ORDER BY obj_id, frame_id ASC";
                pstmt = con->prepareStatement(select_statement);
                pstmt->setInt(1, video_id);
                pstmt->setInt(2, to_frame_num);
            }// if to_frame_num == -1, select all the trajectories
            else{
                select_statement = "SELECT * FROM " + table_name + " WHERE video_id = ? ORDER BY obj_id, frame_id ASC";
                pstmt = con->prepareStatement(select_statement);
                pstmt->setInt(1, video_id);
            }

            res = pstmt->executeQuery();
            int i = 0;
            // go through the result set
            while(res->next()){
               int obj_id = res->getInt(1);
               int x = res->getInt(2);
               int y = res->getInt(3);
               int w = res->getInt(4);
               int h = res->getInt(5);
               int frame_id = res->getInt(6);
//               cout << "obj_id = " << obj_id << " frame_id = " << frame_id << " x = " << x << " y = " << y 
//                    << " w = " << w << " h = " << h << endl;

               if(last_obj_id < 0)
                   last_obj_id = obj_id;

               if(obj_id > last_obj_id){
                   if(!tmp_trajectory.empty()){
                       if(result_or_gt == RESULT)
                           result.insert(make_pair(last_obj_id, tmp_trajectory));
                       else
                           ground_truth.insert(make_pair(last_obj_id, tmp_trajectory));
                       tmp_trajectory.clear();
                   }
               }// add trajectory unit
               tmp_trajectory.push_back(TrajUnit(frame_id, Rect(x, y, w, h)));
               last_obj_id = obj_id;
               i++;
            }

            delete pstmt;
            delete res;
            cout << "read " << (result_or_gt == RESULT ? result.size() : ground_truth.size() ) << " objects " << i << " records" << endl;
        }

    } catch (sql::SQLException &e) {
        cout << "# ERR: SQLException in " << __FILE__;
        cout << "(" << __FUNCTION__ << ") on line "
            << __LINE__ << endl;
        cout << "# ERR: " << e.what();
        cout << " (MySQL error code: " << e.getErrorCode();
        cout << ", SQLState: " << e.getSQLState() <<
            " )" << endl;
        return false;
    }
    return true;
}
*/
bool TrajDebugger::readTrajFile(int result_or_gt, const string& filename, int to_frame_num) {
	if(filename.empty()) {
		cerr << "Filename empty" << endl;
		return false;
	}
	ifstream infile(filename.c_str());
	if(!infile.is_open()){
		cerr << "Failed to open file " << filename << endl;
		return false;
	}

	// Read trajectory data
	std::string line;
	int last_obj_id = -1;	
	TrajVec tmp_trajectory;

	while(!infile.eof()){
		// if the string keep content of last frame
		// use the last one without readling a new line
		std::getline(infile, line);
		if(line.empty())	continue;

		int tmp_obj_id, tmp_frame_id, tmp_occluded = 0;
		Rect tmp_box;
		string tmp_cls;
        // do not read objects appeared after to_frame_num
		readTrajLine(line, tmp_obj_id, tmp_frame_id, tmp_box, tmp_cls, tmp_occluded);
        if(last_obj_id < 0)
            last_obj_id = tmp_obj_id;
        //cout << " frame_id " << tmp_frame_id << " object " << tmp_obj_id << endl;

        // if the current frame exceeds the to_frame_num, skip it for other objects
        if( to_frame_num > 0 && tmp_frame_id > to_frame_num )
            continue;

        // If the line is of next object, insert the trajectory to the map
        if( tmp_obj_id != last_obj_id ){
            // do not insert objects appeared after to_frame_num
            if( !tmp_trajectory.empty() ){
                if(result_or_gt == GROUND_TRUTH)
                    ground_truth.insert(std::make_pair(last_obj_id, tmp_trajectory));
                else
                    result.insert(std::make_pair(last_obj_id, tmp_trajectory));
                // clear tmp_trajectory for next object
                tmp_trajectory.clear();
            }
        }// add trajectory unit

        // append trajectory record for the current object
        if(tmp_frame_id <= to_frame_num || to_frame_num == -1) 
            tmp_trajectory.push_back(TrajUnit(tmp_frame_id, tmp_box, tmp_occluded, tmp_cls));
        last_obj_id = tmp_obj_id;
	}

	if(!tmp_trajectory.empty())
		result_or_gt == RESULT ? result.insert(std::make_pair(last_obj_id, tmp_trajectory)) : ground_truth.insert(std::make_pair(last_obj_id, tmp_trajectory));

	infile.close();
	cout << "read " << (result_or_gt == RESULT ? result.size() : ground_truth.size()) << " objects to " << (result_or_gt ? "result" : "ground truth") << endl;
	// end of file, return true
	return true;
}


void TrajDebugger::readTrajLine(const string& line, int& obj_id, int& frame_id, Rect& box, string& cls, int & occluded)	const {
	char* tokens = NULL;
	const char* delims = " ";
	int x = 0, y = 0, w = 0, h = 0;

	tokens = strtok( strdup(line.c_str()), delims );
	int j = 0;
    while( tokens != NULL ){
        // [obj_num, xmin, ymin, xmax, ymax, frame_num]
        switch(j){
            // trajectory format: obj_id, x_top_left, y_top_left, width, height, frame_id
            // changed to be the same with 
            case 0: obj_id = atoi(tokens); break;
            case 1: x = atoi(tokens);	break;
            case 2: y = atoi(tokens);	break;
            case 3: w = atoi(tokens);	break;
            case 4: h = atoi(tokens);	break;
            case 5: frame_id = atoi(tokens);	break;
			case 6: occluded = atoi(tokens);
			// old ground truth has if_occluded in the 8th column, 
			// new ground truth has if_occluded in the 7th column
			case 7: if(isdigit(tokens[0])) occluded = atoi(tokens); else cls = tokens;	break;
			case 9: cls = tokens; break;
            default:	break;
        }

        tokens = strtok(NULL, delims);
		j++;
	}
	box = Rect(x,y,w,h);
}

int TrajDebugger::getTrajUnitSum()	const {
	int sum = 0;
	for(const auto obj_traj: ground_truth)
		sum += obj_traj.second.size();

	return sum;
}


MatchStruct TrajDebugger::trajVecMatch(const TrajVec& v1, const TrajVec& v2, int match_id) const {

	float overlap_sum = 0, ctr_dist = 0, frame_overlap_ratio = 0;
	int n = 0;

	// compute sum of overlapped area
	for(const auto& traj1: v1) {
		for(const auto& traj2: v2) {	
			if(traj1.frame_id == traj2.frame_id) {
				float f = overlapRatio(traj1.box, traj2.box);
				if(f > frame_overlap_ratio)	frame_overlap_ratio = f;

				overlap_sum += overlappedArea(traj1.box, traj2.box);
				ctr_dist += boxDist(traj1.box, traj2.box);
				++n;
				break;	// break loop to check next frame
			}	
		}
	}
	return MatchStruct(match_id, n, overlap_sum, ctr_dist, frame_overlap_ratio);
}


void TrajDebugger::areaSum(const std::map<int, TrajVec>& trajectory, std::map<int, float>& area_sum_map)	const {
	for(const auto& trajPair: trajectory) {
		// first make every ground truth unmatched
		for(const auto& traj: trajPair.second)
			area_sum_map[trajPair.first] += traj.box.area();
	}
}

void TrajDebugger::groundTruthTrajMatchByInit() {

	if(result.empty() || ground_truth.empty())	return;

	// clear previous match result if there exists
	gt_res_match.clear();
	res_gt_match.clear();

	// compute accumulated area overtime of each ground truth and trajectory result
	areaSum(ground_truth, gt_area_sum);
	areaSum(result, res_area_sum);

	// compute accumulated paiwise overlap area
	for(const auto& gt: ground_truth) {
		for(const auto& obj: result) {
			// if no overlapped in time, skip
			if(gt.second.front().frame_id > gt.second.back().frame_id 
					|| gt.second.back().frame_id < gt.second.front().frame_id)
				continue;

			// match by finding overlapped ground truth on the first frame of trajectory
			// Note that initialization on the first frame has been initialized on the second frame( frame_id == 1)
			// since background model needs two frames
			for(const auto& gtUnit: gt.second){
				if( (gtUnit.frame_id == obj.second.front().frame_id || (gtUnit.frame_id == 0 && obj.second.front().frame_id == 1)) 
						&& gtUnit.box == obj.second.front().box){
					MatchStruct match = trajVecMatch(gt.second, obj.second, obj.first);
					if(res_gt_match.find(obj.first) != res_gt_match.end())
						cout << "Result " << obj.first << " already assigned to " << res_gt_match[obj.first].matched_id << endl;
					else{
						match.overlap_ratio = match.overlap_area_sum/(gt_area_sum[gt.first] + res_area_sum[obj.first]-match.overlap_area_sum);
						gt_res_match[gt.first] = match;
						res_gt_match[obj.first] = match;
						res_gt_match[obj.first].matched_id = gt.first;
						// cout << gt.second.front().frame_id << " " << obj.second.front().frame_id << " "<< gt.second.back().frame_id << " " << obj.second.back().frame_id << endl;
					}
					break;
				}
			}
		}
	}

	printUnmatchedGroundTruth();
}


// match ground truth and trajectory
// find the largest overlapped ratio pair
// to take lifetime into account, use
// sum_t(overlapArea)/sum_t(unionArea)
// thresh is the threshold of the maxial frame overlap ratio
void TrajDebugger::groundTruthTrajMatch(float thresh) {
	if(result.empty() || ground_truth.empty())	return;
	
	if(thresh < 0) {
		groundTruthTrajMatchByInit();
		return;
	}

	// clear previous match result if there exists
	gt_res_match.clear();
	res_gt_match.clear();

	// a sparse matrix recording the accumulated area of ground truth and result
	// no entry if no overlap
	// key: ground_truth_id, value: vector of candidate matches
	std::map<int, vector<MatchStruct> > match_matrix;
	// list of unmatched vector. In case that one trajectory is mapped to more than one objects
	std::list<int> unmatched_gt;
	for(const auto& gt: ground_truth) 
		// first make every ground truth unmatched
		unmatched_gt.push_back(gt.first);

	// compute accumulated area overtime of each ground truth and trajectory result
	std::map<int, float> gt_area_sum, res_area_sum;
	areaSum(ground_truth, gt_area_sum);
	areaSum(result, res_area_sum);

	// compute accumulated paiwise overlap area
	for(const auto& gt: ground_truth) {
		for(const auto& obj: result) {
			
			// if no overlapped in time, skip
			if(gt.second.front().frame_id > gt.second.back().frame_id 
					|| gt.second.back().frame_id < gt.second.front().frame_id)
				continue;
			
			//cout << gt.second.back().frame_id << " " << obj.second.back().frame_id << endl;
			MatchStruct match = trajVecMatch(gt.second, obj.second, obj.first);
			// if threshold > 0, match by find the most overlapped object
			// a vector of overlapped trjectory for each ground truth object
			if(match.overlap_area_sum > 0) {
				std::map<int, vector<MatchStruct> >::iterator it = match_matrix.find(gt.first);
				//cout << match.matched_id << " " << match.overlap_area_sum << endl;
				// if there is no entry for the ground truth
				if(it == match_matrix.end())
					match_matrix[gt.first] = vector<MatchStruct>(1, match);
				else
					match_matrix[gt.first].push_back(match);
			}
		}
	}


	// pick the result with largest overlap ratio overlap
	// if the largest overlap is below the threshold, no match
	// every time check one ground truth at the front
	while(!unmatched_gt.empty()) {
		int gt_id = unmatched_gt.front();
		// get overlapped trajectory vector
		vector<MatchStruct> res_vec = match_matrix[gt_id];
		unmatched_gt.pop_front();
		// if the ground truth has no overlapped result trejctory, skip it.
		if(res_vec.empty())	continue;

		MatchStruct max_match;
		// iterate through each candidate trjectory
		for(auto& match: res_vec) {
			match.overlap_ratio = match.overlap_area_sum/(gt_area_sum[gt_id]+res_area_sum[match.matched_id]-match.overlap_area_sum);
			// check if this trajectory is already matched to other ground truth
			// if the trajectory is already assigned and the current overlap ratio is smaller than the previously assgined pair
			// keep the previous assignment
			if(res_gt_match.find(match.matched_id) != res_gt_match.end() 
					&& match.overlap_ratio < res_gt_match[match.matched_id].overlap_ratio) {
#if DEBUG
				cout << "\t*** Skip match: " << gt_id << "-" << match.matched_id << "(" << match.overlap_ratio << ") due to better match " 
					 << res_gt_match[match.matched_id].matched_id << "-" << match.matched_id << "(" << res_gt_match[match.matched_id].overlap_ratio << ")" << endl;
#endif
				continue;
			}

			if(match.overlap_ratio > max_match.overlap_ratio)
				max_match = match;	
		}

		if(max_match.matched_id < 0 || max_match.max_frame_overlap_ratio < thresh || max_match.overlap_frame_count < 10) {
#if DEBUG
			if(max_match.matched_id >= 0)
			cout << "\t>>>skip gt " << gt_id << "-res " << max_match.matched_id << " max frame overlap: " << max_match.max_frame_overlap_ratio << " overall overlap:" << max_match.overlap_ratio << " overlap frames: " << max_match.overlap_frame_count << endl;
			continue;
#endif
		}

		// if the ground truth has previously assigned to other trajectory and has larger overlap ratio
		if(res_gt_match.find(max_match.matched_id) != res_gt_match.end()) {
			int previous_matched_gt = res_gt_match[max_match.matched_id].matched_id;
			// if the previously assigned trajectory has smaller overlapped ratio than the current match
			// assign the match to the current trajectory and remove the entry for previous match
			// add to the previsouly assigned ground truth to unmatched list
			if(gt_res_match[previous_matched_gt].overlap_ratio < max_match.overlap_ratio) {
				gt_res_match.erase(gt_res_match.find(previous_matched_gt));
				unmatched_gt.push_back(previous_matched_gt);
#if DEBUG
			cout << "\t-->  Overlap match: " <<  previous_matched_gt << "-" << max_match.matched_id << "(" << res_gt_match[max_match.matched_id].overlap_ratio << ") with " << gt_id << "-" << max_match.matched_id << "(" << max_match.overlap_ratio << ")" << endl;
#endif
			} else
				continue;
		}

		// write match in both directions
		gt_res_match[gt_id] = max_match;
		res_gt_match[max_match.matched_id] = max_match;
		res_gt_match[max_match.matched_id].matched_id = gt_id;
	}

	printUnmatchedGroundTruth(match_matrix);	
}

void TrajDebugger::printUnmatchedGroundTruth(const std::map<int, std::vector<MatchStruct> >& match_matrix)	const {

	if(gt_res_match.empty()) {
		cout << "No match found" << endl;
		return;
	} else
		cout << gt_res_match.size() << " matches found " << endl;

	for(const auto& gt: ground_truth)
		if(gt_res_match.find(gt.first) == gt_res_match.end()) {
			std::map<int, TrajVec>::const_iterator gtVecIt = ground_truth.find(gt.first);
			cout << "+++Ground truth " << gt.first << " unmatched: (" << gtVecIt->second.front().frame_id << "-" << gtVecIt->second.back().frame_id << ")";

			std::map<int, std::vector<MatchStruct> >::const_iterator gtMatchIt = match_matrix.find(gt.first);
			if( gtMatchIt != match_matrix.end() && !gtMatchIt->second.empty()) {
				cout << "\t" << gtMatchIt->second.size() << " candidates: ";
				for(const auto& matchObj: gtMatchIt->second)
					cout << matchObj.matched_id << "(" << (res_gt_match.find(matchObj.matched_id) != res_gt_match.end() ? res_gt_match.find(matchObj.matched_id)->second.matched_id : -1) << ")\t";
			}
			cout << endl;
		}
}

void TrajDebugger::cleanGroundTruth() {
	std::map<int, float> max_dist_map;
	// merge overlapping objects and remove long stationary ones
	for(const auto & gt1: ground_truth) {
		for(const auto& gt2: ground_truth) {
			if(gt2.second.size() == 1) {
				ground_truth.erase(ground_truth.find(gt2.first));
				cout << "Remove " << gt2.first << " due to short life" << endl;
				continue;
			}
			// skip itself and non-overlap ground truth objects
			if(gt1.first == gt2.first || gt1.second.front().frame_id > gt2.second.back().frame_id 
					|| gt1.second.back().frame_id < gt2.second.front().frame_id)
				continue;	

			int overlap_frame_count = 0;
			// check pairwise overlapping and distance
			for(const auto & u1: gt1.second) {
				for(const auto & u2: gt2.second) {
					if (u1.frame_id == u2.frame_id && boxDist(u1.box, u2.box) <= STATIONARY_DIST &&
							overlapRatio(u1.box, u2.box) > 0.5) {
						++overlap_frame_count;
						break;
					}
				}
			}

			// overlap are at the beginning and ending of two segments
			if(overlap_frame_count > 3 && 
					(abs(gt1.second.front().frame_id - gt2.second.back().frame_id) <= 22 
					 || abs(gt2.second.front().frame_id - gt1.second.back().frame_id) <= 22)) {
				TrajVec newTraj;
				newTraj.reserve(gt1.second.size() + gt2.second.size());
				const TrajVec & frontTraj = (gt1.second.front().frame_id < gt2.second.front().frame_id) ? gt1.second : gt2.second;
				const TrajVec & backTraj = (gt1.second.front().frame_id >= gt2.second.front().frame_id) ? gt1.second : gt2.second;

				newTraj.insert(newTraj.end(), frontTraj.begin(), frontTraj.end());
				int k = 0;
				for(const auto& u: backTraj)
					// only copy the nonoverlap part
					if(u.frame_id > frontTraj.back().frame_id) {
						newTraj.push_back(u);
						++k;
					}

				newTraj.resize(frontTraj.size()+k);

				// accumulate the movement of each segent, incase the object moves in curve
				std::map<int, float>::iterator dist_it = max_dist_map.find(gt1.first);
				float d = boxDist(gt2.second.front().box, gt2.second.back().box);
				//if(d <= STATIONARY_DIST) d = 0;
				if(dist_it != max_dist_map.end())
					max_dist_map[gt1.first] += d;
				else
					max_dist_map[gt1.first] = boxDist(gt1.second.front().box, gt1.second.back().box) + d;

				ground_truth[gt1.first] = newTraj;
				ground_truth.erase(ground_truth.find(gt2.first));
#if DEBUG
				cout << "remove ground truth " << gt2.first << " due to overlap with object " << gt1.first << endl;
#endif
			}
		}
	}

	for(const auto & gt_dist: max_dist_map) {
		if(gt_dist.second < STATIONARY_DIST){
#if DEBUG
			cout << "remove ground truth " << gt_dist.first << " due to small movement " << gt_dist.second << endl;
#endif
			if(ground_truth.find(gt_dist.first) != ground_truth.end())
				ground_truth.erase(ground_truth.find(gt_dist.first));
			else
				cout << "object " << gt_dist.first << " has already been removed" << endl;
		}
	}
}


// find the start frame of a TrajVec for initialization
// if 0 <= percent < 1, take the proportation of lifetime
// if percent = 1, take the first frame that has area larger than 0.3*maxArea

int TrajDebugger::getInitStartIndex(const TrajVec& traj_vec, const float percent) const {
	if(traj_vec.empty() || percent < 0 || percent > 1)	return -1;

	// find max and min area of the trajectory
	int max_area = INT_MIN, min_area = INT_MAX;
	for(const auto& traj: traj_vec)	{
		if(traj.box.area() > max_area)
			max_area = traj.box.area();
		if(traj.box.area() < min_area)
			min_area = traj.box.area();
	}

	float area_thresh = (max_area-min_area)*percent + min_area;
	for(unsigned int i = 0; i < traj_vec.size(); ++i)
		if(traj_vec.at(i).box.area() >= area_thresh)
			return i;
	return -1;

}

bool TrajDebugger::writeInitBoxesToFile(const std::string& out_file, float init_percent)	const{
	// nothing to write
	if(ground_truth.empty() || out_file.empty())	return false;

	// no need to extract initialization when percent < 0 or percent > 1
	if(init_percent > 1 || init_percent < 0)	return false;
	ofstream out(out_file);
	if(!out.is_open()) return false;

	// temporary map to store initialization boxes, 
	// key frame number, value: vector of object boxes and last appeared frame
	std::map<int, std::vector<std::pair<int, cv::Rect> > > init_boxes;
	for(const auto& gt: ground_truth) {
		// start index of the ground truth vector
		int start_idx = getInitStartIndex(gt.second, init_percent);
		if(start_idx < 0 || start_idx >= (int)gt.second.size())	continue;
		// start frame number
		int start_frame = gt.second[start_idx].frame_id;

		if(init_boxes.find(start_frame) == init_boxes.end()) {
			init_boxes[start_frame] = vector<std::pair<int, cv::Rect> >(1, std::make_pair(gt.second.back().frame_id, gt.second[start_idx].box));	
		} else 
			init_boxes[start_frame].push_back(std::make_pair(gt.second.back().frame_id, gt.second[start_idx].box));
	}

	int num = 0;
	for(const auto& frame_boxes: init_boxes)
		for( const auto& box: frame_boxes.second)
			out << num++ << " " << box.second.x << " " << box.second.y << " " << box.second.width << " " << box.second.height << " " << frame_boxes.first << " " << box.first << endl;

	out.close();
	return true;
}


bool TrajDebugger::writeTrajectoryToFile(int result_or_gt, const string& traj_path)	const {
	const std::map<int, TrajVec> &trajectories = (result_or_gt== RESULT) ? result : ground_truth;
	if(trajectories.empty())	return false;

	ofstream out(traj_path);
	if(out.is_open()){
		for(const auto& traj: trajectories)
			for(const auto vec: traj.second)
				out << traj.first << " " << vec.box.x << " " << vec.box.y << " " << vec.box.width << " " << vec.box.height << " " << vec.frame_id << " " << vec.if_occluded << " " << vec.cls << endl;
		out.close();
		return true;
	}
	return false;
}

bool TrajDebugger::writeGroundTruthTrajMatchStats(ofstream& out_file, const string& video_path)	const{
	if(gt_res_match.empty())	return false;

	size_t pos = video_path.find_last_of('/');
	string video_name = (pos == string::npos) ? video_path : video_path.substr(pos+1);

	if(out_file.is_open()) {
		out_file << result.size() << " " << ground_truth.size() << " " << gt_res_match.size() << " " << video_name << endl;
		return true;
	}
	return false;
}

bool TrajDebugger::writeGroundTruthTrajMatchToFile(const string& out_file) const {

	if(gt_res_match.empty())	return false;

	ofstream out(out_file);
	if(out.is_open()) {
		// output format
		// frameID groundTruthID objectID groundTruthBox objectBox overlapBox overlapArea centerDistance
		// if frameID == 0, all the boxes are empty and 
		// overlapRatio and centerDistance are the summary of ground truth and object
		for(const auto& gt_res_pair: gt_res_match) {
			std::map<int, TrajVec>::const_iterator gt_it = ground_truth.find(gt_res_pair.first);
			std::map<int, TrajVec>::const_iterator res_it = result.find(gt_res_pair.second.matched_id);
			if(gt_it != ground_truth.end() && res_it != result.end()) {
				for(const auto& gt: gt_it->second)
					for(const auto& res: res_it->second)
						if(gt.frame_id == res.frame_id) {
							Rect r = gt.box & res.box;
							out << gt.frame_id << " " << gt_it->first << " " << res_it->first << "  " 
								<< gt.box.x << " " << gt.box.y << " " << gt.box.width << " " << gt.box.height << " "
								<< res.box.x << " " << res.box.y << " " << res.box.width << " " << res.box.height << " "
								<< r.x << " " << r.y << " " << r.width << " " << r.height << " " 
								<< r.area() << " " << boxDist(res.box, gt.box) << " " << overlapRatio(gt.box, res.box) 
								<< gt.box.area() << " " << res.box.area() << endl;
							break;
						}
				// summary
				std::map<int, float>::const_iterator gt_area_sum_it = gt_area_sum.find(gt_it->first);
				std::map<int, float>::const_iterator res_area_sum_it = res_area_sum.find(res_it->first);
				out << "-1 " << gt_it->first << " " << res_it->first << " 0 0 0 0 0 0 0 0 0 0 0 0 " 
					<< gt_res_pair.second.overlap_area_sum << " " << gt_res_pair.second.center_dist_sum 
					<< " " << gt_res_pair.second.overlap_ratio << " " 
					<< (gt_area_sum_it != gt_area_sum.end() ? gt_area_sum_it->second : 0) << " " 
					<< (res_area_sum_it != res_area_sum.end() ? res_area_sum_it->second : 0) << endl;
			}
		}
		out.close();
		return true;
	}
	return false;
}

void TrajDebugger::printTrajectorySummary(int result_or_gt, int obj_id)	const{
	const std::map<int, TrajVec> &trajectories = (result_or_gt== RESULT) ? result : ground_truth;
    cout << ((result_or_gt == RESULT) ? "Result" : "Ground Truth") << endl;
    // obj_id == -1, print summary
    if(obj_id == -1){
		for(const auto& traj: trajectories)
            cout << "object " << traj.first << " (" << traj.second.front().frame_id << "-" 
                << traj.second.back().frame_id << "|" << traj.second.size() << ")" << endl;

        cout << "----------------------------------------------------------------\n" 
            << trajectories.size() << " objects in total " << endl;
    }// print object trajectory
    else{
		int k = -1;
		float max_dist = FLT_MIN;
        std::map<int, TrajVec>::const_iterator it = trajectories.find(obj_id);
        if( it != trajectories.end() ){
			for(unsigned int i = 0; i < it->second.size(); ++i) {
				if(i > 0) {
					float s = boxDist(it->second.at(i-1).box, it->second.at(i).box);
					cout << "-- " << s << " --> ";
					if(s > max_dist) {
						max_dist = s;
						k = i;
					}
				}
                cout << it->second.at(i).frame_id << " " << it->second.at(i).box;
			}
            cout << endl;

			cout << "\nLifetime: " << it->second.size();
			if(it->second.empty())
				cout << endl;
			else
				cout << " Frame " << it->second.front().frame_id << "-" << it->second.back().frame_id << endl;
			if( k > 0)
				cout << "max dist " << it->second.at(k-1).frame_id << "-" << it->second.at(k).frame_id << ": " << max_dist << endl;
        }
    }
}

void TrajDebugger::printMatchSummary()	const{
	if(gt_res_match.empty() || res_gt_match.empty()) {
		cout << "No match found" << endl;
		return;
	}		

	for(const auto& match:gt_res_match) {
		int res_id = match.second.matched_id;
		std::map<int, TrajVec>::const_iterator gt_it = ground_truth.find(match.first);
		std::map<int, TrajVec>::const_iterator res_it = result.find(res_id);
		Rect gt_box;
		if(gt_it != ground_truth.end() && res_it != result.end()) {
			for(unsigned int i = 0; i < gt_it->second.size(); ++i)
				if(gt_it->second[i].frame_id == res_it->second.front().frame_id){
					gt_box = gt_it->second[i].box;
					break;
				}
			cout << "gt " << match.first << "(" << gt_it->second.front().frame_id << "-" << gt_it->second.back().frame_id 
				 << ")-res " << match.second.matched_id << "(" << res_it->second.front().frame_id << "-" << res_it->second.back().frame_id 
				 << ") max overlap " << match.second.max_frame_overlap_ratio << (match.second.max_frame_overlap_ratio < 0.5 ? " *":"") << endl;
		}
	}
}

void TrajDebugger::drawNumBox(cv::Mat& img, const Rect& r, const Scalar& color, int num, double fx, double fy, int location)	const {
	Mat overlay = img.clone();
	float alpha = 0.6;
    //Point p;
    //if(location == TOP_LEFT){
    //    p = r.tl();
    //    p.x += 3*fx;
    //    p.y += 8*fy;
    //}else{
    //    p = r.br();
    //    p.x -= 8*fx;
    //    p.y -= 8*fy;
    //}
    //// draw number
    //stringstream ss;
    //ss << num;

    rectangle(img, r, color, -1);
	//addWeighted(overlay, alpha, img, 1-alpha, 0, img);
	//if(num >= 0)
    //putText(img, ss.str(), p, FONT_HERSHEY_SIMPLEX, 0.4*fx, color, std::max(1.0, fx));
}

int TrajDebugger::drawObject( cv::Mat& img, const std::map<int, TrajVec>::const_iterator it, int to_frame_num, const cv::Scalar& color, int thickness, double fx, double fy, int location)	const {
   
    // do not draw objects have not appeared or left 
    if( it->second.size() < 2 || it->second.front().frame_id > to_frame_num 
            || it->second.back().frame_id < to_frame_num )
        return -1;
    TrajVec::const_iterator it_traj = it->second.begin(), it_last_traj = it_traj++;
    for(; it_traj != it->second.end(); ++it_traj){ 
        if( it_traj->box != Rect() && it_last_traj->box != Rect() ){
            line(img, scale_point(center(it_traj->box), fx, fy), scale_point(center(it_last_traj->box), fx, fy), color, thickness*fx); 
            it_last_traj = it_traj;
        }// only draw trajcetory up to to_frame_num
        if(it_traj->frame_id >= to_frame_num)
            break;
    }
    // draw box
    if( it_traj != it->second.end() && it_traj->frame_id == to_frame_num) {
		drawNumBox(img, scale_rect(it_traj->box, fx, fy), color, it->first, fx, fy, location);
	}
#if DEBUG
    // first box
    if( it->second.front().frame_id == to_frame_num )
        cout << "\nobject " << it->first << "(" << it->second.front().frame_id << "-" << it->second.back().frame_id << "):\t" << it->second.front().box << endl;
#endif

    // return the index of object at to_frame_num
    return it_last_traj - it->second.begin();
}

int TrajDebugger::drawObjectMatch( cv::Mat& img, const std::map<int, TrajVec>::const_iterator it, int to_frame_num, const cv::Scalar& matched_color, const cv::Scalar& unmatched_color, int thickness, double fx, double fy, int location)	const {
   
    // do not draw objects have not appeared or left 
    if( it->second.size() < 2 || it->second.front().frame_id > to_frame_num 
            || it->second.back().frame_id < to_frame_num )
        return -1;
    TrajVec::const_iterator it_traj = it->second.begin(), it_last_traj = it_traj++;
    for(; it_traj != it->second.end(); ++it_traj){ 
        if( it_traj->box != Rect() && it_last_traj->box != Rect() ){
			cv::Scalar c = res_gt_match.find(it->first) != res_gt_match.end() ? matched_color : unmatched_color;
            line(img, scale_point(center(it_traj->box), fx, fy), scale_point(center(it_last_traj->box), fx, fy), c, thickness*fx); 
            it_last_traj = it_traj;
        }// only draw trajcetory up to to_frame_num
        if(it_traj->frame_id >= to_frame_num)
            break;
    }
	
	
    // draw box
    if( it_traj != it->second.end() && it_traj->frame_id == to_frame_num) {
		Rect gt_box, overlap_box;
		// if the trajectory is matched to a ground truth
		// copy the matched part first
		std::map<int, MatchStruct>::const_iterator match_it = res_gt_match.find(it->first);
		if( match_it != res_gt_match.end()) {
			std::map<int, TrajVec>::const_iterator gt_it = ground_truth.find(match_it->second.matched_id);
			if(gt_it != ground_truth.end()) {
				for(const auto& b:gt_it->second)
					if(b.frame_id == to_frame_num) {
						gt_box = b.box;
						break;
					}
			}
			overlap_box = scale_rect(gt_box & it_traj->box, fx, fy);
		}

		Mat overlap_roi = img(overlap_box).clone();
		drawNumBox(img, scale_rect(gt_box, fx, fy), unmatched_color, match_it->second.matched_id, fx, fy, location);
		drawNumBox(img, scale_rect(it_traj->box, fx, fy), unmatched_color, it->first, fx, fy, location);
		if(overlap_box.area()) {
			overlap_roi.copyTo(img(overlap_box));
			drawNumBox(img, overlap_box, matched_color, -1, fx, fy, location);
		}
	}

#if DEBUG
    // first box
    if( it->second.front().frame_id == to_frame_num )
        cout << "\nobject " << it->first << "(" << it->second.front().frame_id << "-" << it->second.back().frame_id << "):\t" << it->second.front().box << endl;
#endif

    // return the index of object at to_frame_num
    return it_last_traj - it->second.begin();
}


void TrajDebugger::drawTrajectory(cv::Mat& img,int result_or_gt, int to_frame_num, const cv::Scalar& color, int thickness, double fx, double fy, int location, int obj_id)	const {
	const std::map<int, TrajVec> &trajectories = (result_or_gt== RESULT) ? result : ground_truth;
    
	//stringstream ss;
    //ss << "frame " << to_frame_num;
    //putText(img, ss.str(), Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.4*fx, GREEN, std::max(1.0, fx));

	// return if the trajectory is empty
	if(trajectories.empty())	return;

    // draw all the objects
    if(obj_id == -1 ){
        for(std::map<int, TrajVec>::const_iterator it = trajectories.begin(); 
                it != trajectories.end(); it++ ) {
			Scalar c = color;
			// if the trajectory is matched to a ground truth
			if(!it->second.empty() && it->second.front().cls == "\"people\"")
				c = PINK;

            drawObject(img, it, to_frame_num, c, thickness, fx, fy, location);
		}
    }// draw only one object
    else{
        const std::map<int, TrajVec>::const_iterator it = trajectories.find(obj_id);
        if(it != trajectories.end()) {
			Scalar c = color;
			if(!it->second.empty() && it->second.front().cls == "\"people\"")
				c = PINK;

            drawObject(img, it, to_frame_num, c, thickness, fx, fy, location);
		}
    }
}


void TrajDebugger::drawTrajectoryMatch(cv::Mat& img, int to_frame_num, const cv::Scalar& matched_color, const cv::Scalar& unmatched_color, int thickness, double fx, double fy, int obj_id)	const {
	if(result.empty() || ground_truth.empty()) {
		cerr << "No enough data for matching" << endl;
		return;
	}
	
    //putText(img, "C-COT", Point(5, 30), FONT_HERSHEY_SIMPLEX, 0.5*fx, GREEN, std::max(1.0, fx));

	// draw all the objects
    if(obj_id == -1 ){
        for(std::map<int, TrajVec>::const_iterator it = result.begin(); 
                it != result.end(); it++ ) 
            drawObjectMatch(img, it, to_frame_num, matched_color, unmatched_color, thickness, fx, fy);
    }// draw only one object
    else{
        const std::map<int, TrajVec>::const_iterator it = result.find(obj_id);
        if(it != result.end())
            drawObjectMatch(img, it, to_frame_num, matched_color, unmatched_color, thickness, fx, fy);
    }

}


string TrajDebugger::getFileNameFromPath(const string& file_path)   const{
    size_t s1 = file_path.find_last_of("/");
    size_t s2 = file_path.rfind(".");
    return file_path.substr(s1+1, s2-s1-1);
}


string TrajDebugger::getFileDirPath(const string& file_path)   const{
    size_t s = file_path.find_last_of("/");
    return s == string::npos ? "" :file_path.substr(0,s);
}

