#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include "evaluator.h"

using namespace std;

// input path
std::string video_path,		
			ground_truth_path, 
			traj_path, 
			det_path, 
			// output path
			new_gt_path, 
			det_stat_path, 
			tracking_stat_path, 
			det_match_path, 
			traj_match_path, 
			output_video_name, 
			out_init_file_path;
int to_frame_num = -1;
int out_w = 0, out_h = 0;
float overlap_thresh = OVERLAP_THRESH, // the threshold to do matching between ground truth and detection/tracking 
	  init_percent = 0,	// percentage of lifetime to extract the initialization frame
	  resize_factor = 1;
bool batch_mode = false;

void print_help() {
	cout << "./evaluator\n\t\t-i <video_name>\n\t\t"
		 << "-t <result_file>\n\t\t-g <ground_truth_file>\n\t\t"
		 << "-d <detection_file>\n\t\t-v <output_video_name(optional)>\n\t\t" 
		 // output files
		 << "-w <detection_stat_path> [write all the detection results to file in batch mode]\n\t\t" 
		 << "-c <new_ground_truth_path> [write cleaned up ground truth to file]\n\t\t"
		 << "-u <tracking_stat_path> [write all the tracking results to file in batch mode]\n\t\t"
		 << "-p <trajectory_gt_match_path> [write trajectory and ground truth match to file]\n\t\t"
		 << "-q <detection_gt_match_path>	[write detection and ground truth match to file]\n\t\t"
		 << "-x <output_init_file_path> [extract one box from ground truth for initialization]\n\t\t"
		 << "-e <percentage> [percentage of lifetime to extract the ground truth]\n\t\t"
		 << "-s <overlap_thresh> [overlap threshold for groundtruth matching with detection or trajectory]\n\t\t"
		 << "-r <resize factor> [output video resize factor]\n\t\t"
		 << "-h <resize height> [output height]\n\t\t"
		 << "-l <resize width> [output width]\n\t\t"
		 // process parameters
		 << "-f <to_frame_num> [number of frame to process]\n\t\t"
		 << "-m [batch mod]\n\t\t"
		 << endl;
}

void parse_opts(int argc, char ** argv, Evaluator & evaluator) {

	if(argc < 3){
		print_help();
		exit(1);
	}

	int opt;
	
	while((opt = getopt(argc, argv, "i:t:g:d:w:s:u:f:v:p:q:c:e:x:r:m")) != -1) {
		switch(opt) {
		case 'i':
			video_path = optarg;
			break;
		case 't':
			traj_path = optarg;
			evaluator.readResultFromFile(traj_path);
			break;
		case 'g':
			ground_truth_path = optarg;
			evaluator.readGroundTruthFromFile(ground_truth_path);
			break;
		case 'd':

			det_path = optarg;
			evaluator.readDetectionFile(det_path);
			break;
		case 'c':
			new_gt_path = optarg;
			break;
		case 'x':
			out_init_file_path = optarg;
			break;
		case 'v':
			output_video_name = optarg;
			break;
		case 'f':
			to_frame_num = atoi(optarg);
			break;
		case 'l':
			out_w = atoi(optarg);
			break;
		case 'h':
			out_h = atoi(optarg);
			break;
		case 's':
			overlap_thresh = atof(optarg);
			break;
		case 'e':
			init_percent = atof(optarg);
			break;
		case 'r':
			resize_factor = atof(optarg);
			break;
		case 'w':
			det_stat_path = optarg;
			break;
		case 'u':
			tracking_stat_path = optarg;
			break;
		case 'p':
			traj_match_path = optarg;
			break;
		case 'q':
			det_match_path = optarg;
			break;
		case 'm':
			batch_mode = true;
			break;
		default:
			break;
		}
	}

	// clean ground truth, and write to new file
	if(!ground_truth_path.empty() && !new_gt_path.empty()) {
		evaluator.cleanGroundTruth();
		evaluator.writeTrajectoryToFile(GROUND_TRUTH, new_gt_path);
	}
}


int main(int argc, char** argv){

	Evaluator evaluator;
	parse_opts(argc, argv, evaluator);

	// if ground truth provided, match and evaluate
	if(!ground_truth_path.empty()) {
		// if ground truth and detection results available, do ground truth-detection matching
		if(!det_path.empty()) 
			evaluator.groundTruthDetectionMatch(overlap_thresh);

		// if ground truth and tracking results available, do ground truth trajectory matching
		if(!traj_path.empty())
			// if initialization is given, just look match with start/end frame
			evaluator.groundTruthTrajMatch(overlap_thresh);

		// write initialization boxes to file
		if(!out_init_file_path.empty()) 
			evaluator.writeInitBoxesToFile(out_init_file_path, init_percent);

		ofstream detection_outfile, tracking_outfile;
		// if output file provided, write detection match statistics to file
		if( !det_stat_path.empty() && !det_path.empty()) {
			detection_outfile.open(det_stat_path.c_str(), std::ofstream::app);
			if(detection_outfile.is_open()) {
				evaluator.writeDetectionStats(detection_outfile, video_path);
				detection_outfile.close();
			}
		}

		// if output file provided, write tracking match statistics to file
		if(!traj_path.empty() && !tracking_stat_path.empty()) {
			tracking_outfile.open(tracking_stat_path.c_str(), std::ofstream::app);
			if(tracking_outfile.is_open()) {
				evaluator.writeGroundTruthTrajMatchStats(tracking_outfile, video_path);
				tracking_outfile.close();
			}
		}

		// write detailed trajectory-ground truth match to file
		if(!traj_path.empty() && !traj_match_path.empty())
			evaluator.writeGroundTruthTrajMatchToFile(traj_match_path);

		// write detailed detection-grond truth match to file
		if(!det_path.empty() && !det_match_path.empty())
			evaluator.writeGroundTruthDetetionMatchToFile(det_match_path);

	}


	// write display video to file
	if(!output_video_name.empty())	evaluator.saveToVideo(video_path, output_video_name, out_w, out_h, resize_factor, to_frame_num);


	if(!batch_mode) {
		int res, gt;
		string line;
		while(1){
			cout << "\n"
				 << "1. Visualize ground truth and result, detections if provided [1 <start_frame> <obj_id>]\n"
				 << "2. Print ground truth summary\n"
				 << "3. Print trajectory summary\n"
				 << "4. Print detection summary\n"
				 << "5. Print ground truth detail\n"
				 << "6. Print trajectory detail\n"
				 << "7. Print match summary\n"
				 << "8. Compare pairwise ground truth and result detail\n"
				 << "9. save to video file\n" 
				 //<< "9. Print FIT ground_truth-to-result match\n"
				 //<< "10. Print FIT result-to-ground_truth match\n"
				 //<< "11. Print MATCH ground_truth-to-result match\n"
				 //<< "12. Print MATCH result-to-ground_truth match\n"
				 << "0. Exit" << endl;
			int n, s = 0, id = -1;
			getline(std::cin, line);
			stringstream ss(line);
			ss >> n;
			switch(n){
				case 1:
					if(ss)	ss >> s;
					if(ss)	ss >> id;
					if(!video_path.empty())
						evaluator.visualize(video_path, s, id);
					else
						cerr << "No video file input" << endl;
					break;
				case 2:
					evaluator.printTrajectorySummary(GROUND_TRUTH);    break;
				case 3:
					evaluator.printTrajectorySummary(RESULT);	break;
				case 4:
					evaluator.printDetectionSummary();    break;
				case 5:
					cout << "Please enter a ground truth number " << endl;
					cin >> gt;
					evaluator.printTrajectorySummary(GROUND_TRUTH, gt);  break;
				case 6:
					cout << "Please enter a trajectory number " << endl;
					cin >> res;

					evaluator.printTrajectorySummary(RESULT, res);	break;
				case 7:
					evaluator.printMatchSummary();
					break;

				case 8:
					cout << "Please enter two numbers you want to compair: <ground_truth_id> <result_id>" << endl;
					cin >> gt >> res;
					if(res > 0 && gt >= 0)
						evaluator.printScoreDetail(gt, res);
					break;
				case 9:
					evaluator.saveToVideo(video_path, output_video_name, out_w, out_h, resize_factor, to_frame_num);
					break;
				//case 9: 
				//	evaluator.greedyMatch(FIT, GROUND_TRUTH);
				//	evaluator.printMatchDetail(GROUND_TRUTH);
				//	break;
				//case 10:
				//	evaluator.greedyMatch(FIT, RESULT);
				//	evaluator.printMatchDetail(RESULT);
				//	break;
				//case 11:
				//	evaluator.greedyMatch(MATCH, GROUND_TRUTH);
				//	evaluator.printMatchDetail(GROUND_TRUTH);
				//	break;
				//case 12:
				//	evaluator.greedyMatch(MATCH, RESULT);
				//	evaluator.printMatchDetail(RESULT);
				//	break;
				case 0:
					return 0;

			}

		}
	}

}
