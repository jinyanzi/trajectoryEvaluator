# trajectoryEvaluator
A debugging tool to visualize trajectory and ground truth, output mathching videos.

Usage:
./evaluator -i <input_video>
    input files:
            -t <trajectory_path>
            -g <ground_truth_path>
            -d <detection_file_path>
    output files:
            -v <output_video_path>
            -w <detection_stat_path> [write all the detection results to file in batch mode]
            -c <new_ground_truth_path> [write cleaned up ground truth to file]
            -u <tracking_stat_path> [write all the tracking results to file in batch mode]
            -p <trajectory_gt_match_path> [write trajectory and ground truth match to file]
            -q <detection_gt_match_path>	[write detection and ground truth match to file]
            -x <output_init_file_path> [extract one box from ground truth for initialization]
            -e <percentage> [percentage of lifetime to extract the ground truth]
            -s <overlap_thresh> [overlap threshold for groundtruth matching with detection or trajectory]
            -r <resize factor> [output video resize factor]
            -h <resize height> [output height]
            -l <resize width> [output width]
    process parameters:
            -f <to_frame_num> [number of frame to process]
            -m [batch mod]
