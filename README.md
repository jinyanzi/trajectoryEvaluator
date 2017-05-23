# trajectoryEvaluator
A debugging tool to visualize trajectory and ground truth, output mathching videos.

Usage:
./evaluator -i <input_video>__
    input files:__
            -t <trajectory_path>__
            -g <ground_truth_path>__
            -d <detection_file_path>__
    output files:__
            -v <output_video_path>__
            -w <detection_stat_path> [write all the detection results to file in batch mode]__
            -c <new_ground_truth_path> [write cleaned up ground truth to file]__
            -u <tracking_stat_path> [write all the tracking results to file in batch mode]__
            -p <trajectory_gt_match_path> [write trajectory and ground truth match to file]__
            -q <detection_gt_match_path>	[write detection and ground truth match to file]__
            -x <output_init_file_path> [extract one box from ground truth for initialization]__
            -e <percentage> [percentage of lifetime to extract the ground truth]__
            -s <overlap_thresh> [overlap threshold for groundtruth matching with detection or trajectory]__
            -r <resize factor> [output video resize factor]__
            -h <resize height> [output height]__
            -l <resize width> [output width]__
    process parameters:__
            -f <to_frame_num> [number of frame to process]__
            -m [batch mod]__
