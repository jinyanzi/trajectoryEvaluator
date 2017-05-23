#! /bin/bash

make

if [[ $1 = "intersection_1" ]]
then
	./evaluator /shared/computer_vision/video_cut/intersection_1/intersection_1.avi /home/jenny/Dropbox/Bits_lab/data/tracking_result/intersection_1_result.txt /home/jenny/Dropbox/Bits_lab/data/ground_truth/intersection_1_ground_truth.txt 
fi

if [[ $1 = "intersection_2" ]]
then
	./evaluator /shared/computer_vision/video_cut/intersection_2/intersection_2.avi /home/jenny/Dropbox/Bits_lab/data/tracking_result/intersection_2_result.txt /home/jenny/Dropbox/Bits_lab/data/ground_truth/intersection_2_ground_truth.txt
fi

if [[ $1 = "intersection_3" ]]
then
	./evaluator /shared/computer_vision/video_cut/intersection_3/intersection_3.avi /home/jenny/Dropbox/Bits_lab/data/tracking_result/intersection_3_result.txt /home/jenny/Dropbox/Bits_lab/data/ground_truth/intersection_3_ground_truth.txt
fi

if [[ $1 = "intersection_4" ]]
then
	./evaluator /shared/computer_vision/video_cut/intersection_4/intersection_4.avi /home/jenny/Dropbox/Bits_lab/data/tracking_result/intersection_4_result.txt /home/jenny/Dropbox/Bits_lab/data/ground_truth/intersection_4_ground_truth.txt
fi

