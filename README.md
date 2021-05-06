# AE4317_individual
Individual assignment for the course AE4317 Autonomous Flight of Micro-Air Vehicles, featuring a computationally cheap object detection computer vision algorithm based on Harris corner detection.

# Instructions
Check the requirements.txt and make sure all listed libraries are installed and working properly.
Unzip the WashingtonOBRace zip file into the main directory. 
Run object_detection.py

# Additional information
Set video to 'True' in line 282 to get a 'live' video of the camera footage, mask and predicted binary mask. 
Additionally, uncommenting lines 100 through 106 lets you view all the inbetween steps of the object detection algorithm. 
WARNING: Turning on video causes the entire program to take extremely long to run (personally, I usually killed the process to get out of viewing the video after a while).

Usual time to run for me was in the 1500 second range with no video. 
