# Project name
Hand and Palm Detection

#### Mentor

> Utkarsh Chauhan

#### Members

||Name|
|-|-|
|1|Cavin Macwan|

#### About Project 
![image](https://user-images.githubusercontent.com/60755716/113109263-720e9100-9223-11eb-971a-e4f22eb3ca86.png)

This project is made using python, and I used OpenCV and mediapipe for hand recognition and hand tracking. 
Hand tracking is useful for forming the basis for sign language understanding and hand gesture control, and 
can also enable the overlay of digital content and information on top of the physical world in augmented reality.


#### Workflow
Initially, we approached the project in the following sequence:
1. Getting the webcam feed
1. Recognizing the palm
1. Tracking the landmarks of the palm
1. Drawing lines and circle on those tracking landmarks


Here Cavin used OpenCV for getting the webcam feed. Then he converted that from BGR to RGB using OpenCV. Followed by tracking 
the landmarks using mediapipe library.MediaPipe Hands utilizes an ML pipeline consisting of multiple models working together: A 
palm detection model that operates on the full image and returns an oriented hand bounding box. A hand landmark model that operates 
on the cropped image region defined by the palm detector and returns high-fidelity 3D hand keypoints. And finally, the landmarks are drawn 
on the hand using OpenCV.

