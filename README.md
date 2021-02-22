# 3D Pose Based Motion Correction for Physical Exercises

_Master project in Life Sciences Engineering, CVLAB (EPFL), 2021_  

**<div align="center"> Summary </div>
With the rise of self-management for treatment of musculoskeletal disorders and especially during these times of pandemic, people tend to exercise alone and without supervision. Recent progresses in fields of pose estimation, action recognition and motion prediction allow us to analyze movements in details and thus identify potential mistakes done while exercising. In this work, we prepare a dataset containing videos, 2D and 3D poses of correct and incorrect executions of different movements that are SQUATS, lunges, planks and pick- ups and labels identifying the mistake in each practice of that exercise. This dataset is used to demonstrate our motion correction model, designed using a graph convolutional network architecture and trained with a differentiable dynamic time warping loss. As a result we are able to correct movement mistakes in 3D pose sequences and output the corrected motion. This model is integrated in a pipeline containing a state-of-the-art 3D human pose estimator to go from raw video images to a sequence of corrected 3D poses. Evaluation of this model is done using an action recognition model trained on the same dataset to recognize whether the sequence is correct or has a particular type of mistake. Results show that our model is successful in correcting incorrect sequences, as most of the time the resulting motions are classified as correct.**

## Dataset

**Examples of acquired images for each action, subject and camera.**
<p float="center">
  <img src="Images/Dataset.png" width="100%" />
</p>

**Example of 3D ground truth poses computation.**
<p float="center">
  <img src="Images/Dataset2.png" width="100%" />
</p>

## Correction model

### Network architecture
<p float="center">
  <img src="Images/corr.png" width="100%" />
</p>

### Results
[![Foo](Images/res1.png)](Images/res1.html)
[![Foo](Images/res2.png)](Images/res2.html)


