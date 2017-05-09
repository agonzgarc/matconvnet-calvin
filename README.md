# MatConvNet-Calvin-Parts

**MatConvNet-Calvin** is a wrapper around MatConvNet that (re-)implements
several state of-the-art papers in object detection and semantic segmentation. 
Calvin is a Computer Vision research group at the University of Edinburgh (http://calvin.inf.ed.ac.uk/). Copyrights by Holger Caesar and Jasper Uijlings, 2015-2016.

**MatConvNet-Calvin-Parts** extends MatConvNet-Calvin for part detection, following our work [1]. 
The model is based on Fast R-CNN [2] and jointly detects objects and their parts. Part detection is aided by the object class and appearance on top of the part appearance. Additionally, we use the expected relative location of parts inside the objects, based on their appearance. 
Code created by Abel Gonzalez-Garcia, 2016. 

Overview of our model:
<img src="http://calvin.inf.ed.ac.uk/wp-content/uploads/data/parts/modelObjPrt.png" alt="Overview of our part detection approach" width="100%">


## Overview
- [Methods](#methods)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Instructions](#instructions)
- [References](#references)
- [Disclaimer](#disclaimer)
- [Contact](#contact)

## Methods
- **Part Detection with Object Context** \[1\]: Recent part detection approach that improves over methods that only use part appearance. This is an extension of the code of [2] to accept object context for part detection. It also includes a new architecture dubbed Offset Net, with essentially an adaptive number of outputs. 
- **Fast R-CNN (FRCN)** \[2\]: State-of-the-art object detection method. The original code was implemented for Caffe. This reimplementation ports it to MatConvNet by adding region of interest pooling and a simplified version of bounding box regression.

## Dependencies
- **Note:** This software does _not_ work on Windows. 
- **MatConvNet:** beta20 (http://github.com/vlfeat/matconvnet)
- **MatConvNet-FCN:** (http://github.com/vlfeat/matconvnet-fcn)
- **Selective Search:** (http://koen.me/research/selectivesearch/)
- **Datasets:** 
  - **PASCAL VOC 2010:** (http://host.robots.ox.ac.uk/pascal/VOC/voc2010/)
  - **PASCAL-Part:** (http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html)

## Installation
- Install Matlab R2015a (or newer) and Git
- Clone the repository and its submodules from your shell
  - `git clone https://github.com/agonzgarc/matconvnet-calvin.git`
  - `cd matconvnet-calvin`
  - `git submodule update --init`
- Execute the following Matlab commands
  - Setup MatConvNet
    - `cd matconvnet/matlab; vl_compilenn('EnableGpu', true); cd ../..;`
  - Setup MatConvNet-Calvin
    - `cd matconvnet-calvin/matlab; vl_compilenn_calvin(); cd ../..;`
  - Add files to Matlab path
    - `setup();`
 - (Optional) Download pretrained models:
    - Parts: `downloadModel('parts_baseline'); downloadModel('parts_objappcls'); downloadModel('parts_offsetnet')`
    - FRCN: `downloadModel('frcn');`
## Instructions
- **Usage:** Run `demo_parts()`
- **What:** This script downloads the datasets (PASCAL VOC2010 and PASCAL-Part), network (AlexNet) and Selective Search code. It creates the structures with all the necessary object and part information.
After this, it first trains the baseline model for joint object and part detection and then it trains our model with object appearance and class branches, initialized with the previously trained baseline model. 
Finally, it trains Offset Net and merges it with the model that contains the object branches to obtain our final model. 
- **Model:** Training the baseline model takes about 8h on a Titan X GPU. On the same GPU, the model with object appearance and class takes about 10h and Offset Net takes 5h. If you just want to use them you can download the pretrained models in the installation step above. Then run the demo to see the test results.
- **Results:** If the program executes correctly, it will print the per-class results in average precision and their mean (mAP) for each of the 105 part classes in PASCAL-Part and 20 objects classes in PASCAL VOC. The baseline model achieves 22.0% mAP for parts (48.7% mAP for objects) on the validation set using no external training data nor bounding-box regression, whereas the model with object appearance and class achieves 25.9% mAP (49.9% mAP for objects). 
- **Note:** The results vary due to the random order of images presented during training. To reproduce the above results we fix the initial seed of the random number generator.
 

## References
- \[1\] **Objects as Context for Part Detection** by Gonzalez-Garcia et al., arXiv 2017, http://arxiv.org/abs/1703.09529 
- \[2\] **Fast R-CNN (FRCN)** by Girshick et al., ICCV 2015, http://arxiv.org/abs/1504.08083

## Disclaimer
This software is covered by the FreeBSD License. See LICENSE.MD for more details.

## Contact
If you run into any problems with this code, please submit a bug report on the Github site of the project. For other inquiries contact a.gonzalez-garcia-at-sms.ed.ac.uk
