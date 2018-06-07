# Semantic Part Detection with Object Context

**Parts-Object-Context** implements our work [1] on semantic part detection.
This method is an extension of Fast R-CNN [2] to jointly detect objects and their parts. 
It uses the object appearance and its class as indicators of what parts to expect.
Additionally, it also models the expected relative location of parts inside the objects based on their appearance, using a new network module called Offset Net.
By combining all these cues, our method detects parts in the context of their objects and significantly outperform part detection approaches using part appearance alone.
Code created by Abel Gonzalez-Garcia. 

Overview of our model:
<img src="http://calvin.inf.ed.ac.uk/wp-content/uploads/data/parts/modelObjPrt.png" alt="Overview of our part detection approach" width="100%">


## Overview
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Instructions](#instructions)
- [References](#references)
- [Disclaimer](#disclaimer)
- [Contact](#contact)

## Dependencies
- **MatConvNet:** beta24 (http://github.com/vlfeat/matconvnet)
- **Selective Search:** (http://koen.me/research/selectivesearch/)
- **Datasets:** 
  - **PASCAL VOC 2010:** (http://host.robots.ox.ac.uk/pascal/VOC/voc2010/)
  - **PASCAL-Part:** (http://www.stat.ucla.edu/~xianjie.chen/pascal_part_dataset/pascal_part.html)
- **Note:** This software builds on MatConvNet-Calvin (https://github.com/nightrome/matconvnet-calvin) and does _not_ work on Windows. 
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
## Instructions
- **Usage:** Run `demo_parts()`
- **What:** This script downloads the datasets (PASCAL VOC2010 and PASCAL-Part), network (AlexNet) and Selective Search code. It creates the structures with all the necessary object and part information.
After this, it first trains the baseline model for joint object and part detection and then it trains our model with object appearance and class branches, initialized with the previously trained baseline model. 
Finally, it trains Offset Net and merges it with the model that contains the object branches to obtain our final model. 
- **Model:** Training the baseline model takes about 8h on a Titan X GPU. On the same GPU, the model with object appearance and class takes about 10h and Offset Net takes 5h. If you just want to use them you can download the pretrained models in the installation step above. Then run the demo to see the test results.
- **Results:** If the program executes correctly, it will print the per-class results in average precision and their mean (mAP) for each of the 105 part classes in PASCAL-Part and 20 objects classes in PASCAL VOC. The baseline model achieves 22.0% mAP for parts (48.7% mAP for objects) on the validation set using no external training data nor bounding-box regression, whereas the model with object appearance and class achieves 25.9% mAP (49.9% mAP for objects). Finally, our full model with relative location achieves 27.3% mAP. 
- **Note:** The results vary due to the random order of images presented during training. To reproduce the above results we fix the initial seed of the random number generator.
 

## References
- \[1\] **Objects as Context for Part Detection** by Gonzalez-Garcia et al., CVPR 2018, http://arxiv.org/abs/1703.09529 
- \[2\] **Fast R-CNN (FRCN)** by Girshick et al., ICCV 2015, http://arxiv.org/abs/1504.08083

## Disclaimer
This software is covered by the FreeBSD License. See LICENSE.MD for more details.

## Contact
If you run into any problems with this code, please submit a bug report on the Github site of the project. For other inquiries contact a.gonzalez-garcia-at-sms.ed.ac.uk
