#include "mex.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace std;

float computeOverlapBBoxes(float bb1x1, float bb1y1, float bb1x2, float bb1y2, float area1,
                             float bb2x1, float bb2y1, float bb2x2, float bb2y2, float area2) {
        float xmin = std::max(bb1x1, bb2x1);
        float xmax = std::min(bb1x2, bb2x2);
        	
        if ((xmin > xmax)) // || (ymin > ymax))
	  //	  {
	  //mexPrintf("zero: %.4f %.4f ; %.4f %.4f\n", xmin, xmax, ymin, ymax);
                return 0;
	//}
	float ymin = std::max(bb1y1, bb2y1);
        float ymax = std::min(bb1y2, bb2y2);
	if (ymin > ymax)
	  return 0;
        else {
                float intersectionArea = (xmax - xmin + 1) * (ymax - ymin + 1);
                //mexPrintf("Intersection: %.4f %.4f ; %.4f %.4f\n", xmin, xmax, ymin, ymax);
                //mexPrintf("Intersection area: %.4f\n",intersectionArea);
                //mexPrintf("Area1: %.4f\n",area1);
                return intersectionArea/ area1; 
        }

}

double diffclock(clock_t clock1,clock_t clock2)
{
	double diffticks=clock1-clock2;
	double diffms=(diffticks)/CLOCKS_PER_SEC;
	return diffms;
}

void mexFunction(int nlhs, mxArray *plhs[],
                int nrhs, const mxArray *prhs[])
{
// input: two arrays of size nbWindows x 4 (x1,y1,x2,y2) and nbWindows2 x 4
// output: overlap table of size nbWindows x nbWindows2
  float * bboxes, *overlapTable, *bboxes2;
  int nbWindows, nbWindows2;
  int dim = 4;

  //check number of inputs
  if (nrhs != 2) {
    mexErrMsgTxt("Two input args required");
  }

  // check dimension of input
  if (mxGetN(prhs[0]) != 4 || mxGetN(prhs[1]) != 4) {
    mexErrMsgTxt("Dimension should be 4");
  }

  // load input data
  nbWindows = mxGetM(prhs[0]);
  bboxes = (float*) mxGetData(prhs[0]);
  bboxes2 = (float*) mxGetData(prhs[1]);
  nbWindows2 = mxGetM(prhs[1]);
  
  // create output matrix
  int dims[2];
  dims[0] = nbWindows;
  dims[1] = nbWindows2;
  plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
  overlapTable = (float*) mxGetData(plhs[0]);
  
  //mxSINGLE_CLASS 
  // precompute area of each bbox (?)
  vector<float> area(nbWindows); 
  vector<float> area2(nbWindows2);
  for (int i = 1; i <= nbWindows; ++i) {
    area[i-1] = (bboxes[i-1 +2*nbWindows]- bboxes[i-1] + 1)*(bboxes[i-1 + 3*nbWindows] - bboxes[i-1 + nbWindows] + 1);
  }
  for(int i = 0; i<nbWindows2; i++) {
    area2[i] = (bboxes2[i +2*nbWindows2]- bboxes2[i] + 1)*(bboxes2[i + 3*nbWindows2] - bboxes2[i + nbWindows2] + 1);
  }
  
  int kk = 0;
  // iterate over bboxes
  for (int j = 1; j <= nbWindows2; ++j) {
    for (int i = 1; i <= nbWindows; ++i) {
      overlapTable[kk++] = 
	  computeOverlapBBoxes(bboxes[i-1], bboxes[i-1 + nbWindows],// x1, y1
			     bboxes[i-1 + 2*nbWindows], bboxes[i-1 + 3*nbWindows],// x2, y2
			     area[i-1], // area i
			     bboxes2[j-1], bboxes2[j-1 + nbWindows2],// x1 y1
			     bboxes2[j-1 + 2*nbWindows2], bboxes2[j-1 + 3*nbWindows2],// x2 y2
			     area2[j-1]); // area j
    }     
  } 
}
