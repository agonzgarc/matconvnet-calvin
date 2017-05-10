function [scores]  = scoreBoxesWithDispWindowsUnsorted(dispWindows, allBoxes, detScores)
        
    if nargin > 2
        ovWithProp = bsxfun(@times, computeOverlapTableSingle(allBoxes,single(dispWindows)), detScores');
    else
        ovWithProp = computeOverlapTableSingle(allBoxes,single(dispWindows));
    end
        
    scores = max(ovWithProp,[],2);
