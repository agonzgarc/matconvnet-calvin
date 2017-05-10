function [scores]  = scoreBoxesWithDispWindowsUnsorted(dispWindows, allBoxes, detScores)
      
    % Compute overlap to each displaced window
    ov = zeros(size(allBoxes,1), size(dispWindows,1));
    for ii = 1:size(dispWindows)
       ov(:,ii) = BoxOverlap(allBoxes,dispWindows(ii,:)); 
    end
    
    if nargin > 2
        ovWithProp = bsxfun(@times, ov, detScores');
    else
        ovWithProp = ov;
    end
        
    scores = max(ovWithProp,[],2);

    