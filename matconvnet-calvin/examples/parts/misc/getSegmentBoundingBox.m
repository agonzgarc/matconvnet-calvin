% Returns the smallest possible bounding box containing all true pixels
function bounding_boxes = getSegmentBoundingBox(input)
    if ~iscell(input)
        bounding_boxes = segmentation_bounding_box(input);
    else
        frames_num = length(input);
        bounding_boxes = cell(frames_num, 1);
        for frame = 1: frames_num
            bounding_boxes{frame} = ...
                segmentation_bounding_box(input{frame});
        end
    end
end

function bounding_box = segmentation_bounding_box(segmentation)
    [xCoord, yCoord] = find(segmentation);
    upperCorner = [min(yCoord), min(xCoord)];
    lowerCorner = [max(yCoord), max(xCoord)];
    bounding_box = [upperCorner, lowerCorner];
end 