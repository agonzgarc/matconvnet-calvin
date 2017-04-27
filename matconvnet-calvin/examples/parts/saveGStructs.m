function saveGStructs(imSet)

global DATAopts;


trash = load(sprintf(DATAopts.imdb, imSet));
imdb = trash.imdb;
clear trash;
allIms = imdb.image_ids;


% Path where to store the Gstructs
mkdir(DATAopts.gStructPath);

offsetsPartIdx = [0; cumsum(cellfun(@(x) size(x,1), imdb.prt_classes))];
offsetsPartIdx = offsetsPartIdx(1:end-1);


saveDir = '/home/abel/LocalData/Graphics/GStructsObjPrt-FromScratch';
mkdir(saveDir);


for idxImg = 1:size(allIms,1)    
    fprintf('Processing img: %d/%d\n', idxImg, size(allIms,1));
    
    % Get Selective Search proposals, change to x,y order
    im = imread(sprintf(DATAopts.imgpath, allIms{idxImg}));
    im = im2double(im);
    selectiveSearchBoxes = selective_search_boxes_min(im, true, 500, 10);
    
    ssBoxes = selectiveSearchBoxes(:, [2 1 4 3]);
    
    % Logical variables that indicate which ssBoxes should be used for 
    % parts or objs
    prtBoxes = true(size(ssBoxes,1),1);
    
    % MAKE THIS SMALLER, REMOVE SMALL PROPOSALS
    objBoxes = true(size(ssBoxes,1),1);
    
    % Object GTs
    objGTBoxes = imdb.objects{idxImg}.bbox;
    objClass = imdb.objects{idxImg}.class_id;
    numObjGTs = size(objGTBoxes,1);


    % Part GTs
    prt_boxes = cell(size(imdb.parts{idxImg},1),1);
    prt_cls_gt = cell(size(imdb.parts{idxImg},1),1);
    for kk = 1:size(imdb.parts{idxImg},1)
        if size(imdb.parts{idxImg}{kk},1) > 0
            prt_boxes{kk} = imdb.parts{idxImg}{kk}.bbox;
            % Need double conversion so cell2mat doesn't crash if empty
            prt_cls_gt{kk} = double(offsetsPartIdx(imdb.objects{idxImg}.class_id(kk)) +  imdb.parts{idxImg}{kk}.class_id);
        end
    end
    
    prtGTBoxes = cell2mat(prt_boxes);
    prtClass = cell2mat(prt_cls_gt);
    numPrtGTs = size(prtGTBoxes,1);

   
    % Boxes variable contains ALL boxes
    boxStruct.boxes = single([prtGTBoxes; objGTBoxes; ssBoxes]);
    
    % Total number of boxes
    numBoxes = size(boxStruct.boxes,1);

    % Indicate whether boxes are for objects or parts
    boxStruct.boxesPrt = [true(numPrtGTs,1); false(numObjGTs,1); prtBoxes];
    boxStruct.boxesObj = [false(numPrtGTs,1); true(numObjGTs,1); objBoxes];

    % Indicate which boxes are gts for objects or parts
    boxStruct.gtPrt = [true(numPrtGTs,1); false(numObjGTs,1); false(size(ssBoxes,1),1)];
    boxStruct.gtObj = [false(numPrtGTs,1); true(numObjGTs,1); false(size(ssBoxes,1),1)];

    % Save class indices
    boxStruct.classPrt = uint16(zeros(numBoxes,1));
    boxStruct.classPrt(boxStruct.gtPrt) = prtClass;
    
    boxStruct.classObj = uint16(zeros(numBoxes,1));
    boxStruct.classObj(boxStruct.gtObj) = objClass;
    
    % Compute overlaps
    boxStruct.overlapPrt = zeros(numBoxes, imdb.prt_num_classes, 'single');
    % Get overlap wrt ground truth boxes
    for ii = 1:numPrtGTs
        boxStruct.overlapPrt(:, prtClass(ii)) = ...
          max(boxStruct.overlapPrt(:, prtClass(ii)), BoxOverlap(boxStruct.boxes, prtGTBoxes(ii, :)));
    end
    
    boxStruct.overlapObj = zeros(numBoxes, imdb.obj_num_classes, 'single');
    % Get overlap wrt ground truth boxes
    for ii = 1:numObjGTs
        boxStruct.overlapObj(:, objClass(ii)) = ...
          max(boxStruct.overlapObj(:, objClass(ii)), BoxOverlap(boxStruct.boxes, objGTBoxes(ii, :)));
    end

    % Compute insideness of boxes
    boxStruct.insideness = single(computeIoATableSingle(boxStruct.boxes, boxStruct.boxes));
    
    save([DATAopts.gStructPath allIms{idxImg} '.mat'], '-struct', 'boxStruct'); 
end
