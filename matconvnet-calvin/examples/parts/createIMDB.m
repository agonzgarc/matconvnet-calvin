function [imdb] = createIMDB(imSet)

% Created by Davide Modolo, 2016


global DATAopts;

% List of images
fID = fopen(sprintf(DATAopts.imgsetpath, imSet));
images = textscan(fID, '%s');
images = images{1};
fclose(fID);

% Create imdb
imdb.name = ['voc_2010_' imSet '_parts'];
 
% IMAGE INFO
imdb.image_dir = DATAopts.imgpath(1:end-6);
imdb.image_ids = images;
imdb.extension = 'jpg';
imdb.image_at = @(i)sprintf('%s/%s.%s',imdb.image_dir,imdb.image_ids{i},imdb.extension);
imdb.sizes = zeros(length(images), 2);

% OBJECTS INFO
imdb.obj_num_classes = 20;
imdb.obj_classes = {'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',...
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', ...
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'}';
imdb.obj_class2id = containers.Map(imdb.obj_classes, 1:imdb.obj_num_classes);

% PART INFOS
tmp = part2ind_4imdb();
tmp2 = part2names_4imdb();
 
imdb.prt_num_classes = 105;
for i = 1:imdb.obj_num_classes
    t = tmp2{i}.keys';
    for j = 1:length(unique(cell2mat(tmp2{i}.values)))
        imdb.prt_classes{i, 1}{j, 1} = t{cell2mat(tmp2{i}.values) == j};   % <= problem here. keys get sorted by name !!!!!!
    end
    t = tmp{i}.keys';
    for j = 1:length(unique(cell2mat(tmp{i}.values)))
        imdb.prt_original_classes{i, 1}{j, 1} = {t{cell2mat(tmp{i}.values) == j}};   % <= problem here. keys get sorted by name !!!!!!
    end 
end 
imdb.prt_class2id = tmp2;
imdb.prt_original_class2id = tmp;
 

imdb.mapping = [];

% ------------- LOOP OVER THE IMAGES
for i = 1:length(images)
    
    fprintf('Image:  %d/%d\n', i, length(images));
     
    % Read original pascal image
    img = imread(imdb.image_at(i));
    imdb.sizes(i, 1) = size(img, 1);
    imdb.sizes(i, 2) = size(img, 2); 

    % load annotation (object ) -- original from pascal   
    % there are needed to assign viewpoint to an object
    rec = VOCreadrecxml(sprintf(DATAopts.annopath_obj, images{i}));
    clear ob;
    for oo = 1:length(rec.objects)        
        ob.class{oo} = rec.objects(oo).class;
        ob.view{oo} = rec.objects(oo).view;
        ob.bbox(oo, :) = rec.objects(oo).bbox;
    end
          
    % load annotation (boxes & parts) -- from pascal parts
    load(sprintf(DATAopts.annopath_prt, images{i}));
    
    % ------------- LOOP OVER THE OBJECTS IN THE IMAGE
    for oo = 1:numel(anno.objects)      
        
        % class info (name and id)
        imdb.objects{i, 1}.class{oo, 1} = anno.objects(oo).class;                       
        imdb.objects{i}.class_id(oo, 1) = anno.objects(oo).class_ind;                
        
        % get bounding box of the object 
        imdb.objects{i}.bbox(oo, :) = getSegmentBoundingBox(anno.objects(oo).mask);  
                
        % check if pascal voc has viewpoint anno for this object instance
        % direct mapping of the boxes doesn't work (new boxes comes from
        % pixel-wise segmentations and tend to be more accurate
        % instead, we compute IoU between the boxes and check if they match
        
        % check what objects in rec have the same name
        objIndx = ismember(ob.class, anno.objects(oo).class);
        boxes = ob.bbox(objIndx, :);

        % compute IoU and assign viewpoint
        IoU = BoxOverlap(boxes, imdb.objects{i}.bbox(oo, :));

        [mIoU, iIoU] = max(IoU);
        if mIoU > 0.4
            imdb.objects{i}.viewpoint{oo, 1} = ob.view(objIndx(iIoU));
        else
            imdb.objects{i}.viewpoint{oo, 1} = [];
        end    
        
        obj_im = img(imdb.objects{i}.bbox(oo, 2):imdb.objects{i}.bbox(oo, 4), ...
            imdb.objects{i}.bbox(oo, 1):imdb.objects{i}.bbox(oo, 3), :);
        imdb.objects{i}.sizes(oo, 1) = size(obj_im, 1); 
        imdb.objects{i}.sizes(oo, 2) = size(obj_im, 2); 
                
        % get annotations for all parts of object 'oo'
        obj = anno.objects(oo); 
        
        % no part annotations for this object instances
        if numel(obj.parts) == 0
            imdb.parts{i, 1}{oo, 1} = [];
        end
        
        % add field to the structure
        [obj.parts(:).skip]=deal(0);
        pc = 0;
        
        % ------------- LOOP OVER THE PARTS IN THE OBJECT
        for pp = 1:numel(obj.parts)
            
            % if we flagged a part as 'already visited' (in case we merged
            % it to a previously selected part), then skip it.
            if obj.parts(pp).skip 
                continue;
            end
            
            % Check if we are ignoring this part, skip if so
            if sum(strcmp(obj.parts(pp).part_name, imdb.prt_original_class2id{imdb.objects{i}.class_id(oo, 1)}.keys)) == 0
                obj.parts(pp).skip = 1;
                continue;
            end
                
            
            % part counter. using pp directly would cause 'wholes' in the
            % structure when skipping a part
            pc = pc + 1;
            
            % some parts in the dataset need to be merged (for example:
            % left upper arm and left lower arm -> left arm - which for use is just arm)
            classes_with_possible_merge = [10, 13, 15, 17];
            parts2merge = {'lfuleg', 'lflleg', 'rfuleg', 'rflleg', ...
                'lbuleg', 'lblleg', 'rbuleg', 'rblleg', ...
                'llarm', 'luarm', 'rlarm', 'ruarm', ...
                'llleg', 'luleg', 'rlleg', 'ruleg'};
            if ismember(imdb.objects{i}.class_id(oo), classes_with_possible_merge) && ...
                    ~isempty(strmatch(obj.parts(pp).part_name, parts2merge)) %#ok<MATCH2>
                
                % find the other part (name) that should be merged with current
                position = find(strncmp(obj.parts(pp).part_name, parts2merge,length(obj.parts(pp).part_name)));

                if mod(position, 2) 
                    merge_with_name = parts2merge(position + 1);
                else
                    merge_with_name = parts2merge(position - 1);
                end
                
                % we know the name, let's look for it
                position_merge_with = find(strcmp({obj.parts.part_name}, merge_with_name) == 1);
                
                if isempty(position_merge_with)
                    
                    % the part we want to merge with is occluded
                    imdb.parts{i, 1}{oo, 1}.class{pc, 1} = obj.parts(pp).part_name;            
                    imdb.parts{i}{oo}.class_id(pc, 1) = imdb.prt_original_class2id{obj.class_ind}(obj.parts(pp).part_name);           
                    imdb.parts{i}{oo}.bbox(pc, :) = getSegmentBoundingBox(obj.parts(pp).mask);
                    
                else 
                    imdb.parts{i, 1}{oo, 1}.class{pc, 1} = [obj.parts(pp).part_name, '+', obj.parts(position_merge_with).part_name];            
                    imdb.parts{i}{oo}.class_id(pc, 1) = imdb.prt_original_class2id{obj.class_ind}(obj.parts(pp).part_name);

                    mask = obj.parts(pp).mask + obj.parts(position_merge_with).mask;
                    mask(mask > 1) = 1;
                    imdb.parts{i}{oo}.bbox(pc, :) = getSegmentBoundingBox(mask);
                    
                    % TEST for correctness: 
                    % display the two individual boxes + the new (merged) one
                    % X = [getSegmentBoundingBox(obj.parts(pp).mask); getSegmentBoundingBox(obj.parts(position_merge_with).mask); imdb.parts{i}{oo}.bbox(pp, :)];
                    % showboxes(img, X);
                    
                    obj.parts(position_merge_with).skip = 1;
                end
                                                 
            else
                % get bounding box of the part
                imdb.parts{i, 1}{oo, 1}.class{pc, 1} = obj.parts(pp).part_name;            
                imdb.parts{i}{oo}.class_id(pc, 1) = imdb.prt_original_class2id{obj.class_ind}(obj.parts(pp).part_name);           
                imdb.parts{i}{oo}.bbox(pc, :) = getSegmentBoundingBox(obj.parts(pp).mask);
           
            end
            
            % size of the part
            prt_im = img(imdb.parts{i}{oo}.bbox(pc, 2):imdb.parts{i}{oo}.bbox(pc, 4), ...
                imdb.parts{i}{oo}.bbox(pc, 1):imdb.parts{i}{oo}.bbox(pc, 3), :);
            imdb.parts{i}{oo}.sizes(pc, 1) = size(prt_im, 1); 
            imdb.parts{i}{oo}.sizes(pc, 2) = size(prt_im, 2); 

            % flag the part difficult if it is smaller than 20x20
            if imdb.parts{i}{oo}.sizes(pc, 1) < 20 || imdb.parts{i}{oo}.sizes(pc, 2) < 20                        
                imdb.parts{i}{oo}.difficult(pc, :) = 1;
            else
                imdb.parts{i}{oo}.difficult(pc, :) = 0;
            end 

            % 4 columns matrix:
            % col 1: part ID (within the object class)
            % col 2: object ID 
            % col 3: object index within the image (some images have multiple objects)
            % col 4: image number
            imdb.mapping = [imdb.mapping; [imdb.parts{i}{oo}.class_id(pc), imdb.objects{i}.class_id(oo), oo, i]]; 
        end
        
    end
end  

