function pimap = part2ind_4imdb()
% Define the part index of each objects. 
% One can merge different parts by using the same index for the
% parts that are desired to be merged. 
% For example, one can merge 
% the left lower leg (llleg) and the left upper leg (luleg) of person by setting: 
% pimap{15}('llleg')      = 19;               % left lower l    eg
% pimap{15}('luleg')      = 19;               % left upper leg
%
% Created by Davide Modolo, 2016


pimap = cell(20, 1);                    
% Will define part index map for the 20 PASCAL VOC object classes in ascending
% alphabetical order (the standard PASCAL VOC order). 
for ii = 1:20
    pimap{ii} = containers.Map('KeyType','char','ValueType','int32');
end

% [1aeroplane]
pimap{1}('body')        = 1;                
pimap{1}('stern')       = 2; 

pimap{1}('lwing')       = 3;                % left wing
pimap{1}('rwing')       = 3;                % right wing

% pimap{1}('tail')        = 4;                

for ii = 1:10
    pimap{1}(sprintf('engine_%d', ii)) = 4; % multiple engines
end
% for ii = 1:10
%     pimap{1}(sprintf('wheel_%d', ii)) = 6;  % multiple wheels
% end

% [2bicycle]
pimap{2}('fwheel')      = 1;                % front wheel
pimap{2}('bwheel')      = 1;                % back wheel

pimap{2}('saddle')      = 2;               
pimap{2}('handlebar')   = 3;                % handle bar
pimap{2}('chainwheel')  = 4;                % chain wheel
% 
% for ii = 1:10
%     pimap{2}(sprintf('headlight_%d', ii)) = 5;
% end

% [3bird] 
pimap{3}('head')        = 1;

% pimap{3}('leye')        = 2;                % left eye
% pimap{3}('reye')        = 2;                % right eye

pimap{3}('beak')        = 2;               
pimap{3}('torso')       = 3;            
pimap{3}('neck')        = 4;

pimap{3}('lwing')       = 5;                % left wing
pimap{3}('rwing')       = 5;                % right wing

pimap{3}('lleg')        = 6;                % left leg
pimap{3}('rleg')        = 6;               % right leg

pimap{3}('rfoot')       = 7;               % right foot
pimap{3}('lfoot')       = 7;               % left foot

pimap{3}('tail')        = 8;

% [4boat]
% only has silhouette mask 

% [5bottle]
pimap{5}('cap')         = 1;
pimap{5}('body')        = 2;

% [6bus]
pimap{6}('frontside')   = 1;
pimap{6}('leftside')    = 2;
pimap{6}('rightside')   = 3;
pimap{6}('backside')    = 4;
pimap{6}('roofside')    = 5;

pimap{6}('leftmirror')  = 6;
pimap{6}('rightmirror') = 6;

pimap{6}('fliplate')    = 7;                % front license plate
pimap{6}('bliplate')    = 7;                % back license plate

for ii = 1:10
    pimap{6}(sprintf('door_%d',ii)) = 8;
end
for ii = 1:10
    pimap{6}(sprintf('wheel_%d',ii)) = 9;
end
% for ii = 1:10
%     pimap{6}(sprintf('headlight_%d',ii)) = 10;
% end
for ii = 1:20
    pimap{6}(sprintf('window_%d',ii)) = 10;
end

% [7car]
pimap{7}('frontside')   = 1;
pimap{7}('leftside')    = 2;
pimap{7}('rightside')   = 3;
pimap{7}('backside')    = 4;
pimap{7}('roofside')    = 5;

% pimap{7}('leftmirror')  = 6;
% pimap{7}('rightmirror') = 6;

pimap{7}('fliplate')    = 6;                % front license plate
pimap{7}('bliplate')    = 6;                % back license plate

for ii = 1:10
    pimap{7}(sprintf('door_%d',ii)) = 7;
end
for ii = 1:10
    pimap{7}(sprintf('wheel_%d',ii)) = 8;
end
for ii = 1:10
    pimap{7}(sprintf('headlight_%d',ii)) = 9;
end
for ii = 1:20
    pimap{7}(sprintf('window_%d',ii)) = 10;
end

% [8cat]
pimap{8}('head')        = 1;

pimap{8}('leye')        = 2;                % left eye
pimap{8}('reye')        = 2;                % right eye

pimap{8}('lear')        = 3;                % left ear
pimap{8}('rear')        = 3;                % right ear

pimap{8}('nose')        = 4;
pimap{8}('torso')       = 5;   
pimap{8}('neck')        = 6;

pimap{8}('lfleg')       = 7;               % left front leg
pimap{8}('rfleg')       = 7;               % right front leg
pimap{8}('lbleg')       = 7;               % left back leg
pimap{8}('rbleg')       = 7;               % right back leg

pimap{8}('lfpa')        = 8;               % left front paw
pimap{8}('rfpa')        = 8;               % right front paw
pimap{8}('lbpa')        = 8;               % left back paw
pimap{8}('rbpa')        = 8;               % right back paw

pimap{8}('tail')        = 9;               

% [9chair]
% only has sihouette mask 

% [10cow]
pimap{10}('head')       = 1;

% pimap{10}('leye')       = 2;                % left eye
% pimap{10}('reye')       = 2;                % right eye

pimap{10}('lear')       = 2;                % left ear
pimap{10}('rear')       = 2;                % right ear

pimap{10}('muzzle')     = 3;

pimap{10}('lhorn')      = 4;                % left horn
pimap{10}('rhorn')      = 4;                % right horn

pimap{10}('torso')      = 5;            
pimap{10}('neck')       = 6;

pimap{10}('lfuleg')     = 7;               % left front upper leg
pimap{10}('lflleg')     = 7;               % left front lower leg

pimap{10}('rfuleg')     = 7;               % right front upper leg
pimap{10}('rflleg')     = 7;               % right front lower leg

pimap{10}('lbuleg')     = 7;               % left back upper leg
pimap{10}('lblleg')     = 7;               % left back lower leg

pimap{10}('rbuleg')     = 7;               % right back upper leg
pimap{10}('rblleg')     = 7;               % right back lower leg

pimap{10}('tail')       = 8;               

% [11diningtable]
% only has silhouette mask 

% [12dog]
pimap{12}('head')        = 1;

% pimap{12}('leye')        = 2;                % left eye
% pimap{12}('reye')        = 2;                % right eye

pimap{12}('lear')        = 2;                % left ear
pimap{12}('rear')        = 2;                % right ear

pimap{12}('nose')        = 3;
pimap{12}('torso')       = 4;   
pimap{12}('neck')        = 5;

pimap{12}('lfleg')       = 6;                % left front leg
pimap{12}('rfleg')       = 6;               % right front leg
pimap{12}('lbleg')       = 6;               % left back leg
pimap{12}('rbleg')       = 6;               % right back leg

pimap{12}('lfpa')        = 7;               % left front paw
pimap{12}('rfpa')        = 7;               % right front paw
pimap{12}('lbpa')        = 7;               % left back paw
pimap{12}('rbpa')        = 7;               % right back paw

pimap{12}('tail')        = 8;   
pimap{12}('muzzle')     = 9;          		% muzzle


% [13horse]
pimap{13}('head')       = 1;
% 
% pimap{13}('leye')       = 2;                % left eye
% pimap{13}('reye')       = 2;                % right eye

pimap{13}('lear')       = 2;                % left ear
pimap{13}('rear')       = 2;                % right ear

pimap{13}('muzzle')     = 3;
pimap{13}('torso')      = 4;            
pimap{13}('neck')       = 5;

pimap{13}('lfuleg')     = 6;               % left front upper leg
pimap{13}('lflleg')     = 6;               % left front lower leg

pimap{13}('rfuleg')     = 6;               % right front upper leg
pimap{13}('rflleg')     = 6;               % right front lower leg

pimap{13}('lbuleg')     = 6;               % left back upper leg
pimap{13}('lblleg')     = 6;               % left back lower leg

pimap{13}('rbuleg')     = 6;               % right back upper leg
pimap{13}('rblleg')     = 6;               % right back lower leg

pimap{13}('tail')       = 7; 
%  
% pimap{13}('lfho')       = 9;                     % hoof
% pimap{13}('rfho')       = 9;
% pimap{13}('lbho')       = 9;
% pimap{13}('rbho')       = 9;

% [14motorbike]
pimap{14}('fwheel')     = 1;
pimap{14}('bwheel')     = 1;

pimap{14}('handlebar')  = 2;
% pimap{14}('saddle')     = 3;
for ii = 1:10
    pimap{14}(sprintf('headlight_%d', ii)) = 3;
end

% [15person]
pimap{15}('head')       = 1;
% 
% pimap{15}('leye')       = 2;                    % left eye
% pimap{15}('reye')       = 2;                    % right eye
% 
% pimap{15}('lear')       = 3;                    % left ear
% pimap{15}('rear')       = 3;                    % right ear
% 
% pimap{15}('lebrow')     = 4;                    % left eyebrow    
% pimap{15}('rebrow')     = 4;                    % right eyebrow
% 
% pimap{15}('nose')       = 5;                    
% pimap{15}('mouth')      = 6;                    
pimap{15}('hair')       = 2;                   
pimap{15}('torso')      = 3;                   
pimap{15}('neck')       = 4;  

pimap{15}('llarm')      = 5;                   % left lower arm
pimap{15}('luarm')      = 5;                   % left upper arm

pimap{15}('rlarm')      = 5;                   % right lower arm
pimap{15}('ruarm')      = 5;                   % right upper arm

pimap{15}('lhand')      = 6;                   % left hand
pimap{15}('rhand')      = 6;                   % right hand

pimap{15}('llleg')      = 7;               	% left lower leg
pimap{15}('luleg')      = 7;               	% left upper leg

pimap{15}('rlleg')      = 7;               	% right lower leg
pimap{15}('ruleg')      = 7;               	% right upper leg

pimap{15}('lfoot')      = 8;               	% left foot
pimap{15}('rfoot')      = 8;               	% right foot

% [16pottedplant]
pimap{16}('pot')        = 1;
pimap{16}('plant')      = 2;

% [17sheep]
pimap{17}('head')       = 1;

% pimap{17}('leye')       = 2;                % left eye
% pimap{17}('reye')       = 2;                % right eye

pimap{17}('lear')       = 2;                % left ear
pimap{17}('rear')       = 2;                % right ear

pimap{17}('muzzle')     = 3;

pimap{17}('lhorn')      = 4;                % left horn
pimap{17}('rhorn')      = 4;                % right horn

pimap{17}('torso')      = 5;            
pimap{17}('neck')       = 6;

pimap{17}('lfuleg')     = 7;               % left front upper leg
pimap{17}('lflleg')     = 7;               % left front lower leg

pimap{17}('rfuleg')     = 7;               % right front upper leg
pimap{17}('rflleg')     = 7;               % right front lower leg

pimap{17}('lbuleg')     = 7;               % left back upper leg
pimap{17}('lblleg')     = 7;               % left back lower leg

pimap{17}('rbuleg')     = 7;               % right back upper leg
pimap{17}('rblleg')     = 7;               % right back lower leg

pimap{17}('tail')       = 8;               

% [18sofa]
% only has sihouette mask 

% [19train]
pimap{19}('head')       = 1;
pimap{19}('hfrontside') = 2;                	% head front side                
pimap{19}('hleftside')  = 3;                	% head left side
pimap{19}('hrightside') = 4;                	% head right side
% pimap{19}('hbackside')  = 5;                 	% head back side
pimap{19}('hroofside')  = 5;                	% head roof side
% 
for ii = 1:10
    pimap{19}(sprintf('headlight_%d',ii)) = 6;
end

for ii = 1:10
    pimap{19}(sprintf('coach_%d',ii)) = 7;
end

for ii = 1:10
    pimap{19}(sprintf('cfrontside_%d', ii)) = 8;   % coach front side
end

for ii = 1:10
    pimap{19}(sprintf('cleftside_%d', ii)) = 9;   % coach left side
end

for ii = 1:10
    pimap{19}(sprintf('crightside_%d', ii)) = 10;  % coach right side
end

for ii = 1:10
    pimap{19}(sprintf('cbackside_%d', ii)) = 11;   % coach back side
end

for ii = 1:10
    pimap{19}(sprintf('croofside_%d', ii)) = 12;   % coach roof side
end


% [20tvmonitor]
pimap{20}('screen')     = 1;

