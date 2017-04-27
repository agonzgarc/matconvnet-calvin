function pimap = part2names_4imdb()

% Removed parts  that are too tiny or rare to be considered
%
% Created by Davide Modolo, 2016

pimap = cell(20, 1);                    
for ii = 1:20
    pimap{ii} = containers.Map('KeyType','char','ValueType','int32');
end

% [1aeroplane]
pimap{1}('body')        = 1;                
pimap{1}('stern')       = 2; 
pimap{1}('wing')        = 3;        
pimap{1}('engine')      = 4;

% [2bicycle]
pimap{2}('wheel')       = 1;               
pimap{2}('saddle')      = 2;               
pimap{2}('handlebar')   = 3;              
pimap{2}('chainwheel')  = 4;                    

% [3bird] 
pimap{3}('head')        = 1;
pimap{3}('beak')        = 2;               
pimap{3}('torso')       = 3;            
pimap{3}('neck')        = 4;
pimap{3}('wing')        = 5;            
pimap{3}('leg')         = 6;           
pimap{3}('foot')        = 7;       
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
pimap{6}('mirror')      = 6;
pimap{6}('liplate')     = 7;
pimap{6}('door')        = 8;
pimap{6}('wheel')       = 9;
pimap{6}('window')      = 10;

% [7car]
pimap{7}('frontside')   = 1;
pimap{7}('leftside')    = 2;
pimap{7}('rightside')   = 3;
pimap{7}('backside')    = 4;
pimap{7}('roofside')    = 5;
% pimap{7}('mirror')      = 6;
pimap{7}('liplate')     = 6;
pimap{7}('door')        = 7;
pimap{7}('wheel')       = 8;
pimap{7}('headlight')   = 9;
pimap{7}('window')      = 10;

% [8cat]
pimap{8}('head')        = 1;
pimap{8}('eye')         = 2;             
pimap{8}('ear')         = 3;     
pimap{8}('nose')        = 4;
pimap{8}('torso')       = 5;   
pimap{8}('neck')        = 6;
pimap{8}('leg')         = 7;     
pimap{8}('paw')         = 8;           
pimap{8}('tail')        = 9;               

% [9chair]
% only has sihouette mask 

% [10cow]
pimap{10}('head')       = 1;
% pimap{10}('eye')        = 2;           
pimap{10}('ear')        = 2;            
pimap{10}('muzzle')     = 3;
pimap{10}('horn')       = 4;                
pimap{10}('torso')      = 5;            
pimap{10}('neck')       = 6;
pimap{10}('leg')        = 7;             
pimap{10}('tail')       = 8;               

% [11diningtable]
% only has silhouette mask 

% [12dog]
pimap{12}('head')        = 1;
pimap{12}('ear')         = 2;
pimap{12}('nose')        = 3;
pimap{12}('torso')       = 4;   
pimap{12}('neck')        = 5;
pimap{12}('leg')         = 6;
pimap{12}('paw')         = 7;               
pimap{12}('tail')        = 8;   
pimap{12}('muzzle')      = 9;


% [13horse]
pimap{13}('head')       = 1;
% pimap{13}('eye')        = 2;
pimap{13}('ear')        = 2;
pimap{13}('muzzle')     = 3;
pimap{13}('torso')      = 4;            
pimap{13}('neck')       = 5;
pimap{13}('leg')        = 6;
% pimap{13}('hoof')       = 7; 
pimap{13}('tail')       = 7; 

% [14motorbike]
pimap{14}('wheel')      = 1;
pimap{14}('handlebar')  = 2;
pimap{14}('headlight')  = 3;

% [15person]
pimap{15}('head')       = 1;                  
pimap{15}('hair')       = 2;                   
pimap{15}('torso')      = 3;                   
pimap{15}('neck')       = 4;  
pimap{15}('arm')        = 5;
pimap{15}('hand')       = 6;
pimap{15}('leg')        = 7;
pimap{15}('foot')       = 8;

% [16pottedplant]
pimap{16}('pot')        = 1;
pimap{16}('plant')      = 2;

% [17sheep]
pimap{17}('head')       = 1;
pimap{17}('ear')        = 2; 
pimap{17}('muzzle')     = 3;
pimap{17}('horn')       = 4;

pimap{17}('torso')      = 5;            
pimap{17}('neck')       = 6;
pimap{17}('leg')        = 7; 
pimap{17}('tail')       = 8;               

% [18sofa]
% only has sihouette mask 

% [19train]
pimap{19}('head')       = 1;
pimap{19}('hfrontside') = 2;                	               
pimap{19}('hleftside')  = 3;                	
pimap{19}('hrightside') = 4;                	               	
pimap{19}('hroofside')  = 5;   
pimap{19}('headlight')  = 6;
pimap{19}('coach')      = 7;
pimap{19}('cfrontside') = 8;   % coach front side
pimap{19}('cleftside')  = 9;
pimap{19}('crightside') = 10;
pimap{19}('cbackside')  = 11;
pimap{19}('croofside')  = 12;

% [20tvmonitor]
pimap{20}('screen')     = 1;