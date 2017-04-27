function setupDataOptsPrts(vocYear, testName, datasetDir)

global DATAopts;

% Setup VOC data
devkitroot = [datasetDir, 'VOCdevkit', '/'];
DATAopts.year = vocYear;
DATAopts.dataset = sprintf('VOC%d', DATAopts.year);
DATAopts.datadir        = [devkitroot, DATAopts.dataset, '/'];
DATAopts.resdir         = [devkitroot, 'results', '/', DATAopts.dataset '/'];
DATAopts.localdir       = [devkitroot, 'local', '/', DATAopts.dataset, '/'];
DATAopts.imdb           = [DATAopts.datadir, '/imdb-%s.mat'];
DATAopts.gStructPath    = [DATAopts.resdir, 'GStructs', '/'];
DATAopts.imgsetpath     = [DATAopts.datadir, 'ImageSets', '/', 'Main', '/', '%s.txt'];
DATAopts.imgpath        = [DATAopts.datadir, 'JPEGImages', '/', '%s.jpg'];
DATAopts.clsimgsetpath  = [DATAopts.datadir, 'ImageSets', '/', 'Main', '/', '%s_%s.txt'];
DATAopts.annopath_obj       = [DATAopts.datadir, 'Annotations', '/', '%s.xml'];
% Keep object annopath as standard for ap evaluation
DATAopts.annopath       = DATAopts.annopath_obj;
DATAopts.annopath_prt       = [DATAopts.datadir, 'Annotations_Part', '/', '%s.mat'];
DATAopts.annocachepath	= [DATAopts.localdir, '%s_anno.mat'];
DATAopts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};
DATAopts.nclasses = length(DATAopts.classes);
DATAopts.testset = testName;
DATAopts.minoverlap = 0.5;