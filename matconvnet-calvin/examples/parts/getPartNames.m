function [allPartNames, idxPartGlobal2idxClass]  = getPartNames(imdb)
    idxPartGlobal2idxClass = zeros(imdb.prt_num_classes+1,1);
    allPartNames = cell(imdb.prt_num_classes+1,1);
    numPartsCls = cellfun(@(x) size(x,1), imdb.prt_classes);

    k = 1;
    % Create map from idxPartGlobal to the class
    for idxClass = 1:imdb.obj_num_classes
       idxPartGlobal2idxClass(k+1:k+numPartsCls(idxClass)) =  ones(numPartsCls(idxClass),1)*idxClass;
       allPartNames(k+1:k+numPartsCls(idxClass)) = imdb.prt_classes{idxClass};
       k = k + numPartsCls(idxClass);
    end
    allPartNames = allPartNames(2:end);
end