function [regions_new cc_new]= vl_mser_pourfard(I,MinDiversity,MaxVariation,Delta,BrightOnDark,DarkOnBright,MaxArea,MinArea)
[r,f] = vl_mser(I,'MinDiversity',MinDiversity,'MaxVariation',MaxVariation,...
    'Delta',Delta,'BrightOnDark',BrightOnDark,'DarkOnBright',DarkOnBright,...
    'MaxArea',MaxArea,'MinArea',MinArea/numel(I))

% compute regions mask
M = zeros(size(I)) ;
for x=r'
    s = vl_erfill(I,x) ;
    M(s) = M(s) + 1;
end
%Why M increase?????
% adjust convention
f = vl_ertr(f) ;


% unique(M)
% imshow((M/max(M(:))));
% contour(M);
uu=0;
ww=size(unique(M),1);
for k=1:ww
    [L,n]= bwlabel(M==k);%default connectivity is 8
%     regionsCell=cell(n,1);
        for i=1:n
        uu=uu+1;
        [r, c] = find(L==i);
        % rc = [r c]
        regionsCell{uu,1}=int32([c, r]);
    end
end
regions_new = MSERRegions(regionsCell);
cc_new = region2cc(regions_new, size(I));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%==========================================================================
% Convert MSER regions to connected component struct
%==========================================================================
function cc = region2cc(regions, imageSize)

if isempty(coder.target)
    % Convert PixelList to PixelIdxList and pack into connected component struct.
    
    pixelIdxList = cell(1, regions.Count);
    for i = 1:regions.Count
        locations = regions.PixelList{i};
        pixelIdxList{i} = sub2ind(imageSize, locations(:,2), locations(:,1));
        %         i
    end
    
    cc.Connectivity = 8;
    cc.ImageSize    = imageSize;
    cc.NumObjects   = regions.Count;
    cc.PixelIdxList = pixelIdxList;
    
else
    % Code generation path
    idxCount = coder.internal.indexInt([0;cumsum(regions.Lengths)]);
    regionIndices = coder.nullcopy(zeros(sum(regions.Lengths),1));
    
    % MSER regions are stored as x,y locations. convert them to
    % linear indices for return in the CC struct.
    for k = 1:regions.Count
        
        range = idxCount(k)+1:idxCount(k+1);
        
        locations = regions.PixelList(range, :);
        
        idx = sub2ind(imageSize, locations(:,2), locations(:,1));
        
        regionIndices(range,1) = idx;
    end
    
    cc.Connectivity  = 8;
    cc.ImageSize     = imageSize;
    cc.NumObjects    = regions.Count;
    cc.RegionIndices = regionIndices;
    cc.RegionLengths = cast(regions.Lengths, 'like', idxCount);
    
end
