function points= vl_sift_pourfard(I)
I=single(I);
fc = [100;100;10;0] ;
[f,d] = vl_sift(I,'frames',fc,'orientations');
% [f,d] = vl_sift(I,'frames',fc,'orientations',2,'WindowSize',3,'Levels',...
%     0,'FirstOctave',0,'PeakThresh',10,'EdgeThresh',...
%     -inf,'NormThresh',3,'Magnif',2,'WindowSize');

%   VL_SIFT() accepts the following options:
%
%   Octaves:: maximum possible
%     Set the number of octave of the DoG scale space.
%
%   Levels:: 3
%     Set the number of levels per octave of the DoG scale space.
%
%   FirstOctave:: 0
%     Set the index of the first octave of the DoG scale space.
%
%   PeakThresh:: 0
%     Set the peak selection threshold.
%
%   EdgeThresh:: 10
%     Set the non-edge selection threshold.
%
%   NormThresh:: -inf
%     Set the minimum l2-norm of the descriptors before
%     normalization. Descriptors below the threshold are set to zero.
%
%   Magnif:: 3
%     Set the descriptor magnification factor. The scale of the
%     keypoint is multiplied by this factor to obtain the width (in
%     pixels) of the spatial bins. For instance, if there are there
%     are 4 spatial bins along each spatial direction, the
%     ``side'' of the descriptor is approximatively 4 * MAGNIF.
%
%   WindowSize:: 2
%     Set the variance of the Gaussian window that determines the
%     descriptor support. It is expressend in units of spatial
%     bins.
%
%   Frames::
%     If specified, set the frames to use (bypass the detector). If
%     frames are not passed in order of increasing scale, they are
%     re-orderded.
%
%   Orientations::
%     If specified, compute the orientations of the frames overriding
%     the orientation specified by the 'Frames' option.
%
%   Verbose::
%     If specfified, be verbose (may be repeated to increase the
%     verbosity level).