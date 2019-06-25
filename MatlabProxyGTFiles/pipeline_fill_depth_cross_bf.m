% in-painting function fill_depth_cross_bf.m
% from https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

% needed to call this command before the mex crap would work
mex mex_cbf_windows.cpp cbf_windows.cpp

% change this to the directory of the dataset you need to process
ROOT = 'G:/depth_vids_d435/train';

% change this to the number of scenes in your dataset
for folderInd = 0:2445
    % modify pow and/or samples to work with the number of frames you have
    % used to process your proxy ground truth images
    for pow = 0:7
        samples = power(2, pow);
        
        rgb_filename = sprintf('%s/depths_vid_%04d/med_image_%03d.png', ROOT, folderInd, samples);
        dep_filename = sprintf('%s/depths_vid_%04d/med_depth_%03d.png', ROOT, folderInd, samples);
        
        img_rgb = imread(rgb_filename);
        img_dep = (double(imread(dep_filename)) / 255.) * 10.;

        img_dep_filled = fill_depth_cross_bf(img_rgb, img_dep);
        img_dep_filled = medfilt2(img_dep_filled, [5,5]);
        
        sprintf("Saving %s", dep_filename)
        imwrite(img_dep_filled / 10., dep_filename);
        
        figure(1);
        subplot(1, 1, 1); imagesc(img_dep_filled);
    end
end