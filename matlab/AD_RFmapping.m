function AD_RFmapping
% Generate regular ON and OFF RF maps from the exported session 4 inputs.
% Run AD_RFmapping from MATLAB after installing the matching core update.

    matlabDir = fileparts(mfilename('fullpath'));
    codeDir = fileparts(matlabDir);
    fmatDir = fullfile(matlabDir, 'buzcode-master', ...
        'externalPackages', 'FMAToolbox');
    addpath(matlabDir);
    addpath(fullfile(matlabDir, 'Utils'));
    addpath(genpath(fmatDir));

    % Your recording paths.
    data_PATH = 'R:\Basic_Sciences\Phys\SenzaiLab\Aparna\Data_by_mouse';
    mouseID = 'm002';
    pipeline_output_PATH = 'small-spread-neuropixel-config';
    date_PATH = '2026-09-25';
    pipeline_session_output_PATH = 'ad_session_2026_09_25';
    session_number = 4;  % Folder number; the split metadata's session_id is 3.

    params.sessionDir = fullfile(data_PATH, mouseID, pipeline_output_PATH, ...
        date_PATH, pipeline_session_output_PATH, 'sessions', ...
        sprintf('session_%02d', session_number));
    params.trialsMatFilename = ...
        'AD_headfixed_rfmapping_stationary_2026-9-25_RF_Mapping_Trials.mat';

    % Identity used in result names and optional comparison exports.
    params.base_dir = [fullfile(data_PATH, mouseID), filesep];
    params.date = '260925';
    params.sessionList = session_number;
    params.probelist = 'A';
    params.onlyReadGoodUnits = true;

    is_on = true;
    is_off = true;

    % Regular square-stimulus mapping uses the logged screen coordinates.
    params.isBackgroundMoving = false;
    params.isAllocentricPixelBins = false;
    params.isRotation = false;
    params.rotationOffsetSign = +1;
    params.isVerticalBar = false;
    params.barBinWidthDeg = 3;
    params.isFineResolution = false;
    params.isUseRealCoordinate = true;
    params.total_deg = 360;
    params.screenWidthPix = 960;
    params.screenHeightPix = 240;
    params.screenDeg = 360;

    % Same response window and bin width as the existing RFmapping wrapper.
    params.VSTimeWindow = [-0.1 0.4];  % Seconds relative to each measured onset.
    timeBinWidthMs = 1;
    nbinsExact = diff(params.VSTimeWindow) * 1000 / timeBinWidthMs;
    assert(abs(nbinsExact - round(nbinsExact)) < 1e-9, ...
        'The response window must contain an integer number of time bins.');
    params.nbins = round(nbinsExact);

    % Native Windows: run locate_rf.py separately after MATLAB finishes.
    params.runRfDetection = false;
    if ispc
        params.rfPythonExecutable = fullfile(codeDir, '.venv', 'Scripts', 'python.exe');
    else
        params.rfPythonExecutable = fullfile(codeDir, '.venv', 'bin', 'python');
    end
    params.rfPythonScript = fullfile(codeDir, 'locate_rf.py');
    params.rfTimeRange = [0 0.2];
    params.maxZeroBins = 2;
    params.clusterFormingZ2d = 1.8;
    params.clusterFormingZ1d = 1;
    params.dropBins = 2;
    params.rfWrapX = true;
    params.rfCollapseFrom2d = false;

    % Check the intended inputs before starting the analysis.
    requiredFiles = { ...
        fullfile(params.sessionDir, 'stimulus', params.trialsMatFilename), ...
        fullfile(params.sessionDir, 'stimulus', 'on_list_times.npy')};
    for probe = params.probelist
        neuralDir = fullfile(params.sessionDir, 'neural', ['Probe', probe]);
        requiredFiles = [requiredFiles, { ...
            fullfile(neuralDir, 'adc_spike_times.npy'), ...
            fullfile(neuralDir, 'spike_clusters.npy'), ...
            fullfile(neuralDir, 'cluster_KSLabel.tsv')}]; %#ok<AGROW>
    end
    for index = 1:numel(requiredFiles)
        assert(isfile(requiredFiles{index}), ...
            'Missing session 4 RF input: %s', requiredFiles{index});
    end

    % FindInInterval.m documents the helper; Sync requires its MEX binary.
    % Build the supplied C source once if MATLAB has a selected C compiler.
    if exist('FindInInterval', 'file') ~= 3
        assert(~isempty(mex.getCompilerConfigurations('C', 'Selected')), ...
            ['FindInInterval needs a MEX binary. Run mex -setup C in MATLAB ' ...
             'to select a supported compiler, then rerun AD_RFmapping.']);
        intervalDir = fullfile(fmatDir, 'General');
        fprintf('Building FindInInterval for this MATLAB installation...\n');
        mex('-outdir', intervalDir, fullfile(intervalDir, 'FindInInterval.c'));
        rehash;
        assert(exist('FindInInterval', 'file') == 3, ...
            'The compiled FindInInterval MEX binary is not on the MATLAB path.');
    end

    fprintf('Session 4 input directory:\n%s\n', params.sessionDir);
    fprintf('Response window: %g to %g ms; %g ms/bin; %d bins.\n', ...
        params.VSTimeWindow(1) * 1000, params.VSTimeWindow(2) * 1000, ...
        timeBinWidthMs, params.nbins);
    if is_on
        fprintf('----- Working on ON RF -----\n');
        params.lum = 1;
        RFmapping_core(params);
    end
    if is_off
        fprintf('----- Working on OFF RF -----\n');
        params.lum = 0;
        RFmapping_core(params);
    end
    fprintf('Session 4 RF generation complete.\n');
end
