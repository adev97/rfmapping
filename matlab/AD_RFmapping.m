function AD_RFmapping(mode)
% Generate regular ON and OFF RF maps from the exported session 4 inputs.
% Run AD_RFmapping from MATLAB after installing the matching core update.
% Use AD_RFmapping('detect') to run Python on already saved RF maps.

    if nargin == 0
        mode = 'all';
    end
    mode = validatestring(mode, {'all', 'detect'});

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

    % Run Python automatically using this project's environment.
    params.runRfDetection = true;
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

    if params.runRfDetection || strcmp(mode, 'detect')
        assert(isfile(params.rfPythonExecutable), ...
            'Missing Python executable: %s. Run uv sync --extra analysis in the repository.', ...
            params.rfPythonExecutable);
        assert(isfile(params.rfPythonScript), ...
            'Missing RF detection script: %s', params.rfPythonScript);
        fprintf('Python executable:\n%s\n', params.rfPythonExecutable);
    end
    if strcmp(mode, 'detect')
        runSavedRfDetection(params, is_on, is_off);
        fprintf('Session 4 Python RF detection complete.\n');
        return;
    end

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
    % Build locally, then copy the MEX binary into this project. Keeping the
    % compiler/linker files local avoids build-file access failures on R:.
    intervalDir = fullfile(fmatDir, 'General');
    intervalMex = fullfile(intervalDir, ['FindInInterval.', mexext]);
    if ~isfile(intervalMex)
        assert(~isempty(mex.getCompilerConfigurations('C', 'Selected')), ...
            ['FindInInterval needs a MEX binary. Run mex -setup C in MATLAB ' ...
             'to select a supported compiler, then rerun AD_RFmapping.']);
        previousDir = pwd;
        buildDir = tempname;
        mkdir(buildDir);
        buildCleanup = onCleanup(@() cleanupMexBuild(previousDir, buildDir));
        copyfile(fullfile(intervalDir, 'FindInInterval.c'), ...
            fullfile(buildDir, 'FindInInterval.c'));
        cd(buildDir);
        fprintf('Building FindInInterval locally:\n%s\n', buildDir);
        mex('-outdir', buildDir, 'FindInInterval.c');
        copyfile(fullfile(buildDir, ['FindInInterval.', mexext]), ...
            intervalMex, 'f');
        clear buildCleanup;  % Restore the working folder and remove build files.
    end
    assert(isfile(intervalMex), 'Missing compiled MEX file: %s', intervalMex);
    addpath(intervalDir, '-begin');
    clear FindInInterval;
    rehash path;
    resolvedHelper = which('FindInInterval');
    assert(exist('FindInInterval', 'file') == 3, ...
        ['FindInInterval MEX is at:\n%s\nMATLAB resolves the name to:\n%s\n' ...
         'Run which FindInInterval -all to check for another copy.'], ...
        intervalMex, resolvedHelper);
    helperIndices = feval('FindInInterval', [0; 1; 2], [0.5 1.5]);
    assert(isequal(helperIndices, [2; 2]), ...
        'FindInInterval did not return the expected indices in its startup check.');
    fprintf('Using FindInInterval MEX:\n%s\n', resolvedHelper);

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

function runSavedRfDetection(params, is_on, is_off)
    % These filenames match the core's regular exported-session output layout.
    assert(isscalar(params.sessionList) && ...
        ~any([params.isVerticalBar, params.isBackgroundMoving, ...
              params.isRotation, params.isAllocentricPixelBins]), ...
        'Detection-only mode requires one regular square-mapping session.');
    timeBinWidthMs = diff(params.VSTimeWindow) * 1000 / params.nbins;
    timeFolder = sprintf('%g_%g_%gms', params.VSTimeWindow(1) * 1000, ...
        params.VSTimeWindow(2) * 1000, timeBinWidthMs);
    if params.onlyReadGoodUnits
        unitSelectionFolder = 'good';
    else
        unitSelectionFolder = 'all';
    end
    suffixes = {};
    if is_on
        suffixes{end + 1} = '';
    end
    if is_off
        suffixes{end + 1} = '_off';
    end
    jobs = cell(0, 2);
    for index = 1:numel(suffixes)
        suffix = suffixes{index};
        for probe = params.probelist
            rfmapPath = fullfile(params.sessionDir, 'data', ...
                ['rfmapping', suffix], unitSelectionFolder, timeFolder, ...
                ['Probe', probe], sprintf('regular_unitsSpikeCounts_%s_%d%s.rfmap', ...
                params.date, params.sessionList, suffix));
            assert(isfile(rfmapPath), ...
                'Missing saved RF map: %s. Generate the maps with AD_RFmapping first.', ...
                rfmapPath);
            jobs(end + 1, :) = {rfmapPath, probe}; %#ok<AGROW>
        end
    end
    for index = 1:size(jobs, 1)
        RFmapping_run_python(jobs{index, 1}, jobs{index, 2}, params);
    end
end

function cleanupMexBuild(previousDir, buildDir)
    % Restore the working folder on success or error before deleting the build.
    cd(previousDir);
    if isfolder(buildDir)
        rmdir(buildDir, 's');
    end
end
