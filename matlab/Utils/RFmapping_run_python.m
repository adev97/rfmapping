function RFmapping_run_python(rfmapPath, probe, params)
    commandArgs = { ...
        char(params.rfPythonExecutable), '-u', ...
        char(params.rfPythonScript), char(rfmapPath), ...
        '--probe', char(probe), ...
        '--time-range', sprintf('%.17g', params.rfTimeRange(1)), ...
        sprintf('%.17g', params.rfTimeRange(2)), ...
        '--max-zero-bins', sprintf('%d', params.maxZeroBins), ...
        '--cluster-forming-z-2d', sprintf('%.17g', params.clusterFormingZ2d), ...
        '--cluster-forming-z-1d', sprintf('%.17g', params.clusterFormingZ1d), ...
        '--drop-bins', sprintf('%d', params.dropBins)};
    if params.rfCollapseFrom2d
        commandArgs{end + 1} = '--collapse-from-2d';
    end
    if ~params.rfWrapX
        commandArgs{end + 1} = '--no-wrap-x';
    end

    [rfDirectory, rfName] = fileparts(rfmapPath);
    % Paired notebooks use the regular ON source; other maps need distinct CSV targets.
    if ~isempty(regexp(rfName, ['^regular_unitsSpikeCounts_', params.date, '_[0-9]+$'], 'once'))
        [~, mouse] = fileparts(regexprep(params.base_dir, '[\\/]+$', ''));
        dataDirectory = rfDirectory;
        for level = 1:4
            dataDirectory = fileparts(dataDirectory);
        end
        commandArgs = [commandArgs, { ...
            '--unit-prefix', [mouse, ':', params.date, ':', probe], ...
            '--comparison-output-dir', fullfile(dataDirectory, 'tc_comparison')}];
    end

    if ispc
        % Encode the PowerShell call so cmd never interprets the path strings.
        % Single-quoted PowerShell arguments preserve spaces and apostrophes.
        quotedArgs = cellfun(@PowerShellQuote, commandArgs, 'UniformOutput', false);
        script = ['& ', strjoin(quotedArgs, ' '), ...
            ' 2>&1 | ForEach-Object { $_.ToString() }; ', ...
            'if ($null -eq $LASTEXITCODE) { exit 1 }; exit $LASTEXITCODE'];
        encoded = char(matlab.net.base64encode(unicode2native(script, 'UTF-16LE')));
        command = ['powershell.exe -NoProfile -NonInteractive -EncodedCommand ', encoded];
    else
        quotedArgs = cellfun(@ShellQuote, commandArgs, 'UniformOutput', false);
        command = strjoin(quotedArgs, ' ');
    end

    fprintf('Running Python RF detection:\n%s\nRF source:\n%s\n', ...
        params.rfPythonExecutable, rfmapPath);
    [status, output] = system([command, ' 2>&1'], '-echo');
    if status ~= 0
        error('RFmapping:PythonDetectionFailed', ...
            'RF detection failed for %s (exit %d):\n%s', rfmapPath, status, output);
    end
    fprintf('Python RF detection complete for %s.\n', rfmapPath);
end

function quoted = PowerShellQuote(value)
    quote = char(39);
    quoted = [quote, strrep(char(value), quote, [quote, quote]), quote];
end

function quoted = ShellQuote(value)
    % POSIX single quotes preserve paths containing spaces, quotes, and shell syntax.
    quote = char(39);
    quoted = [quote, strrep(char(value), quote, [quote, '"', quote, '"', quote]), quote];
end
