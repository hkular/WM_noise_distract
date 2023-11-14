function WM_noise_uncert_fast()
%% WM_noise_uncert
% addpath(genpath('/Applications/Psychtoolbox'))
% whole report working memory experiment with phase reversing fuzzy orientation and dynamic noise distractors 
% written by HK Sept 2022
%-------------------------------------------------------------------------
clear all;  % clear everything out!
close all;  % close existing figures
warning('off','MATLAB:dispatcher:InexactMatch');  % turn off the case mismatch warning (it's annoying)
dbstop if error  % tell us what the error is if there is one
AssertOpenGL;    % make sure openGL rendering is working (aka psychtoolbox is on the path)
%% 
%--------------------------------------------------------------------------
% Basic settings and dialog box
%--------------------------------------------------------------------------
if ~exist([pwd, filesep,'Data',filesep], 'dir')
        mkdir([pwd, filesep,'Data',filesep]);
end
p.datadir = [pwd,filesep,'Data',filesep];
info.TheDate = datestr(now,'yymmdd'); 

p.subNum = '06'; % keep as as string until after GUI
p.uniqueID = 'DB'; 
p.sessionNum = '2'; % keep as string until after GUI 
p.nruns = '1'; % default 1 run at a time


p.runsave = [p.datadir 'WMuncert_S',num2str(p.subNum),'_',num2str(info.TheDate),'_Sess',p.sessionNum,'_Main.mat'];
if ~exist(p.runsave, 'file') % we're just starting
    p.runNum = 1;
else
    load(p.runsave)
    p.runNum = last_run_num + 1;
end

% Dialog box to get subject info
prompt = {'PID','SubNum','Session','Run', 'nruns'};
defAns = {p.uniqueID, p.subNum, p.sessionNum, num2str(p.runNum), num2str(p.nruns)};

box = inputdlg(prompt,'Enter Subject Information:',1,defAns);
if length(box)==length(defAns)
    p.subName = char(box{1});
    p.subNum = str2double(box{2});
    p.sessionNum = str2double(box{3});
    p.runNum = str2double(box{4});
    p.nruns = str2double(box{5});
else
    error('Oops! Not the right number of inputs to GUI!')
end



% -------------------------------------------------------------------------
% If it's the first run and session, get some additional info.
%--------------------------------------------------------------------------
if p.runNum == 1 && p.sessionNum == 1 && p.subNum~=0
    prompt = {'Age','Gender (M = male, F = female, O = non-binary or other)','Handedness (R = right, L = left, O = other)'};
    defAns = {'','',''};
    box = inputdlg(prompt,'Additional Subject Info',1,defAns);
    if length(box) == length(defAns)                            % check to make sure something was typed in
        info.age = str2num(box{1});
        info.gender = upper(box{2});
        info.handedness = upper(box{3});
    else
        error('Number of inputs provided to GUI does not match!')
        return;                                                 % if nothing was entered or the subject hit cancel, bail out
    end
end

%--------------------------------------------------------------------------
% experiment options
%-------------------------------------------------------------------------
p.environment = 1; % 1 = lab Linux machine, 2 = iMac, 3 = PC
p.MRI = 1; % Are we running in the scanner or not? 
p.portCodes = 0;  %1 = use p.portCodes (we're in the booth)
p.debug = 0; % if MRI/linux, but you want to use the laptop keys, set p.debug to 1 
p.windowed = 0; % 1 = small win for easy debugging!
p.gammacorrect = true;
p.shorttrials = false;
p.room = '0';
%-------------------------------------------------------------------------
% Get the screen settings! 
%-------------------------------------------------------------------------
KbName('UnifyKeyNames');
if p.MRI == 1 % Scanner projector
    p.monWidth = 56.9;   % horizontal dimension of viewable screen (cm)
    p.monHeight = 32;  % vertical distance (cm)
    p.viewDist = 81;   % viewing distance (cm)
    p.start = KbName('t');
    Screen('Preference', 'SkipSyncTests',1);
    p.ccwFast = KbName('b');
    p.ccwSlow = KbName('y');
    p.cwSlow = KbName('g');
    p.cwFast = KbName('r');    
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
elseif p.environment == 1 && ~p.MRI % Behavior rooms
    p.monWidth = 39;
    p.monHeight = 29.5;
    p.viewDist = 52;
    p.start = KbName('space');
    Screen('Preference', 'SkipSyncTests',0);
    p.ccwFast = KbName('LeftArrow');
    p.ccwSlow = KbName('UpArrow');
    p.cwSlow = KbName('DownArrow');
    p.cwFast = KbName('RightArrow');    
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
elseif p.environment == 2  % get realistic size when debugging on Macbook
    p.monWidth = 30.41;
    p.monHeight = 21.24;
    p.viewDist = 52;
    p.start = KbName('space');
    Screen('Preference', 'SkipSyncTests',1);
    Screen('preference','Conservevram', 8192);
    p.ccwFast = KbName('LeftArrow');
    p.ccwSlow = KbName('UpArrow');
    p.cwSlow = KbName('DownArrow');
    p.cwFast = KbName('RightArrow');    
    p.keys = [p.ccwFast, p.ccwSlow, p.cwSlow, p.cwFast];
end
p.escape = KbName('q');

%-------------------------------------------------------------------------
% Make sure support functions and general use scripts on path
%-------------------------------------------------------------------------
p.root = pwd;
addpath([p.root,'/SupportFunctions'])
if p.environment == 1 && p.MRI == 0
    % look for general use scripts in the right place on pclexp server.
    if p.portCodes == 1
        p.GeneralUseScripts ='/home/serencesadmin/Documents/MATLAB/Holly/GeneralUseScripts';
    else
        p.GeneralUseScripts = '/mnt/pclexp/Holly/GeneralUseScripts';
    end
    % Set up data folder in the current folder.
    addpath(p.GeneralUseScripts,[p.GeneralUseScripts,'/Calibration']);
    if ~exist([p.root, filesep,'Data',filesep], 'dir')
        mkdir([p.root, filesep,'Data',filesep]);
    end
    p.datadir = [p.root, filesep,'Data',filesep];
else
    % look for general use script in the current folder
    p.GeneralUseScripts = [pwd,'/GeneralUseScripts'];
    addpath(p.GeneralUseScripts,[p.GeneralUseScripts,'/Calibration']);
end

%-------------------------------------------------------------------------
% If we're on a linux machine, but not in EEG, get the behavior room!
%-------------------------------------------------------------------------
if p.environment==1 && ~p.MRI
    prompt = {'Room Letter'};            % what information do we want from the subject?
    defAns = {''};                                           %s fill in some stock answers - here the fields are left blank
    box = inputdlg(prompt,'Enter Room Info',1,defAns);       % build the GUI
    if length(box) == length(defAns)                            % check to make sure something was typed in
        p.room = upper(box{1});
    else
        return;                                                 % if nothing was entered or the subject hit cancel, bail out
    end
else
    p.room = 'A';
end
%-------------------------------------------------------------------------
% Change to our desired resolution and refresh rate
%-------------------------------------------------------------------------
if p.environment == 1 && ~p.MRI
    s = setScreen_Default(); % just use Default for this experiment! 1600 x 1200, 85 Hz
    if s == 0
        fprintf('Screen successfully set to Experiment Mode!');
    end
end
%% Preferences
    expdir = pwd;
    % set the random seed
    rng('default')
    rng('shuffle')
    % Save the random seed settings!!
    t.rng_settings = rng;
    t.MySeed = t.rng_settings.Seed;

    % get time info
    info.TimeStamp = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)

%% Screen parameters
    Screens = Screen('Screens'); %look at available screens
    %ScreenNr = Screens(end); %pick screen with largest screen number
    ScreenNr = 0;
    p.ScreenSizePixels = Screen('Rect', ScreenNr);
    tmprect = get(0, 'ScreenSize');
    computer_res = tmprect(3:4);
    if computer_res(1) ~= p.ScreenSizePixels(3) || computer_res(2) ~= p.ScreenSizePixels(4)
        Screen('CloseAll');clear screen;ShowCursor;
        disp('*** ATTENTION *** screensizes do not match''')
    end
    CenterX = p.ScreenSizePixels(3)/2;
    CenterY = p.ScreenSizePixels(4)/2;
    [width, height] = Screen('DisplaySize', ScreenNr); % this is in mm
    ScreenHeight = height/10; % in cm, ? cm in the scanner?
    VisAngle = (2*atan2(ScreenHeight/2, p.viewDist))*(180/pi); % visual angle of the whole screen
    p.ppd = p.ScreenSizePixels(4)/VisAngle; % pixels per degree visual angle
    p.fNyquist = 0.5*p.ppd;
    p.hz = Screen('NominalFrameRate', ScreenNr); % get refresh rate
    p.black=BlackIndex(ScreenNr); p.white=WhiteIndex(ScreenNr);
    p.gray=round((p.black+p.black+p.white)/3);
    if round( p.gray)==p.white
         p.gray = p.black;
    end
    
%% Initialize data files and open 
if p.shorttrials
    p.NumTrials = 4;
else
    p.NumTrials = 16; % trials per run
end

%Experimental params required for counterbalancing
p.NumOrientBins = 12; %must be multiple of the size of your orientation space (here: 180)
p.OrientBins = reshape(1:180,180/p.NumOrientBins,p.NumOrientBins);
p.Kappa = [50 5000];
p.Distractor = [0 1];
p.nDistsTrial = 10;
p.nBlocks = 3;

cd(p.datadir); 
    if p.runNum == 1 
        p.TrialNumGlobal = 0; 
        p.StartTrial = p.runNum;
        %----------------------------------------------------------------------
        %COUNTERBALANCING ACT--------------------------------------------------
        %---------------------------------------------------------------------
        TrialStuff = [];
        % make mini design matrix for each set of 3 runs,
        % columns that are fully counterbalanced: [ori distractorlevel kappa]
        designMat = fullfact([12 2 2]); % 12 ori bins and 2 distractor levels and 12 orientation bandwidths
        % shuffle trials
        trial_cnt = 1:length(designMat);
        trial_cnt_shuffled1 = Shuffle(trial_cnt); % first block of runs 1-3
        trial_cnt_shuffled2 = Shuffle(trial_cnt); % second block of runs 4-6
        trial_cnt_shuffled3 = Shuffle(trial_cnt); % third block of runs 7-9
        trial_cnt_shuffled = [trial_cnt_shuffled1, trial_cnt_shuffled2,  trial_cnt_shuffled3];

        for i = 1:length(designMat)*3
            trial.orient = randsample(p.OrientBins(:,(designMat(trial_cnt_shuffled(i),1))),1);% orientation is full counterbalanced
            trial.distractor = p.Distractor(designMat(trial_cnt_shuffled(i),2));
            trial.kappa = p.Kappa(designMat(trial_cnt_shuffled(i),3));
            TrialStuff = [TrialStuff trial];
        end
        p.designMat = designMat;
        p.trial_cnt_shuffled = trial_cnt_shuffled;

    else 
        load(p.runsave);
        p.TrialNumGlobal = TheData(end).p.TrialNumGlobal;
        p.StartTrial = TheData(end).p.TrialNumGlobal+1;
    end
   
    cd(expdir); %Back to experiment dir   
    
%% Main Parameters 

%Timing params -- 
     t.PhaseReverseFreq = 8; %in Hz, how often gratings reverse their phase
     t.PhaseReverseTime = 1/t.PhaseReverseFreq;
     t.TargetTime = 4*t.PhaseReverseTime; % 500 ms multiple of Phase reverse time
     t.TR = 1.25; % ms for a TR
     t.BeginFixation = 3*t.TR; % time to wait for scanner to begin!
     t.EndFixation = 13.8*t.TR; % buffer at end
    
    % Delay period timing
    
     t.DelayTime = 12; %total delay in sec
     t.distDur = t.PhaseReverseTime;
     t.DistractorTime = 11;%    
     t.isi1 = (t.DelayTime-t.DistractorTime)/2; %time between memory stimulus and distractor    
     t.isi2 = t.isi1; %time between distractor and recall probe  
     t.ResponseTime = 3;
     t.possible_iti = [3 5 8]; % for iti jitter

     t.ActiveTrialDur = t.TargetTime+t.isi1+t.DistractorTime+t.isi2+t.ResponseTime; %non-iti portion of trial

     % ITI
     t.iti = NaN(p.NumTrials,1);
     ITIweights = [0.5*p.NumTrials; 0.25*p.NumTrials; 0.25*p.NumTrials];
     ITIunshuffled = repelem(t.possible_iti,ITIweights);
     t.iti = ITIunshuffled(randperm(length(ITIunshuffled)));

    % trial flips    
    t.flipTime(1, p.runNum) = t.ActiveTrialDur + t.BeginFixation;
    for i = 2:p.NumTrials
        t.flipTime(i, p.runNum) = t.flipTime(i-1, p.runNum) + t.ActiveTrialDur;
    end; clear i

    % Check TR's
    t.predictedRunTime = t.BeginFixation + t.EndFixation + sum(t.iti) + ...
        p.NumTrials*(t.TargetTime + t.DelayTime + t.ResponseTime);

    t.nTR_theoretical = 273; % maybe not
    t.nTR_actual = t.predictedRunTime/t.TR;

    t.MeantToBeTime = t.BeginFixation + t.ActiveTrialDur*p.NumTrials + sum(t.iti) + t.EndFixation;

% Vis params
    
    %Stimulus params (general) 
    p.Smooth_size = round(.75*p.ppd); %size of fspecial smoothing kernel
    p.Smooth_sd = round(.4*p.ppd); %smoothing kernel sd
    p.PatchSize = round(2*7*p.ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
    p.OuterDonutRadius = (7*p.ppd)-(p.Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
    p.InnerDonutRadius = (2*p.ppd)+(p.Smooth_size/2); %Size of donut insides, automatically defined in pixels.
    p.OuterFixRadius = round(0.15*p.ppd); %outer dot radius (in pixels)
    p.InnerFixRadius = p.OuterFixRadius/2; %set to zero if you a donut-hater
    p.FixColor = p.black;
    MyPatch = [(CenterX-p.PatchSize/2) (CenterY-p.PatchSize/2) (CenterX+p.PatchSize/2) (CenterY+p.PatchSize/2)];
    

    %Stimulus params (specific) 
    p.SF = 2; % 2spatial frequency in cpd
    p.ContrastTarget = .5; % 
    p.whitenoiseContrast = 1;
    p.Noise_f_bandwidth = 2;%2; % is actually 2 frequency of the noise bandwidth
    p.Noise_fLow = p.SF/p.Noise_f_bandwidth; %Noise low spatial frequency cutoff
    p.Noise_fHigh = p.SF*p.Noise_f_bandwidth; %Noise high spatial frequency cutoff
    
    %Resp Params
    p.ResponseLineWidth = 2; %in pixel
    p.ResponseLineColor = p.white;
   
%-------------------------------------------------------------------------- 

 %% window setup 
 PsychJavaTrouble;
 if p.windowed == 0
     [window, ScreenSize] = Screen('OpenWindow', ScreenNr, p.gray);
 else
     % if we're dubugging open a 640x480 window that is a little bit down from the upper left
     % of the big screen
     [window, ScreenSize]=Screen('OpenWindow', ScreenNr, p.gray, [0 0 1024 768]);
 end
    
    %[window] = Screen('OpenWindow',ScreenNr, p.gray,[],[],2);
    t.ifi = Screen('GetFlipInterval',window);
    t.ExpStartTime = GetSecs;
    
    if p.debug == 0
        HideCursor;
    end
    
Screen('TextSize', window, 25);
DrawFormattedText(window,'Loading...This might take a minute.', 'center', ScreenSize(4)/2.5);
Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
Screen('Flip', window);
    

for b = 1:p.nruns % block loop
%% preallocate in block loop
data = struct();

% behavior
data.Accuracy = NaN(p.NumTrials, 1);
data.Response = NaN(p.NumTrials, 1);
data.RTresp = NaN(p.NumTrials, 1);
data.TestOrient = randsample(1:180,p.NumTrials); 
% preallocate cells so get multiple values per trial
data.trajectory = cell(p.NumTrials, 1);
data.RTresp = cell(p.NumTrials, 1);


% timing
t.TrialStartTime = NaN(p.NumTrials, 1);
t.stimFlips = NaN(p.NumTrials, 2); % stim on/off
t.distFlips = NaN(p.NumTrials,p.nDistsTrial*t.PhaseReverseFreq,2);  % dist on/off
t.respFlips = NaN(p.NumTrials, 2);
t.TrialStory = cell(p.NumTrials*6,2);

%% Make target stimuli

% start with a meshgrid
    X=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5; Y=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5;
    [x,y] = meshgrid(X,Y);
    % make a donut with gaussian blurred edge
    donut_out = x.^2 + y.^2 <= (p.OuterDonutRadius)^2;
    donut_in = x.^2 + y.^2 >= (p.InnerDonutRadius)^2;
    donut = donut_out.*donut_in;
    donut = filter2(fspecial('gaussian', p.Smooth_size, p.Smooth_sd), donut);
    
    % 4D array - target position, x_size, y_size, numtrials
    % initialize with middle grey (background color), then fill in a
    % 1 or 2 as needed for each trial.  
  TargetsAreHere = ones(p.PatchSize,p.PatchSize,2) * p.gray; % last dimension 2 phases
        startTrialThisRun = (p.NumTrials * p.runNum) - p.NumTrials + 1;
        % call function that creates filtered gratings
        [image_final1, image_final2] = FilteredGratingsV3(p.PatchSize, p.SF, p.ppd, p.fNyquist, p.Noise_fLow, p.Noise_fHigh, p.gray, p.whitenoiseContrast, TrialStuff(startTrialThisRun).orient, TrialStuff(startTrialThisRun).kappa);
        %Make it a donut
        stim_phase1 = image_final1.*donut;
        stim_phase2 = image_final2.*donut;
        %Give the grating the right contrast level and scale it
        TargetsAreHere(:,:,1) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase1)));
        TargetsAreHere(:,:,2) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase2)));
          
%% make distractor stimuli
    % make dynamic distractors 
  
    % start with a meshgrid
    X=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5; Y=-0.5*p.PatchSize+.5:1:.5*p.PatchSize-.5;
    [x,y] = meshgrid(X,Y);
    % make a domut with gaussian blurred edge
    donut_out = x.^2 + y.^2 <= (p.OuterDonutRadius)^2;
    donut_in = x.^2 + y.^2 >= (p.InnerDonutRadius)^2;
    donut = donut_out.*donut_in;
    donut = filter2(fspecial('gaussian', p.Smooth_size, p.Smooth_sd), donut);
 % now make a matrix with with all my distractors for all my trials
    DistractorsAreHere = NaN(p.PatchSize,p.PatchSize,p.NumTrials, p.nDistsTrial); % last dimension makes it dynamic
    
    distractor_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
    sine_contrast = std(distractor_sine(:));
    
    for num = 1 : p.nDistsTrial
        %Make uniform noise, put it into fourrier space, make sf filer
        noise = rand(p.PatchSize,p.PatchSize)*2-1;
        fn_noise = fftshift(fft2(noise));
        sfFilter = Bandpass2([p.PatchSize p.PatchSize], p.Noise_fLow/p.fNyquist, p.Noise_fHigh/p.fNyquist);
        %Get rid of gibbs ringing artifacts
        smoothfilter = fspecial('gaussian', 10, 4);   % make small gaussian blob
        sfFilter = filter2(smoothfilter, sfFilter); % convolve smoothing blob w/ s.f. filter
        %Bring noise back into real space
        filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
        %Scale the contrast of the noise back up (it's lost some in the fourier
        %domain) by relating it to the contrast of the grating distractor (before gaussian was applied)
        current_noise_contrast = std(filterednoise(:));
        scaling_factor = sine_contrast/current_noise_contrast;
        filterednoise = filterednoise*scaling_factor;
        %Make it a disc
        filterednoise_phase = filterednoise .* donut;
        %Make sure to scale contrast to where it does not get clipped
        DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(TrialStuff(startTrialThisRun).distractor * filterednoise_phase)));
    end
       
    
    % start event tracker for trial story
    e = 0; 

    %% Gamma Correction
p.OriginalCLUT = Screen('LoadClut', window); OriginalCLUT = p.OriginalCLUT;
if p.gammacorrect
    %OriginalCLUT = Screen('LoadClut', wptr);
    MyCLUT = zeros(256,3); MinLum = 0; MaxLum = 1;
   if strcmp(p.room,'A') % EEG Room
        CalibrationFile = 'LabEEG-05-Jul-2017';
    elseif strcmp(p.room,'B') % Behavior Room B
        CalibrationFile = 'LabB_20-Jul-2022.mat';
    elseif strcmp(p.room,'C') % Behavior Room C
        CalibrationFile = 'LabC-13-Jun-2016.mat';
    elseif strcmp(p.room,'D') % Beahvior room D
        CalibrationFile = 'LabD_20-Jul-2022.mat';
    elseif p.MRI == 1
        CalibrationFile = 'calib_3TE_rearproj_outerbore_24-Feb-2020.mat';
    else
        error('Oops! no calibration file specified!')
    end
    [gamInverse,dacsize] = LoadCalibrationFileRR(CalibrationFile, expdir, p.GeneralUseScripts);
    LumSteps = linspace(MinLum, MaxLum, 256)';
    MyCLUT(:,:) = repmat(LumSteps, [1 3]);
    MyCLUT = round(map2map(MyCLUT, repmat(gamInverse(:,3),[1 3]))); %Now the screen output luminance per pixel is linear!
    Screen('LoadCLUT', window, MyCLUT);
    clear CalibrationFile gamInverse
end
    
%% Welcome and wait for trigger
    %Welcome welcome ya'll

    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    % if in fMRI use this->
    
    % instructions
    if p.MRI == 0
        Screen(window,'TextSize',25);
        Screen('DrawText',window, 'Fixate. Press spacebar to begin.', CenterX-200, CenterY-100, p.black);
    else
        Screen(window,'TextSize',25);
        DrawFormattedText(window,'Remain perfectly still, even when there is nothing on the screen!','center',CenterY + 50, p.white);
        DrawFormattedText(window,'Waiting for trigger (5) from scanner.', 'center',CenterY + 100, p.white);
    end
    Screen('Flip', window);
    FlushEvents('keyDown'); %First discard all characters from the Event Manager queue.
    ListenChar(2);
    % just sittin' here, waitin' on my trigger...
   % Wait for a button press to continue with next block
    while 1
        [keyIsDown,~,keyCode]= KbCheck();
        if keyIsDown
            kp = find(keyCode);
            if kp == p.start
                break;
            end
        end
    end
    FlushEvents('keyDown');
    t.StartTime = GetSecs();
    
    GlobalTimer = 0; %this timer keeps track of all the timing in the experiment. TOTAL timing.
    TimeUpdate = t.StartTime; %what time is it now?
    % present begin fixation
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    %TIMING!:
    GlobalTimer = GlobalTimer + t.BeginFixation;
    TimePassed = 0; %Flush the time the previous event took
    while (TimePassed<t.BeginFixation) %pre TRs...
        TimePassed = (GetSecs-TimeUpdate);%And determine exactly how much time has passed since the start of the expt.
        [resp, ~] = checkForResp(p.start, p.escape);
        if resp == -1; escaperesponse(p.keys, OriginalCLUT); end
        if TimePassed>=(t.BeginFixation)
            Screen('FillOval', window,  p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('Flip', window);
        end
    end
    TimeUpdate = TimeUpdate + t.BeginFixation;
    t.task_start_time = GetSecs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% A TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:p.NumTrials
        t.TrialStartTime(n) = GlobalTimer; %Get the starttime of each single block (relative to experiment start)
        TimeUpdate = t.task_start_time + t.TrialStartTime(n);
        p.TrialNumGlobal = p.TrialNumGlobal+1;
        
        TimeUpdate = GetSecs;
        %% Target rendering

        for revs = 1:t.TargetTime/t.PhaseReverseTime
            StimToDraw = Screen('MakeTexture', window, TargetsAreHere(:,:,rem(revs,2)+1),1);
            Screen('DrawTexture', window, StimToDraw, [], MyPatch, [], 0); 
            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
            %Screen('Close', StimToDraw);
            %TIMING!:
            t.stimFlips(n,revs) = GetSecs;
            GlobalTimer = GlobalTimer + t.PhaseReverseTime;
            ReversalTimePassed = 0; %Flush time passed.
            % Wait the time! 
            while (ReversalTimePassed<t.PhaseReverseTime) %As long as the stimulus is on the screen...
                ReversalTimePassed = (GetSecs-t.stimFlips(n,revs)); %And determine exactly how much time has passed since the start of the expt.
            end
            TimeUpdate = TimeUpdate + t.PhaseReverseTime;
        end
  
        e = e+1;% event counter
        t.TrialStory(e,:) = [{'target'}, num2str(t.TargetTime)];
         %% delay 1
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
         t.stimFlips(n,2) = GetSecs;
        %TIMING!:
        GlobalTimer = GlobalTimer + t.isi1;
        %delay1TimePassed = 0; %Flush time passed.
        delay1TimePassed = (GetSecs-TimeUpdate); 
        while (delay1TimePassed<t.isi1) %As long as the stimulus is on the screen...
            delay1TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            
        end
        TimeUpdate = TimeUpdate + t.isi1; %Update Matlab on what time it is.
        e = e+1;
        t.TrialStory(e,:) = [{'delay 1'} num2str(t.isi1)];

       %% Distractor
     
     for d = 1:p.nDistsTrial
        DistToDraw(d) = Screen('MakeTexture', window, DistractorsAreHere(:,:,d));
     end

       % display loop for dynamic noise
       
        dd=[];
           for k=1:t.PhaseReverseFreq
               dd=[dd;randperm(p.nDistsTrial)']; % cylce through 1-11 8 times no repeats
           end        
           
           for k = 1:(p.nDistsTrial*t.PhaseReverseFreq)

               Screen('DrawTexture', window, DistToDraw(dd(k)), [], MyPatch, [], 0);
       
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                Screen('DrawingFinished', window);
                flipTime = Screen('Flip', window);
                while GetSecs - flipTime < t.distDur; end
                t.distFlips(n,k) = GetSecs - flipTime;
            
           end

        TimeUpdate = TimeUpdate + t.DistractorTime;
        GlobalTimer = GlobalTimer + t.DistractorTime;
        e = e+1;
        t.TrialStory(e,:) = [{'distractor'} num2str(t.DistractorTime)];
        
        Screen('Close', [DistToDraw]);
clear d DistToDraw


%% delay 2
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        %TIMING!:
        GlobalTimer = GlobalTimer + t.isi2;
        %delay2TimePassed = 0; %Flush time passed.
        delay2TimePassed = (GetSecs-TimeUpdate);
        while (delay2TimePassed<t.isi2) %As long as the stimulus is on the screen...
            delay2TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
        end
        TimeUpdate = TimeUpdate + t.isi2; %Update Matlab on what time it is.
        e = e+1;
        t.TrialStory(e,:) = [{'delay 2'} num2str(t.isi2)];
        
        if p.debug % don't print stuff if we're not debugging
            fprintf('\n%d\t%d\t%d\t%d\t%d\t%d%s\n', p.TrialNumGlobal, TrialStuff(p.TrialNumGlobal).orient, TrialStuff(p.TrialNumGlobal).kappa, TrialStuff(p.TrialNumGlobal).distractor, data.TestOrient(n));
        end

%% response window
% get RT
% full report spin a line
        
        test_orient = data.TestOrient(n);
        orient_trajectory = [test_orient];
        InitX = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * cos(test_orient*pi/180)+CenterX)); 
        InitY = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * sin(test_orient*pi/180)-CenterY));
        Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
        Screen('DrawLines', window, [2*CenterX-InitX, InitX; 2*CenterY-InitY, InitY], p.ResponseLineWidth, p.ResponseLineColor,[],1);
        Screen('BlendFunction', window, GL_ONE, GL_ZERO);
        Screen('FillOval', window, p.gray, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window,[],1);
        t.respFlips(n) = GetSecs;
        
        GlobalTimer = GlobalTimer + t.ResponseTime;
        resp_start = GetSecs;
        react = [0];
        RespTimePassed = GetSecs-resp_start; %Flush time passed.
        %RespTimePassed = (GetSecs-TimeUpdate);
        while RespTimePassed<t.ResponseTime  %As long as no correct answer is identified
            RespTimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
         
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            % buttons
            if keyCode(p.ccwFast) %BIG step CCW
                test_orient = rem(test_orient+2+1440,180);
                react = [react (secs - resp_start)];
                % alternate way of getting RT
                % RT(n,tt) = secs-resp_start;
            elseif keyCode(p.ccwSlow) %small step CCW
                test_orient = rem(test_orient+.5+1440,180);
                react = [react (secs - resp_start)];
            elseif keyCode(p.cwSlow) %small step CW
                test_orient = rem(test_orient-.5+1440,180);
                react = [react (secs - resp_start)];
            elseif keyCode(p.cwFast) %BIG step CW
                test_orient = rem(test_orient-2+1440,180);
                react = [react (secs - resp_start)];
            elseif keyCode(KbName('ESCAPE')) % If user presses ESCAPE, exit the program.
                save(p.runsave,'p','info','t','data', 'TrialStuff');
                Screen('CloseAll');
                ListenChar(1);
                if exist('OriginalCLUT','var')
                    if exist('ScreenNr','var')
                        Screen('LoadCLUT', ScreenNr, OriginalCLUT);
                    else
                        Screen('LoadCLUT', 0, OriginalCLUT);
                    end
                end
                error('User exited program.');
            end
                test_orient(test_orient==0)=180;
                orient_trajectory = [orient_trajectory test_orient]; % how to preallocate this one?
                UpdatedX = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * cos(test_orient*pi/180)+CenterX));
                UpdatedY = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * sin(test_orient*pi/180)-CenterY));
                Screen('BlendFunction', window, GL_ONE, GL_ZERO);
                Screen('FillRect', window, p.gray);
                Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
                Screen('DrawLines', window, [2*CenterX-UpdatedX, UpdatedX; 2*CenterY-UpdatedY, UpdatedY], p.ResponseLineWidth, p.ResponseLineColor, [], 1);
                Screen('BlendFunction', window, GL_ONE, GL_ZERO);
                Screen('FillOval', window, p.gray, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);        
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius]);
                Screen('Flip', window, [], 1,[], []);
        end
        FlushEvents('keyDown'); %First discard all characters from the Event Manager queue
        data.Response(n) = test_orient;
        data.RTresp{n} = react; % problem recording all the reacts for velocity
        data.trajectory{n} = orient_trajectory; % problem here
          
        % change to make if no keys pressed NaN
        if data.Response(n) == data.TestOrient(n)
           data.Response(n) = NaN;
        end
          
        
        TimeUpdate = TimeUpdate + t.ResponseTime; %Update Matlab on what time it is.
        e = e+1;
        t.TrialStory(e,:) = [{'response'} num2str(t.ResponseTime)];


        
        %% iti
    
        Screen('FillRect',window,p.gray);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
      
         % Make things during ITI
        
        % TARGET for next trial
        % 4D array - target position, x_size, y_size, numtrials
        % initialize with middle grey (background color), then fill in a
        % phase 1 or 2 as needed for each trial.
        TargetsAreHere = ones(p.PatchSize,p.PatchSize,2) * p.gray; % last dimension 2 phases 
        % call function that creates filtered gratings
        [image_final1, image_final2] = FilteredGratingsV3(p.PatchSize, p.SF, p.ppd, p.fNyquist, p.Noise_fLow, p.Noise_fHigh, p.gray, p.whitenoiseContrast, TrialStuff(n+1).orient, TrialStuff(n+1).kappa);
        %Make it a donut
        stim_phase1 = image_final1.*donut;
        stim_phase2 = image_final2.*donut;
        %Give the grating the right contrast level and scale it
        TargetsAreHere(:,:,1) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase1)));
        TargetsAreHere(:,:,2) = max(0,min(255,p.gray+p.gray*(p.ContrastTarget * stim_phase2)));
        
        % DISTRACTOR for next trial
        DistractorsAreHere = NaN(p.PatchSize,p.PatchSize, p.nDistsTrial); % last dimension makes it dynamic
        distractor_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
        sine_contrast = std(distractor_sine(:));
          for num = 1 : p.nDistsTrial
                %Make uniform noise, put it into fourrier space, make sf filer
                noise = rand(p.PatchSize,p.PatchSize)*2-1;
                fn_noise = fftshift(fft2(noise));
                sfFilter = Bandpass2([p.PatchSize p.PatchSize], p.Noise_fLow/p.fNyquist, p.Noise_fHigh/p.fNyquist);
                %Get rid of gibbs ringing artifacts
                smoothfilter = fspecial('gaussian', 10, 4);   % make small gaussian blob
                sfFilter = filter2(smoothfilter, sfFilter); % convolve smoothing blob w/ s.f. filter
                %Bring noise back into real space
                filterednoise = real(ifft2(ifftshift(sfFilter.*fn_noise)));
                %Scale the contrast of the noise back up (it's lost some in the fourier
                %domain) by relating it to the contrast of the grating distractor (before gaussian was applied)
                current_noise_contrast = std(filterednoise(:));
                scaling_factor = sine_contrast/current_noise_contrast;
                filterednoise = filterednoise*scaling_factor;
                %Make it a disc
                filterednoise_phase = filterednoise .* donut;              
                %Make sure to scale contrast to where it does not get clipped
                DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(TrialStuff(n+1).distractor * filterednoise_phase)));
          end
      
        %TIMING!:
        
        GlobalTimer = GlobalTimer + t.iti(n);
        %TimePassed = 0; %Flush time passed.
        TimePassed = (GetSecs-TimeUpdate);
        while (TimePassed<t.iti(n)) %As long as the stimulus is on the screen...
            TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
             
            if TimePassed>=(t.iti(n))
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                Screen('Flip', window);
            end
        end
        TimeUpdate = TimeUpdate + t.iti(n); %Update Matlab on what time it is.
        e = e+1;
        t.TrialStory(e,:) = [{'iti'} num2str(t.iti(n))];
        
        % Here, use the "untilTime" function relative to the ENTIRE experiment to sop up
        % any timing sloppiness during each individual trial... 
        WaitSecs('UntilTime',t.task_start_time + t.flipTime(n)); % Wait until the correct start time to flip this!! 
           
        
end %end of experimental trial loop
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% END OF TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    % final fixation:
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    
    GlobalTimer = GlobalTimer + t.EndFixation;
    %closingtime = 0; 
    resp = 0;
    closingtime = GetSecs - TimeUpdate;
    while closingtime < t.EndFixation
        closingtime = GetSecs-TimeUpdate;
        ListenChar(1); %Unsuppressed keyboard mode
        if CharAvail
            [press] = GetChar;
            if strcmp(press,'1')
                resp = str2double(press);
            end
        end
    end
    
    
    t.EndTime = GetSecs; %Get endtime of the experiment in seconds
    t.TotalExpTime = (t.EndTime-t.StartTime); %Gets the duration of the total run.
    t.TotalExpTimeMins = t.TotalExpTime/60; %TOTAL exp time in mins including begin and end fixation.
    t.GlobalTimer = GlobalTimer; %Spits out the exp time in secs excluding begin and end fixation.
    
    
    
    %----------------------------------------------------------------------
    %LOOK AT BEHAVIORAL PERFOPRMANCE---------------------------------------
    %----------------------------------------------------------------------
    targets_were = [TrialStuff(p.TrialNumGlobal+1-p.NumTrials:p.TrialNumGlobal).orient]';
    acc(1,:) = abs(targets_were-data.Response);
    acc(2,:) = abs((360-(acc(1,:)*2))/2); 
    acc(3,:) = 360-(acc(1,:));
    acc = min(acc);
    acc = acc';
     
    
    %Add minus signs back in
    acc(mod(targets_were-acc,360)==data.Response)=-acc(mod(targets_were-acc,360)==data.Response);
    acc(mod((targets_were+180)-acc,360)==data.Response)=-acc(mod((targets_were+180)-acc,360)==data.Response);
    data.Accuracy = acc;
%     figure;histogram(data.Accuracy,-90:1:90); set(gca,'XLim',[-90 90],'XTick',[-90:45:90]);
%     title(['Mean accuracy was ' num2str(mean(abs(data.Accuracy))) ' degrees'],'FontSize',16)
%     
    
    %----------------------------------------------------------------------
    %SAVE OUT THE DATA-----------------------------------------------------
    %----------------------------------------------------------------------
    cd(p.datadir); %Change the working directory back to the experimental directory
    if exist(['WM_noise_uncert_S',num2str(p.subNum),'_',num2str(info.TheDate),'_Sess',p.sessionNum,'_Main.mat'])
        load(['WM_noise_uncert_S',num2str(p.subNum),'_',num2str(info.TheDate),'_Sess',p.sessionNum,'_Main.mat']);
    end
    %First I make a list of variables to save:
    TheData(p.runNum).info = info;
    TheData(p.runNum).t = t;
    TheData(p.runNum).p = p;
    TheData(p.runNum).data = data;
    last_run_num = p.runNum;
    save(p.runsave, 'TheData', 'TrialStuff', 'last_run_num')
    cd(expdir)
    
    p.runNum = p.runNum+1;
    clear acc
end  % end of block loop 
    %----------------------------------------------------------------------
    %WINDOW CLEANUP--------------------------------------------------------
    %----------------------------------------------------------------------
    %This closes all visible and invisible screens and puts the mouse cursor
    %back on the screen
    Screen('CloseAll');
   % load('OriginalCLUT_labC.mat');
    if exist('OriginalCLUT','var')
        if exist('ScreenNr','var')
            Screen('LoadCLUT', ScreenNr, OriginalCLUT);
        else
            Screen('LoadCLUT', 0, OriginalCLUT);
        end
    end
    clear screen
    ListenChar(1);
    ShowCursor;

    
end
       

