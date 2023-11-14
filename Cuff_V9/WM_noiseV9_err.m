%% %%   HK distractor and uncertainty 4/2023
% addpath(genpath('/Applications/Psychtoolbox'))
% Inputs
% p parameter
% info subject info
% nruns: number of runs to execute sequentially 
% startRun: run number to start with if interrupted (default is 1)

% Stimulus categories
% Target: gabor orientation w/noise - set size + uncertainty 3 levels
% structured noise
% distractor: dynamic noise. arrythmic 25Hz

 % Experimental design
 % Run duration: 64 mins
 % Block duration: 6.4 mins
 % Task: orientation full report
%% 
function WM_noiseV9_err(p, nruns, startRun, TrialStuff)

 %% Prepare and collect basic info
    
    %Set Paths
    expdir = pwd;
    datadir = 'Data';
    addpath(pwd);
 
    % set the random seed
    rng('default')
    rng('shuffle')
    t.MySeed = rng; % Save the random seed settings!!
    
    % get time info
    info.TheDate = datestr(now,'yymmdd'); %Collect todays date (in t.)
    info.TimeStamp = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)
 
 
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
    %% Screen parameters 
    %ScreenNr = Screens(end); %pick screen with largest screen number
    ScreenNr = 0; % set to smallest when working with dual monitor setup to have display on laptop
    p.ScreenSizePixels = Screen('Rect', ScreenNr);
    tmprect = get(0, 'ScreenSize');
    computer_res = tmprect(3:4);
    if computer_res(1) ~= p.ScreenSizePixels(3) || computer_res(2) ~= p.ScreenSizePixels(4)
        Screen('CloseAll');clear screen;ShowCursor;
        disp('*** ATTENTION *** screensizes do not match''')
    end
    if p.windowed == 0
        CenterX = p.ScreenSizePixels(3)/2;
        CenterY = p.ScreenSizePixels(4)/2;
    else
        CenterX = 1024/2;
        CenterY = 768/2;
    end % if windowed
    [width, height] = Screen('DisplaySize', ScreenNr); % this is in mm
    ScreenHeight = height/10; % in cm, ? cm in the scanner?
    ViewDistance = 57; % (57 cm is the ideal distance where 1 cm equals 1 visual degree)
    VisAngle = (2*atan2(ScreenHeight/2, ViewDistance))*(180/pi); % visual angle of the whole screen
    p.ppd = p.ScreenSizePixels(4)/VisAngle; % pixels per degree visual angle
    p.black=BlackIndex(ScreenNr); p.white=WhiteIndex(ScreenNr);
    if p.gammacorrect; p.gray=round((p.black+p.black+p.black+p.white)/4); else; p.gray=round((p.black+p.white)/2);end % darker if gammacorrect
    if round( p.gray)==p.white
         p.gray = p.black;
    end
    p.fNyquist = 0.5*p.ppd;
%% Initialize data files and open 
cd(datadir); 
    if exist(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
        load(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
        p.runNum = length(TheData) + 1; % set the number of the current run
        p.startRun = startRun;
        p.nruns = nruns;
        p.TrialNumGlobal = TheData(end).p.TrialNumGlobal;
        p.NumTrials = TheData(end).p.NumTrials;
        p.NumOrientBins = TheData(end).p.NumOrientBins;
        p.OrientBins = TheData(end).p.OrientBins;
        p.Distractor = TheData(end).p.Distractor;
        p.Kappa = TheData(end).p.Kappa;
        p.StartTrial = TheData(end).p.TrialNumGlobal+1;
        p.Block = TheData(end).p.Block+1;
        p.designMat = TheData(end).p.designMat;
        p.trial_cnt_shuffled = TheData(end).p.trial_cnt_shuffled;
         p.stairstep = TheData(2).p.stairstep;

    else
        p.runNum = startRun; %If no data file exists this must be the first run
        p.Block = p.runNum;
        p.TrialNumGlobal = 0; 
        p.nruns = nruns;
        p.stairstep = .7; % size of % increment/decrement in contrast for distractor task
        %Experimental params required for counterbalancing
        p.NumOrientBins = 2; %must be multiple of the size of your orientation space (here: 180) 
        %p.OrientBins = reshape(1:180,180/p.NumOrientBins,p.NumOrientBins);
        p.OrientBins = reshape([(1:1:45), (136:1:180), (46:1:90), (91:1:135)],[], p.NumOrientBins);  % bin 1 is horiz, bin 2 is vertical       
        p.Kappa = [50 100 5000];
        p.Distractor = [0 .65]; % absent present half contrast !!! change back
        p.DistractorTask = [0 1]; % !!! change back
        %----------------------------------------------------------------------
        %COUNTERBALANCING ACT--------------------------------------------------
        %---------------------------------------------------------------------
        if exist('TrialStuff', 'var')
            p.NumTrials = 36;
        else
            TrialStuff = [];
            % make mini design matrix for each set of 3 runs,
            % columns that are fully counterbalanced: [ori distractorlevel kappa]
            designMat = fullfact([2 3 2 2]); % 2 ori bins and 2 distractor levels and 3 orientation bandwidths 2 distractor task
            designMat = repmat(designMat, 15,1);
            % shuffle trials
            trial_cnt = 1:length(designMat);
            trial_cnt_shuffled = Shuffle(trial_cnt);
            
            for i = 1:length(designMat)
                trial.orient = randsample(p.OrientBins(:,(designMat(trial_cnt_shuffled(i),1))),1);% orientation is full counterbalanced
                trial.kappa = p.Kappa(designMat(trial_cnt_shuffled(i),2));
                trial.distractor = p.Distractor(designMat(trial_cnt_shuffled(i),3));
                trial.distractortask = p.DistractorTask(designMat(trial_cnt_shuffled(i),4));
                TrialStuff = [TrialStuff trial];
            end
            p.designMat = designMat;
            p.trial_cnt_shuffled = trial_cnt_shuffled;
            if p.shortTrial == 0
                p.NumTrials = 36;% %NOT TRIVIAL! --> must be divisible by MinTrialNum AND by the number of possible iti's (which is 3)
                %currently MinNumTrials is 360, meaning 10 blocks of 36 trials
            else
                p.NumTrials = 12;
            end
        end
    end
   
    cd(expdir); %Back to experiment dir
%% Main Parameters 

%Timing params -- 
    t.PhaseReverseFreq = 8; %in Hz, how often gratings reverse their phase
    t.PhaseReverseTime = 1/t.PhaseReverseFreq;
    t.TargetTime = 4*t.PhaseReverseTime; % 500 ms multiple of Phase reverse time
    
    p.nDistsTrial = 11; % number of different ones to make
    t.DistractorTime = 3.5;% actual distractor time 
    t.DistFreq = 25; % how fast distractors flip...make it faster 25Hz
    t.DistFlipTime = 1/t.DistFreq;
    p.nDistFrames = round(t.DistractorTime * t.DistFreq); % how many frames are we flipping through
    p.nchangeframes = 3; % change three frames in a row otherwise 25Hz is too fast to see
    % !!! change more frames in a row
    % pre-randomize frames
    t.DistArray = [];
    for i = 1:(p.nDistFrames/p.nDistsTrial)
        t.DistArray = [t.DistArray;randperm(p.nDistsTrial)']; % randomize frames no repeats 
    end
    t.DelayTime = 3.5; %total delay in sec
    t.detect = 1; % how long to wait for distractor change detection
    t.isi1 = 0; %time between memory stimulus and distractor - 0
    t.isi2 = t.isi1; %time between distractor and recall probe - 0
    t.ResponseTime = 3;
    t.ActiveTrialDur = t.TargetTime+t.isi1+t.DistractorTime+t.isi2+t.ResponseTime; %non-iti portion of trial
    t.possible_iti = [2 4 6]; 
    t.iti = NaN(p.NumTrials,1);
    ITIweights = [0.5*p.NumTrials; 0.25*p.NumTrials; 0.25*p.NumTrials];
    ITIunshuffled = repelem(t.possible_iti,ITIweights);
    t.iti = ITIunshuffled(randperm(length(ITIunshuffled)));
    t.CueStartsBefore = 1; %starts 1 second before the stimulus comes on 
    t.BeginFixation = 3; %16 TRs need to be extra (16trs * .8ms)
    t.EndFixation = 3;
    t.MeantToBeTime = t.BeginFixation + t.ActiveTrialDur*p.NumTrials + sum(t.iti) + t.EndFixation;
    
     % trial flips    
    t.flipTime(1, p.runNum) = t.ActiveTrialDur + t.BeginFixation;
    for i = 2:p.NumTrials
        t.flipTime(i, p.runNum) = t.flipTime(i-1, p.runNum) + t.ActiveTrialDur;
    end; clear i
    
    %Stimulus params (general) 
    p.Smooth_size = round(.75*p.ppd); %size of fspecial smoothing kernel
    p.Smooth_sd = round(.4*p.ppd); %smoothing kernel sd
    p.PatchSize = round(2*7*p.ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
    p.OuterDonutRadius = (7*p.ppd)-(p.Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
    p.InnerDonutRadius = (2*p.ppd)+(p.Smooth_size/2); %Size of donut insides, automatically defined in pixels.
    p.OuterFixRadius = .2*p.ppd; %outter dot radius (in pixels)
    p.InnerFixRadius = p.OuterFixRadius/2; %set to zero if you a donut-hater
    p.FixColor = p.black;
    p.ResponseLineWidth = 2; %in pixel
    p.ResponseLineColor =p.white;
    MyPatch = [(CenterX-p.PatchSize/2) (CenterY-p.PatchSize/2) (CenterX+p.PatchSize/2) (CenterY+p.PatchSize/2)];

    %Stimulus params (specific) 
    p.SF = 2; %spatial frequency in cpd 
    p.ContrastTarget = .5; % 
    p.whitenoiseContrast = .5; % 
    p.changecontrast = p.stairstep*p.whitenoiseContrast;
    p.Noise_f_bandwidth = 2;% frequency of the noise bandwidth
    p.Noise_fLow = p.SF/p.Noise_f_bandwidth; %Noise low spatial frequency cutoff
    p.Noise_fHigh = p.SF*p.Noise_f_bandwidth; %Noise high spatial frequency cutoff

%% window setup and gamma correction
 % clock
    PsychJavaTrouble;
    if p.windowed == 0
        [window, ScreenSize] = Screen('OpenWindow', ScreenNr, p.gray);
    else
        % if we're dubugging open a 640x480 window that is a little bit down from the upper left
        % of the big screen
        [window, ScreenSize]=Screen('OpenWindow', ScreenNr, p.gray, [0 0 1024 768]);
    end
    t.ifi = Screen('GetFlipInterval',window);
    if p.gammacorrect % 
        OriginalCLUT = Screen('LoadClut', window);
        MyCLUT = zeros(256,3); MinLum = 0; MaxLum = 1;
        if strcmp(p.room,'A') % EEG Room
            CalibrationFile = 'LabEEG-05-Jul-2017';
        elseif strcmp(p.room,'B') % Behavior Room B
            CalibrationFile = 'LabB_20-Jul-2022.mat';
        elseif strcmp(p.room,'C') % Behavior Room C !!! check room C
            CalibrationFile = 'LabC-13-Jun-2016.mat';
        elseif strcmp(p.room,'D') % Beahvior room D
            CalibrationFile = 'LabD_20-Jul-2022.mat';
        else
            error('No calibration file specified')
        end
        [gamInverse,dacsize] = LoadCalibrationFileRR(CalibrationFile, expdir, p.GeneralUseScripts);
        LumSteps = linspace(MinLum, MaxLum, 256)';
        MyCLUT(:,:) = repmat(LumSteps, [1 3]);
        MyCLUT = map2map(MyCLUT, repmat(gamInverse(:,3),[1 3])); %Now the screen output luminance per pixel is linear!
        Screen('LoadCLUT', window, MyCLUT);
        clear CalibrationFile gamInverse
    end
    
    if p.debug == 0
        HideCursor;
    end

for b = startRun:nruns % block loop
 %% preallocate in block loop
data = struct();

% behavior
data.Accuracy = NaN(p.NumTrials, 1);
data.Response = NaN(p.NumTrials, 1);
data.RTresp = NaN(p.NumTrials, 1);
data.TestOrient = randsample(1:180,p.NumTrials);
data.DistChange = NaN(p.NumTrials,1);
data.DistResp = NaN(p.NumTrials,1);
data.ChangeFrame = NaN(p.NumTrials,p.nchangeframes);
% preallocate cells so get multiple values per trial
data.trajectory = cell(p.NumTrials, 1);

% timing
t.TrialStartTime = NaN(p.NumTrials, 1);
t.stimFlips = NaN(p.NumTrials, 2);
t.distFlips = NaN(p.NumTrials,p.nDistsTrial,1); 
t.respFlips = NaN(p.NumTrials, 1);

%% Make target stimuli
% start with a meshgrid
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
    DistractorsAreHere = NaN(p.PatchSize,p.PatchSize, p.nDistsTrial); % last dimension makes it dynamic 
    distractor_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
    sine_contrast = std(distractor_sine(:));
    DistractorChangeHere = NaN(p.PatchSize,p.PatchSize, p.nchangeframes); % change 3 frames
     % figure out which frame to change the contrast
    if TrialStuff(startTrialThisRun).distractortask == 1        
        whichframes = (round(p.nDistFrames/3):2*round(p.nDistFrames/3));
        data.ChangeFrame(startTrialThisRun,2) = randsample(whichframes,1); % randomly select which frames in the middle second of the task to change
        data.ChangeFrame(startTrialThisRun,1) = data.ChangeFrame(startTrialThisRun,2)-1;data.ChangeFrame(startTrialThisRun,p.nchangeframes) = data.ChangeFrame(startTrialThisRun,2)+1;
        delta = randsample([-1,1],1);    % randomly select increment or decrement   
        data.DistChange(startTrialThisRun) = TrialStuff(startTrialThisRun).distractor + (TrialStuff(startTrialThisRun).distractor*p.stairstep*delta);%save the change    
    end
   
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
        % change contrast here
        %Make it a disc
        filterednoise_phase = filterednoise .* donut;
        %Make sure to scale contrast to where it does not get clipped
        DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(TrialStuff(startTrialThisRun).distractor * filterednoise_phase)));
        if TrialStuff(startTrialThisRun).distractortask == 1
            for i = 1: p.nchangeframes
                if num == t.DistArray(data.ChangeFrame(startTrialThisRun,i))
                    DistractorChangeHere(:,:,i) = max(0,min(255,p.gray+p.gray*(data.DistChange(startTrialThisRun) * filterednoise_phase))); %
                end %if1
            end% for i
        end %if2
    end %for
    


%% Welcome and wait for trigger
    %Welcome welcome ya'll

    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen(window,'TextSize',30);
    Screen('DrawText',window, 'Fixate. Press spacebar to begin.', CenterX-200, CenterY-100,p.black); % change location potentially
    Screen('Flip', window);
    FlushEvents('keyDown'); %First discard all characters from the Event Manager queue.
    ListenChar(2);
    % just sittin' here, waitin' on my trigger...
    while 1
        [keyIsDown, secs, keyCode] = KbCheck([-1]); % KbCheck([-1])
        if keyCode(KbName('space'))
            t.StartTime = GetSecs;
            break; %let's go!
        end
    end
    FlushEvents('keyDown');
    
    GlobalTimer = 0; %this timer keeps track of all the timing in the experiment. TOTAL timing.
    TimeUpdate = t.StartTime; %what time is it now?
    % present begin fixation
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    %TIMING!:
    GlobalTimer = GlobalTimer + t.BeginFixation;
    TimePassed = 0; %Flush the time the previous event took
    while (TimePassed<t.BeginFixation) %For as long as the cues are on the screen...
        TimePassed = (GetSecs-TimeUpdate);%And determine exactly how much time has passed since the start of the expt.
        if TimePassed>=(t.BeginFixation-t.CueStartsBefore)
            Screen('FillOval', window,  p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('Flip', window);
        end
    end
    TimeUpdate = TimeUpdate + t.BeginFixation;
    %t.task_start_time = GetSecs;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% A TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:p.NumTrials
        t.TrialStartTime(n) = GlobalTimer; %Get the starttime of each single block (relative to experiment start)
        %TimeUpdate = t.task_start_time + t.TrialStartTime(n);
        TimeUpdate = t.StartTime + t.TrialStartTime(n);
        p.TrialNumGlobal = p.TrialNumGlobal+1;
        
        %TimeUpdate = GetSecs; % added this 4/19/23
        %% Target rendering

        for revs = 1:t.TargetTime/t.PhaseReverseTime
            StimToDraw = Screen('MakeTexture', window, TargetsAreHere(:,:,rem(revs,2)+1));
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
                %ReversalTimePassed = (GetSecs-t.stimFlips(n,revs)); %And determine exactly how much time has passed since the start of the expt.
                ReversalTimePassed = (GetSecs-TimeUpdate);
            end
            TimeUpdate = TimeUpdate + t.PhaseReverseTime;
        end
        %.03 off
         %% delay 1
%         Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
%         Screen('DrawingFinished', window);
%         Screen('Flip', window);
%         %TIMING!:
%         GlobalTimer = GlobalTimer + t.isi1;
%         delay1TimePassed = (GetSecs-TimeUpdate); 
%         while (delay1TimePassed<t.isi1) %As long as the stimulus is on the screen...
%             delay1TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
%         end
%         TimeUpdate = TimeUpdate + t.isi1; %Update Matlab on what time it is.
%        
       %% Distractor  

       for d = 1:p.nDistsTrial
           DistToDraw(d) = Screen('MakeTexture', window, DistractorsAreHere(:,:,d));
       end %for
       if TrialStuff(n).distractortask == 1 && TrialStuff(n).distractor > 0
           for i = 1: p.nchangeframes
               ChangeToDraw(i) = Screen('MakeTexture', window, DistractorChangeHere(:,:, i));
           end %for i
           change = true;
       else
           change = false;
       end %if
       react = NaN;
       dist_start = GetSecs;
       for k = 1:round(t.DistractorTime * t.DistFreq)
           if change && sum(ismember(data.ChangeFrame(n,:),k))>0
               i = find(data.ChangeFrame(n,:) == k);
               Screen('DrawTexture', window, ChangeToDraw(i), [], MyPatch, [],0); %filtermode then alpha 
           else
               Screen('DrawTexture', window, DistToDraw(t.DistArray(k)), [], MyPatch, [],0);
           end %endif
           Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
           Screen('DrawingFinished', window);
           Screen('Flip', window);
           t.DistFlips(n,k) = GetSecs;
           %GlobalTimer = GlobalTimer + t.DistFlipTime;
           [keyIsDown, secs, keyCode] = KbCheck(-1);
           react = secs - dist_start;
            FlipTimePassed = 0; %Flush time passed.
           % Wait the time!
           while (FlipTimePassed<t.DistFlipTime) %As long as the stimulus is on the screen...
               FlipTimePassed = (GetSecs-t.DistFlips(n,k)); %And determine exactly how much time has passed since the start of the expt.
               %FlipTimePassed = (GetSecs-TimeUpdate);
           end%while
           %TimeUpdate = TimeUpdate + t.DistFlipTime;
       end%for
       data.DistResp(n) = react;
       Screen('Close', [DistToDraw]);
       clear d DistToDraw ChangeToDraw change

 %.02 off
%% delay 2
%         Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
%         Screen('DrawingFinished', window);
%         Screen('Flip', window);
%         %TIMING!:
%         GlobalTimer = GlobalTimer + t.isi2;
%         %delay2TimePassed = 0; %Flush time passed.
%         delay2TimePassed = (GetSecs-TimeUpdate);
%         while (delay2TimePassed<t.isi2) %As long as the stimulus is on the screen...
%             delay2TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
%         end
%         TimeUpdate = TimeUpdate + t.isi2; %Update Matlab on what time it is.
% 
%       
%         if p.debug % don't print stuff if we're not debugging
%             fprintf('\n%d\t%d\t%d\t%d\t%d\t%d%s\n', n, TrialStuff(n).orient, TrialStuff(n).kappa, TrialStuff(n).distractor, data.TestOrient(n));
%         end

%% response window
% get RT
% full report spin a line, in quadrant we are probing

        resp_start = GetSecs;        
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
        react = 0;
        RespTimePassed = GetSecs-resp_start; %Flush time passed.
        %RespTimePassed = 0;% Flush time passed
        %RespTimePassed = (GetSecs-TimeUpdate);
        while RespTimePassed<t.ResponseTime  %As long as no correct answer is identified
            RespTimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.         
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            % buttons
            if keyCode(p.ccwFast) %BIG step CCW
                test_orient = rem(test_orient+2+1440,180);
                react = secs - resp_start;
                % alternate way of getting RT
                % RT(n,tt) = secs-resp_start;
            elseif keyCode(p.ccwSlow) %small step CCW
                test_orient = rem(test_orient+.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(p.cwSlow) %small step CW
                test_orient = rem(test_orient-.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(p.cwFast) %BIG step CW
                test_orient = rem(test_orient-2+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('ESCAPE')) % If user presses ESCAPE, exit the program.
%                 cd(datadir); %Change the working directory back to the experimental directory
%                 if exist(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
%                     load(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
%                 end
%                 %First I make a list of variables to save:
%                 TheData(p.runNum).info = info;
%                 TheData(p.runNum).t = t;
%                 TheData(p.runNum).p = p;
%                 TheData(p.runNum).data = data;
%                 eval(['save(''WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'', ''TheData'', ''TrialStuff'', ''-V7.3'')']);
%                 cd(expdir)
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
                orient_trajectory = [orient_trajectory test_orient]; 
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
        data.RTresp(n) = react;
        data.trajectory{n} = orient_trajectory; % 
          
        % change to make if no keys pressed NaN
        if data.Response(n) == data.TestOrient(n)
           data.Response(n) = NaN;
        end                 
        TimeUpdate = TimeUpdate + t.ResponseTime; %Update Matlab on what time it is.

        %% iti
        if p.debug
            TrialStuff(n).distractor
            TrialStuff(n).distractortask
            n
        end
        Screen('FillRect',window,p.gray);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        
         % Make things during ITI must be less than <2sec shortest iti

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
         % figure out which frame to change the contrast

    DistractorsAreHere = NaN(p.PatchSize,p.PatchSize, p.nDistsTrial); % last dimension makes it dynamic 
    distractor_sine = (sin(p.SF/p.ppd*2*pi*(y.*sin(0*pi/180)+x.*cos(0*pi/180))));
    sine_contrast = std(distractor_sine(:));
    DistractorChangeHere = NaN(p.PatchSize,p.PatchSize, p.nchangeframes); % change 3 frames
     % figure out which frame to change the contrast
    if TrialStuff(n+1).distractortask == 1        
        whichframes = (round(p.nDistFrames/3):2*round(p.nDistFrames/3));
        data.ChangeFrame(n+1,2) = randsample(whichframes,1); % randomly select which frames in the middle second of the task to change
        data.ChangeFrame(n+1,1) = data.ChangeFrame(n+1,2)-1;
        data.ChangeFrame(n+1,p.nchangeframes) = data.ChangeFrame(n+1,2)+1;
        delta = randsample([-1,1],1);    % randomly select increment or decrement   
        data.DistChange(n+1) = TrialStuff(n+1).distractor + (TrialStuff(n+1).distractor*p.stairstep*delta);%save the change    
    end
   
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
        % change contrast here
        %Make it a disc
        filterednoise_phase = filterednoise .* donut;
        %Make sure to scale contrast to where it does not get clipped
        DistractorsAreHere(:,:,num) = max(0,min(255,p.gray+p.gray*(TrialStuff(n+1).distractor * filterednoise_phase)));
        if TrialStuff(n+1).distractortask == 1
            for i = 1: p.nchangeframes
                if num == t.DistArray(data.ChangeFrame(n+1,i))
                    DistractorChangeHere(:,:,i) = max(0,min(255,p.gray+p.gray*(data.DistChange(n+1) * filterednoise_phase))); %
                end %if1
            end% for i
        end %if2
    end %for
        %TIMING!:
        
         GlobalTimer = GlobalTimer + t.iti(n);
        TimePassed = 0; %Flush time passed.
        %TimePassed = (GetSecs-TimeUpdate);
        while (TimePassed<t.iti(n)) %As long as the stimulus is on the screen...
            TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
             
            if TimePassed>=(t.iti(n))
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                Screen('Flip', window);
            end
        end
        TimeUpdate = TimeUpdate + t.iti(n);  %Update Matlab on what time it is.
       
         %WaitSecs('UntilTime',t.task_start_time + t.flipTime(n)); % Wait until the correct start time to flip this!! 

end %end of experimental trial/block loop
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%% END OF TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
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
    %figure;hist(data.Accuracy,-90:1:90); set(gca,'XLim',[-90 90],'XTick',[-90:45:90]);
    %title(['Mean accuracy was ' num2str(mean(abs(data.Accuracy))) ' degrees'],'FontSize',16)
    
    % get average of trial accuracy if worse than 45 average, tell to stop
    performance = nanmean(abs(data.Accuracy));
    % count non-responses to perform attention check
    % check = sum(isnan(data.Response));
    % change feedback depending on performance and attention check
%     if check<6 && performance<45 
         blockStr = ['Finished block ' num2str(p.runNum) ' out of ' num2str(nruns)];
         feedbackStr = [blockStr sprintf('\n') 'Press the spacebar to continue'];
%     elseif check>6
%          blockStr = ['Finished block ' num2str(p.runNum) ' out of ' num2str(nruns)];
%         feedbackStr = [blockStr sprintf('\n') 'STOP and check with experimenter before continuing'];
%     elseif performance>45
%         blockStr = ['Finished block ' num2str(p.runNum) ' out of ' num2str(nruns)];
%         feedbackStr = [blockStr sprintf('\n') 'STOP and check with experimenter before continuing'];   
%     end
%     
    % Look at distractor detection: staircase change if not 73-77%%
    ind = find([TrialStuff(p.TrialNumGlobal+1-p.NumTrials:p.TrialNumGlobal).distractortask]' == 1); % get which trials should've had a detection
    p.check = (sum(~isnan(data.DistResp(ind))))/(numel(ind))*100; % check what percent did we catch
    if p.check > 80
        % if they are too good then make it harder change by 10% less
        if p.stairstep >.1
            p.stairstep = p.stairstep - .1; % !!! no idea if .1 is right
        end
    elseif p.check < 70
        % if they are not good then make it easier by changing 10% more
        if p.stairstep <1
            p.stairstep = p.stairstep + .1;
        end
    end
    %----------------------------------------------------------------------
    %SAVE OUT THE DATA-----------------------------------------------------
    %----------------------------------------------------------------------
%     cd(datadir); %Change the working directory back to the experimental directory
%     if exist(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
%         load(['WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
%     end
%     %First I make a list of variables to save:
%     TheData(p.runNum).info = info;
%     TheData(p.runNum).t = t;
%     TheData(p.runNum).p = p;
%     TheData(p.runNum).data = data;
%     eval(['save(''WM_noiseV9_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'', ''TheData'', ''TrialStuff'', ''-V7.3'')']);
%     cd(expdir)
%     
    FlushEvents('keyDown'); 
    
    clear TargetsAreHere DistractorsAreHere
    
    % final fixation and feedback:
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    % may need to change spacing
    DrawFormattedText(window,[feedbackStr],CenterX-200,CenterY,p.white);
    Screen('Flip',window);
    
    while 1
            [keyIsDown, secs, keyCode] = KbCheck([-1]); % KbCheck([-1])
            if keyCode(KbName('space'))
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                Screen('Flip', window);
                break; %next block
            end
        end
        FlushEvents('keyDown');
    
    
     GlobalTimer = GlobalTimer + t.EndFixation;
     
    closingtime = 0; resp = 0;
    
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
    
    p.runNum = p.runNum+1;
 
    clear acc
end  % end of block loop
    %----------------------------------------------------------------------
    %WINDOW CLEANUP--------------------------------------------------------
    %----------------------------------------------------------------------
    %This closes all visible and invisible screens and puts the mouse cursor
    %back on the screen
    Screen('CloseAll');
    %load('OriginalCLUT_labC.mat');
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
