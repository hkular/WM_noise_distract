%% %%  from RR WM_DistractV5_Main_Practice.m adapted by HK, 2021-2022, for HK Sherlock
% addpath(genpath('/Applications/Psychtoolbox'))
% Inputs
% nruns: number of runs to execute sequentially 
% startRun: run number to start with if interrupted (default is 1)

% Stimulus categories
% Target: gabor orientation w/noise - set size + uncertainty with Maggie's
% structured noise
% distractor: dynamic noise - 3 contrast levels 0, .15, 1
 
 % Experimental design
 % Run duration: 67 mins
 % Block duration: 5.5 mins
 % Task: orientation full report
%% 
function WM_noiseV5practice(p, info, nruns, startRun)

 %% Prepare and collect basic info
    
    %Set Paths
    expdir = pwd;
    datadir = 'Data/practice';
    addpath(pwd);
 
    % set the random seed
    rng('default')
    rng('shuffle')
    t.MySeed = rng; % Save the random seed settings!!
    
    % get time info
    info.TheDate = datestr(now,'yymmdd'); %Collect todays date (in t.)
    info.TimeStamp = datestr(now,'HHMM'); %Timestamp for saving out a uniquely named datafile (so you will never accidentally overwrite stuff)
    
 
 %% Screen parameters !! 
 Screen('Preference', 'SkipSyncTests', 0);% 1 to run on mac
  %Screen('preference','Conservevram', 8192);
 %Screens = Screen('Screens'); %look at available screens
    %ScreenNr = Screens(end); %pick screen with largest screen number
    ScreenNr = 0; % set to smallest when working with dual monitor setup to have display on laptop
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
    ViewDistance = 57; % in cm, ? cm in the scanner!!! (57 cm is the ideal distance where 1 cm equals 1 visual degree)
    VisAngle = (2*atan2(ScreenHeight/2, ViewDistance))*(180/pi); % visual angle of the whole screen
    p.ppd = p.ScreenSizePixels(4)/VisAngle; % pixels per degree visual angle
    p.MyGrey = 132;% ask RR why this grey
    p.fNyquist = 0.5*p.ppd;
    black=BlackIndex(ScreenNr); white=WhiteIndex(ScreenNr);
    gammacorrect = true;
 
%% Initialize data files and open 
cd(datadir); 
    if exist(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
        load(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
        runnumber = length(TheData) + 1; % set the number of the current run
        p.startRun = startRun;
        p.nruns = nruns;
        p.TrialNumGlobal = TheData(end).p.TrialNumGlobal;
        p.NumTrials = TheData(end).p.NumTrials;
        p.NumOrientBins = TheData(end).p.NumOrientBins;
        p.OrientBins = TheData(end).p.OrientBins;
        p.Kappa = TheData(end).p.Kappa;
        p.StartTrial = TheData(end).p.TrialNumGlobal+1;
        p.Block = TheData(end).p.Block+1;
        p.designMat = TheData(end).p.designMat;
        p.trial_cnt_shuffled = TheData(end).p.trial_cnt_shuffled;
    else
        runnumber = 1; %If no data file exists this must be the first run
        p.Block = runnumber;
        p.TrialNumGlobal = 0;
        p.startRun = startRun; 
        p.nruns = nruns;
        %Experimental params required for counterbalancing
        p.NumOrientBins = 2; %must be multiple of the size of your orientation space (here: 180) 
        p.OrientBins = reshape(1:180,180/p.NumOrientBins,p.NumOrientBins);
        p.Kappa = [5000 500 50];
        [TrialStuff, designMat, trial_cnt_shuffled, MinNumTrials] = CounterBalancingAct_npracticeV5(p.OrientBins, p.Kappa);
        p.designMat = designMat;
        p.trial_cnt_shuffled = trial_cnt_shuffled;
        p.NumTrials = 12;%30; %NOT TRIVIAL!!! --> must be divisible by MinTrialNum AND by the number of possible iti's (which is 3)
        %currently MinNumTrials is 360, meaning 12 blocks of 30 trials
        
    end
   
    cd(expdir); %Back to experiment dir
%% Main Parameters 

%Timing params -- 
    t.TargetTime = 0.5;
    t.FeedbackTime = 0.2;
    t.DelayTime = 0.05; %total delay in sec
    t.isi1 = t.DelayTime; %time between memory stimulus and distractor
    t.isi2 = 0; %time between distractor and recall probe
    t.ResponseTime = 3;
    t.ActiveTrialDur = t.TargetTime+t.isi1+t.FeedbackTime+t.isi2+t.ResponseTime; %non-iti portion of trial
    t.possible_iti = [2 4 6]; % changed from 3 5 8, can maybe do linspace 1-5? for iti jitter
    t.iti = Shuffle(repmat(t.possible_iti,1,p.NumTrials/length(t.possible_iti)));
    t.CueStartsBefore = 1; %starts 1 second before the stimulus comes on 
    t.BeginFixation = 3; %16 TRs need to be extra (16trs * .8ms)
    t.EndFixation = 3;
    %t.TrialStory = []; %Will be the total story so you can go back and look at all that happened
    
    %Stimulus params (general) 
    p.Smooth_size = round(1*p.ppd); %size of fspecial smoothing kernel
    p.Smooth_sd = round(.5*p.ppd); %smoothing kernel sd
    p.PatchSize = round(2*7*p.ppd); %Size of the patch that is drawn on screen location, so twice the radius, in pixels
    p.OuterDonutRadius = (7*p.ppd)-(p.Smooth_size/2); %Size of donut outsides, automatically defined in pixels.
    p.InnerDonutRadius = (1.5*p.ppd)+(p.Smooth_size/2); %Size of donut insides, automatically defined in pixels.
    p.OuterFixRadius = .2*p.ppd; %outter dot radius (in pixels)
    p.InnerFixRadius = p.OuterFixRadius/2; %set to zero if you a donut-hater
    p.FixColor = black;
    p.ResponseLineWidth = 2; %in pixel
    p.ResponseLineColor = white;
    MyPatch = [(CenterX-p.PatchSize/2) (CenterY-p.PatchSize/2) (CenterX+p.PatchSize/2) (CenterY+p.PatchSize/2)];
    

    %Stimulus params (specific) 
    p.SF = 2; %spatial frequency in cpd is actally 2
    p.ContrastTarget = .5; % have to scale up
    p.whitenoiseContrast = 1;
    p.Noise_f_bandwidth = 2;%2; % is actually 2 frequency of the noise bandwidth
    p.Noise_fLow = p.SF/p.Noise_f_bandwidth; %Noise low spatial frequency cutoff
    p.Noise_fHigh = p.SF*p.Noise_f_bandwidth; %Noise high spatial frequency cutoff
    t.MeantToBeTime = t.BeginFixation + t.ActiveTrialDur*p.NumTrials + sum(t.iti) + t.EndFixation;
    p.TestOrient = randsample(1:180,p.NumTrials);
    
   
 %% window setup and gamma correction
 % clock
    PsychJavaTrouble;
   if p.windowed == 0
        [window, ScreenSize] = Screen('OpenWindow', ScreenNr, p.MyGrey);
    else
        % if we're dubugging open a 640x480 window that is a little bit down from the upper left
        % of the big screen
        [window, ScreenSize]=Screen('OpenWindow', ScreenNr, p.MyGrey, [0 0 1024 768]);
    end
    t.ifi = Screen('GetFlipInterval',window);
    if gammacorrect
        OriginalCLUT = Screen('LoadClut', window);
        MyCLUT = zeros(256,3); MinLum = 0; MaxLum = 1;
        if strcmp(p.room,'A') % EEG Room
            CalibrationFile = 'LabEEG-05-Jul-2017';
        elseif strcmp(p.room,'B') % Behavior Room B
            CalibrationFile = 'LabB_20-Jul-2022.mat';
        elseif strcmp(p.room,'C') % Behavior Room C
            CalibrationFile = 'LabC-13-Jun-2016.mat';
        elseif strcmp(p.room,'D') % Beahvior room D
            CalibrationFile = 'LabD_20-Jul-2022.mat';
        else
            disp('No calibration file specified')
        end
        [gamInverse,dacsize] = LoadCalibrationFileRR(CalibrationFile, expdir, p.GeneralUseScripts);
        LumSteps = linspace(MinLum, MaxLum, 256)';
        MyCLUT(:,:) = repmat(LumSteps, [1 3]);
        MyCLUT = map2map(MyCLUT, repmat(gamInverse(:,3),[1 3])); % changed 4 to 3 index position 2 Now the screen output luminance per pixel is linear!
        Screen('LoadCLUT', window, MyCLUT);
        clear CalibrationFile gamInverse
    end
    
    HideCursor;
%% Make target stimuli
for b = startRun:nruns % block loop
    
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
    TargetsAreHere = ones(p.PatchSize,p.PatchSize,p.NumTrials) * p.MyGrey; 

    runner = 1; %Will count within-block trials
    
    startTrialThisBlock = (p.NumTrials * runnumber) - p.NumTrials + 1;
    
    for n = startTrialThisBlock:(startTrialThisBlock+p.NumTrials - 1) 
            % call function that creates filtered gratings
            [image_final] = FilteredGratings(n,p,t,TrialStuff);
            %Make it a donut
            target = image_final.*donut;
            %Give the grating the right contrast level and scale it
            TargetsAreHere(:,:,runner) = max(0,min(255,p.MyGrey+p.MyGrey*(p.ContrastTarget * target)));  
        
        runner = runner + 1;
    end
            
%% Welcome and wait for trigger
    %Welcome welcome ya'll

    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])

    % if in behavioral use this ->
    Screen(window,'TextSize',30);
    Screen('DrawText',window, 'Fixate. Press spacebar to begin.', CenterX-200, CenterY-100, black); % change location potentially
    Screen('Flip', window);
    FlushEvents('keyDown'); %First discard all characters from the Event Manager queue.
    ListenChar(2);
    % just sittin' here, waitin' on my trigger...
    KbName('UnifyKeyNames');
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
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% A TRIAL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:p.NumTrials
        t.TrialStartTime(n) = GlobalTimer; %Get the starttime of each single block (relative to experiment start)
        TimeUpdate = t.StartTime + t.TrialStartTime(n);
        p.TrialNumGlobal = p.TrialNumGlobal+1;
        
        %% Target rendering

            % draw target
            StimToDraw = Screen('MakeTexture', window, TargetsAreHere(:,:,n)); % draw target
            Screen('DrawTexture', window, StimToDraw, [], MyPatch, [], 0); 

            Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
            
            %TIMING!:
            GlobalTimer = GlobalTimer + t.TargetTime;
            TargetTimePassed = 0; %Flush time passed.
            while (TargetTimePassed<t.TargetTime) %As long as the stimulus is on the screen...
                TargetTimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            end
            TimeUpdate = TimeUpdate + t.TargetTime; %Update Matlab on what time it is.
            Screen('Close', StimToDraw);
      
        %t.TrialStory = [t.TrialStory; {'target'} num2str(t.TargetTime)];
clear i
         %% delay 1
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        %TIMING!:
        GlobalTimer = GlobalTimer + t.isi1;
        delay1TimePassed = 0; %Flush time passed.
        while (delay1TimePassed<t.isi1) %As long as the stimulus is on the screen...
            delay1TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
        end
        TimeUpdate = TimeUpdate + t.isi1; %Update Matlab on what time it is.
        %t.TrialStory = [t.TrialStory; {'delay 1'} num2str(t.isi1)];

%% response window
% get RT
% full report spin a line, in quadrant we are probing
        
        react = 0;
        resp_start = GetSecs;
        test_orient = p.TestOrient(n);
        orient_trajectory = [test_orient];
        InitX = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * cos(test_orient*pi/180)+CenterX));
        InitY = round(abs((p.OuterDonutRadius+p.Smooth_size/2) * sin(test_orient*pi/180)-CenterY));
        Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
        Screen('DrawLines', window, [2*CenterX-InitX, InitX; 2*CenterY-InitY, InitY], p.ResponseLineWidth, p.ResponseLineColor,[],1);
        Screen('BlendFunction', window, GL_ONE, GL_ZERO);
        Screen('FillOval', window, p.MyGrey, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window,[],1);
        GlobalTimer = GlobalTimer + t.ResponseTime;
        RespTimePassed = GetSecs-resp_start; %Flush time passed.
        while RespTimePassed<t.ResponseTime  %As long as no correct answer is identified
            RespTimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            [keyIsDown, secs, keyCode] = KbCheck(-1);
            %scanner buttons are: b y g r (form left-to-right)
            if keyCode(KbName('LeftArrow'))%keyCode(KbName('b')) %BIG step CCW
                test_orient = rem(test_orient+2+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('UpArrow'))%keyCode(KbName('y')) %small step CCW
                test_orient = rem(test_orient+.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('DownArrow'))%keyCode(KbName('g')) %small step CW
                test_orient = rem(test_orient-.5+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('RightArrow'))%keyCode(KbName('r')) %BIG step CW
                test_orient = rem(test_orient-2+1440,180);
                react = secs - resp_start;
            elseif keyCode(KbName('ESCAPE')) % If user presses ESCAPE, exit the program.
                cd(datadir); %Change the working directory back to the experimental directory
                if exist(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
                    load(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
                end
                %First I make a list of variables to save:
                TheData(runnumber).info = info;
                TheData(runnumber).t = t;
                TheData(runnumber).p = p;
                TheData(runnumber).data = data;
                TheData(runnumber).trajectory = Trajectory;
                eval(['save(''WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'', ''TheData'', ''TrialStuff'', ''-v7.3'')']);
                cd(expdir)
    
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
                Screen('FillRect', window, p.MyGrey);
                Screen('BlendFunction', window, GL_DST_ALPHA, GL_ONE_MINUS_DST_ALPHA);
                Screen('DrawLines', window, [2*CenterX-UpdatedX, UpdatedX; 2*CenterY-UpdatedY, UpdatedY], p.ResponseLineWidth, p.ResponseLineColor, [], 1);
                Screen('BlendFunction', window, GL_ONE, GL_ZERO);
                Screen('FillOval', window, p.MyGrey, [CenterX-(p.InnerDonutRadius-p.Smooth_size/2) CenterY-(p.InnerDonutRadius-p.Smooth_size/2) CenterX+(p.InnerDonutRadius-p.Smooth_size/2) CenterY+(p.InnerDonutRadius-p.Smooth_size/2)]);        
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius]);
                Screen('Flip', window, [], 1,[], []);
        end
        FlushEvents('keyDown'); %First discard all characters from the Event Manager queue
        data.Response(n) = test_orient;
        data.RTresp(n) = react;
        if data.Response(n) == p.TestOrient(n)
           data.Response(n) = NaN;
        end
        TimeUpdate = TimeUpdate + t.ResponseTime; %Update Matlab on what time it is.
        %t.TrialStory = [t.TrialStory; {'response'} num2str(t.ResponseTime)];
        Trajectory{n} = orient_trajectory;
        

        
        %% feedback screen
feedback_start = GetSecs;
Screen('FillRect',window,p.MyGrey);
Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
Screen('DrawingFinished', window);
Screen('Flip', window);
green = [28 225 48];
red = [225, 28, 31];


%d[np.abs(d)>90]-= 180*np.sign(d[np.abs(d)>90])
err(1,:) = abs(TrialStuff(p.TrialNumGlobal).orient-data.Response(n));
err(2,:) = abs((360-(err(1,:)*2))/2); 
err(3,:) = 360-(err(1,:));
error = abs(min(err)); 


GlobalTimer = GlobalTimer + t.FeedbackTime;
feedbackTimePassed = GetSecs-feedback_start;
while (feedbackTimePassed<t.FeedbackTime)
      if error<20
            % correct display
            Screen('FillRect',window,p.MyGrey);
            Screen('FillOval', window, green, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
      elseif error>=20
            % incorrect display
            Screen('FillRect',window,p.MyGrey);
            Screen('FillOval', window, red, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
            Screen('DrawingFinished', window);
            Screen('Flip', window);
      end
      feedbackTimePassed = (GetSecs-TimeUpdate);
end

 TimeUpdate = TimeUpdate + t.FeedbackTime; %Update Matlab on what time it is.
 %t.TrialStory = [t.TrialStory; {'feedback'} num2str(t.FeedbackTime)];
clear err error
        %% iti 
        Screen('FillRect',window,p.MyGrey);
        Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
        Screen('DrawingFinished', window);
        Screen('Flip', window);
        %TIMING!:
        
        GlobalTimer = GlobalTimer + t.iti(n);
        TimePassed = 0; %Flush time passed.
        while (TimePassed<t.iti(n)) %As long as the stimulus is on the screen...
            TimePassed = (GetSecs-TimeUpdate); %And determine exactly how much time has passed since the start of the expt.
            if TimePassed>=(t.iti(n)-t.CueStartsBefore)
                Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
                Screen('Flip', window);
            end
        end
        TimeUpdate = TimeUpdate + t.iti(n); %Update Matlab on what time it is.
        %t.TrialStory = [t.TrialStory; {'iti'} num2str(t.iti(n))];
        
        
end %end of experimental trial/block loop
    
    
  %----------------------------------------------------------------------
    %LOOK AT BEHAVIORAL PERFOPRMANCE---------------------------------------
    %----------------------------------------------------------------------
    targets_were = [TrialStuff(p.TrialNumGlobal+1-p.NumTrials:p.TrialNumGlobal).orient];
    acc(1,:) = abs(targets_were-data.Response);
    acc(2,:) = abs((360-(acc(1,:)*2))/2); 
    acc(3,:) = 360-(acc(1,:));
    acc = min(acc); 
    %Add minus signs back in
    acc(mod(targets_were-acc,360)==data.Response)=-acc(mod(targets_were-acc,360)==data.Response);
    acc(mod((targets_were+180)-acc,360)==data.Response)=-acc(mod((targets_were+180)-acc,360)==data.Response);
    data.Accuracy = acc;

    
    % get average of trial accuracy if worse than 45 average, tell to stop
    performance = nanmean(abs(data.Accuracy));
    % count non-responses to perform attention check
    check = sum(isnan(data.Response));
    % change feedback depending on performance and attention check
    if check<6 && performance<45 
        blockStr = ['Finished practice run ' num2str(runnumber) ' out of ' num2str(nruns)];
        feedbackStr = [blockStr sprintf('\n') 'Press the spacebar to continue'];
    elseif check>6
         blockStr = ['Finished practice run ' num2str(runnumber) ' out of ' num2str(nruns)];
        feedbackStr = [blockStr sprintf('\n') 'STOP and check with experimenter before continuing'];
    elseif performance>45
        blockStr = ['Finished practice run ' num2str(runnumber) ' out of ' num2str(nruns)];
        feedbackStr = [blockStr sprintf('\n') 'STOP and check with experimenter before continuing'];   
    end
    
    
    % final fixation and feedback:
    Screen('FillOval', window, p.FixColor, [CenterX-p.OuterFixRadius CenterY-p.OuterFixRadius CenterX+p.OuterFixRadius CenterY+p.OuterFixRadius])
    Screen('Flip', window);
    % may need to change spacing
    DrawFormattedText(window,[feedbackStr],CenterX-200,CenterY,white);
    Screen('Flip',window);
    
    while 1
            [keyIsDown, secs, keyCode] = KbCheck([-1]); % KbCheck([-1])
            if keyCode(KbName('space'))
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
    
    
    
  
    
      %----------------------------------------------------------------------
    %SAVE OUT THE DATA-----------------------------------------------------
    %----------------------------------------------------------------------
    cd(datadir); %Change the working directory back to the experimental directory
    if exist(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'])
        load(['WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat']);
    end
    %First I make a list of variables to save:
    TheData(runnumber).info = info;
    TheData(runnumber).t = t;
    TheData(runnumber).p = p;
    TheData(runnumber).data = data;
    TheData(runnumber).trajectory = Trajectory;
    eval(['save(''WM_noiseV5practice_S', num2str(info.SubNum), '_', num2str(info.TheDate), '_Main.mat'', ''TheData'', ''TrialStuff'', ''-v7.3'')']);
    cd(expdir)
    
    FlushEvents('keyDown'); 
    
    runnumber = runnumber+1;
    clear acc
end  % end of block loop
    %----------------------------------------------------------------------
    %WINDOW CLEANUP--------------------------------------------------------
    %----------------------------------------------------------------------
    %This closes all visible and invisible screens and puts the mouse cursor
    %back on the screen
    Screen('CloseAll');
    load('OriginalCLUT_labC.mat')
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
