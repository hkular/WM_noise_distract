%% WM_noise_preprocess
% works with V9 and newer
%setup
clear
close all
clc

% input subjects you wish to process
subs = ['01';'02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'18';'21';'22';'23';'24';'25']; % S1-13 funky data with err n-1

%subs = ['37';'38';'39';'40';'41';'42';'43';'44'; '45';'46';'47';'48';'49';'51';'52';'53';'54';'55';'56';'57';'58';'59';'60';'61';'62';'63';'64';'65';'66';'67';'68';'70';'71';'73';'74';'77';'78';'80']; 
%change path to local machine
%my_path = '/mnt/neurocube/local/serenceslab/holly/behavior/noisefx/V7';
my_path = pwd;%'/Users/hollykular/Documents/GitHub/noisefx/V7';
%info keeping
TheDate = datestr(now,'yymmdd');


%% 
% setting up containers
all_dat = [];
%find the filenames for the subject data
myFolder = pwd;
filePattern = fullfile(myFolder, 'WM_noiseV10_S*');
theFiles = dir(filePattern);       
% for loop runs through each subject
for s = 1:size(subs,1)   
   % set up containers
    subject = [];
    resp = [];
    accuracy = [];
    dist = [];
      
        %load this subjects data
        load([theFiles(s).name]);
        for run = 1:length(TheData) % for each run
            % collect data
            subject = [subject; repmat(TheData(run).info.SubNum, [length(TheData(run).data.Response),1])];
            resp = [resp;TheData(run).data.Response];
            accuracy = [accuracy;TheData(run).data.Accuracy];
            dist = [dist;TheData(run).data.DistResp];

        end
     
    ss = num2cell(str2num(subject)); [TrialStuff(1:length(ss)).('subject')] = ss{:};
    rs = num2cell(resp); [TrialStuff(1:length(rs)).('resp')] = rs{:};
    ac = num2cell(accuracy); [TrialStuff(1:length(ac)).('acc')] = ac{:};
    ds = num2cell(dist); [TrialStuff(1:length(ds)).('dist')] = ds{:};
    
    collect = TrialStuff(1:length(resp));

    all_dat = [all_dat collect];
    
    clear collect ss rs ac  
end
%% let's look at the staircases
% p.stairstep * p.change is what happened, so p.change is constant at 10
% deg and then p.stairstep changes run by run

    stairs = [];
    subject = [];
    runs = [];
for s = 1:size(subs,1)

    load([theFiles(s).name]);
    for run = 1:length(TheData)
        stairs = [stairs; TheData(run).p.stairstep];
        subject = [subject; TheData(run).info.SubNum];
        runs = [runs; run];
    end
end

stair = array2table([stairs runs str2num(subject)]);
% graph it for each subject

% Unique labels
unique_labels = str2num(subs);

% Number of rows and columns in the grid
numRows = 4;
numCols = 6;

% Create a figure
figure;

% Loop through each unique label
for i = 1:length(unique_labels)
    label_data = stair(stair.Var3 == unique_labels(i), :);
    
    % Create subplots
    subplot(numRows, numCols, i);
    
    % Scatter plot
    scatter(label_data.Var2, label_data.Var1, 'filled');
    
    % Title for the subplot
    title(['Subject ' sprintf('%.0f',unique_labels(i))]);
    
    % Set labels and limits if needed
    xlabel('Run');
    ylabel('stairstep');
    xlim([0 15]);
    ylim([0 1.2]);
end

% Adjust layout for subplots
sgtitle('Grid of Scatter Plots');
%% save csv
    
eval(['save(''WM_noiseV10_', num2str(TheDate), '.mat'', ''all_dat'', ''-v7.3'')']);

 %change this to update the date                                               
writetable(struct2table(all_dat),'WM_noiseV10.csv');
    
% cnt = 0;
% for i = 1:12
% cnt = cnt + sum(isnan(TheData(i).data.Response));
% end

% % fix subject 37 is actually 34
% for i = 1:14
%     TheData(i).info.SubNum = '34';
% end
% save('WM_noiseV6_S34_220906_Main.mat')


% figure out response direction to look at bias
% look at trajectory
% i = 1:14; j = 1:12;
% if TheData(i).trajectory{1,j}(1) == 180
% else
%     TheData(i).trajectory{1,k}
% end

% untangling if change happened on current trial for all trials, runs 2:10
% trial = 37; n = 1;
%     if TrialStuff(n).distractor == 0
%         change(trial) = 0;
%     else
%         if TrialStuff(n).distractortask == 0
%             change(trial) = 0;
%         else
%             

% s28 get acc on last run fix subm=num is 23
% run = 10;
% startTrialThisRun = (TheData(run).p.NumTrials * run) - TheData(run).p.NumTrials+1;
% targets_were = [TrialStuff(startTrialThisRun:360).orient]';
% response = [TheData(run).data.Response];
% acc(1,:) = abs(targets_were-response);
% acc(2,:) = abs((360-(acc(1,:)*2))/2);
% acc(3,:) = 360-(acc(1,:));
% acc = min(acc);
% acc = acc';
% %Add minus signs back in
% acc(mod(targets_were-acc,360)==response)=-acc(mod(targets_were-acc,360)==response);
% acc(mod((targets_were+180)-acc,360)==response)=-acc(mod((targets_were+180)-acc,360)==response);
% TheData(run).data.Accuracy = acc;
% 
% eval(['save(''WM_noiseV9_S23_230511_Main.mat'', ''TheData'', ''TrialStuff'', ''-V7.3'')']);
% 
% for i = 1:length(TheData)
%     TheData(i).info.SubNum = '23';
% end
%         
        
        
