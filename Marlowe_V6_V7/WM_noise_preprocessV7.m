%% WM_noise_preprocess
% works with V7 and newer
%setup
clear
close all
clc

% input subjects you wish to process
%v7 subs = ['01';'03';'10';'13';'30'; '34'; '36';'39';'44';'45';'49';'50';'50';'52';'53'; '54';'58';'59';'60';'61';'62';'63';'64';'65';'66';'67';'68';'69'];
%v8
subs = ['02';'03';'04';'05';'06';'07';'08';'09';'10';'11';'12';'13';'14';'15';'16';'17';'18';'19';'20';'21';'22';'23';'24';'25';'26';'27';'28';'29';'30';'31';'32';'33';'34';'35';'36';'37';'38';'39';'40';'41';'42';'43'];

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
filePattern = fullfile(myFolder, 'WM_noiseV8_*');
theFiles = dir(filePattern);       
% for loop runs through each subject
for s = 1:size(subs,1)
 
   % set up containers
    subject = [];
    respRT = [];
    acc = [];
    resp = [];
      
        %load this subjects data
        load([theFiles(s).name]);
        for run = 1:length(TheData) % for each run    
            subject = [subject; repmat(TheData(run).info.SubNum, [length(TheData(run).data.Response),1])];
            resp = [resp;TheData(run).data.Response];
            for trial = 1:length(TheData(run).data.RTresp)
               respRT = [respRT; TheData(run).data.RTresp{trial,1}(end)]; 
            end
            acc = [acc; TheData(run).data.Accuracy];          
        end
   
    % add trial number and block 
   
    ss = num2cell(str2num(subject)); [TrialStuff(1:length(ss)).('subject')] = ss{:};
    rs = num2cell(resp); [TrialStuff(1:length(rs)).('resp')] = rs{:};
    rrt = num2cell(respRT); [TrialStuff(1:length(rrt)).('respRT')] = rrt{:};
    ac = num2cell(acc); [TrialStuff(1:length(ac)).('acc')] = ac{:};

    collect = TrialStuff(1:length(resp));
    all_dat = [all_dat collect];
    
    %clear collect ss rs rrt ac filePattern
end
eval(['save(''WM_noiseV8_', num2str(TheDate), '.mat'', ''all_dat'', ''-v7.3'')']);

 %change this to update the date                                               
writetable(struct2table(all_dat),'WM_noiseV8.csv');
    
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


