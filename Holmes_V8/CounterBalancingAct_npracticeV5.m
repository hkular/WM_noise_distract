function [TrialStuff, designMat, trial_cnt_shuffled, MinNumTrials] = CounterBalancingAct_npracticeV5(OrientBins, Kappa)
%Syntax:
%CounterBalancingAct(OrientBins, Contrast)
%
%This little function does all my counterbalancing such that I can save out
%what needs to happen on every trial and reload it several times for
%different runs. This is important, because once I fully counterbalance all
%my conditions I end up with a lot of trials, more that can be fit into a
%single run.
%
%IN --->
%     OrientBins: Orientation values in my bins (matrix of orients_in_bin
%     by num_bins big)
%     Distractor level: 0, 50, 100 % contrast
%     Kappa: 100 1000 5000 the bandwidth of the orientation, definitely
%     mess with these numbers
%
%OUT --->
%     TrialStuff: A struct of which each entry has all the information
%     needed for a given trial. Index it by TrialStuff(trial_num) to see
%     all that is in there for that trial. Index it by
%     TrialStuff(trial_num).specific_thing_I_need to get the shit you need.
%     MinNumTrials: The minimum number of trials required for
%     counterbalancing. In the end if you wanna fully counterbalance your
%     life you want to run this many, or any multiple of this many trials.
% Written by HK, Jun 2022

%----------------------------------------------------------------------
%COUNTERBALANCING ACT--------------------------------------------------
%----------------------------------------------------------------------
% make mini design matrix,
% columns that are fully counterbalanced: [ori distractorlevel kappa]

designMat = fullfact([2 3 3]); % 2 ori bins and 3 distractor levels and 3 orientation bandwidths


designMat = repmat(designMat,5,1); % replicate for 90 trials total

% shuffle trials
trial_cnt = 1:length(designMat);
trial_cnt_shuffled = Shuffle(trial_cnt); % shuffle orientation
trial_cnt_pshuff = [1:5,trial_cnt_shuffled(5:length(trial_cnt_shuffled))];%partial shuffle, make first trials easy kappa
%% ----------------------------------------------------------------------

% now let's convert into the structure the experiment script expects
TrialStuff = [];

for i = 1:length(designMat)
    
        trial.orient = randsample(OrientBins(:,(designMat(trial_cnt_shuffled(i),1))),1);% orientation is full counterbalanced
        trial.kappa = Kappa(designMat(trial_cnt_pshuff(i),3)); 
 
        TrialStuff = [TrialStuff  trial]; 
end

MinNumTrials = length(TrialStuff); % my number of trials follows from the minimum I need to get everything counterbalanced...


