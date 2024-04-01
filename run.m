%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off % #ok<WNOFF>
clear
clc

addpath('function');
addpath('Evaluation');
addpath('centered_data');
addpath('basic classifier');

% datasets = ["arts","business","computers","education","entertainment","health","recreation","reference","science","social","society"];
datasets = "emotions"; % Êý¾Ý¼¯
datasets_num = length(datasets);

for i = 1:datasets_num
    dataset_name = char(datasets(i) + '.mat');
    [Avg_Means,Avg_Stds] = Main(dataset_name);
    
    % pathname = 'E:\Master\code\SMDR-IC_code\';
    % filename = char(datasets(i) + "_new_result_2.mat"); 
    % save([pathname,filename],'Avg_Means','Avg_Stds');
    
end