clc;clear;
folderPath = '';
subject_list = {'s1','s2','s3','s4','s5','s6','s7','s8','s9','s10'};
for subject = 1:numel(subject_list)
    subject_file = fullfile(folderPath,subject_list(subject));
    if isdir(subject_file)
        mat_list = dir(subject_file{1});
        for matFile = 1:size(mat_list,1)
            [path,name,fileExt] = fileparts(mat_list(matFile).name);
            if strcmp(fileExt,'.mat')
                matfiles  = fullfile(subject_file,mat_list(matFile).name);
                matData = load(matfiles{1});
                SegData(matData,subject);
            end
        end
    end
end