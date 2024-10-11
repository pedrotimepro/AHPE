function SegData(matData,subject)
    if matData.laterality == "r"
        repetitions = 0;
        oldindex = 1 ;
        newindex = 1;
        for i = 1:length(matData.repetition);
            if matData.repetition(i) ~= repetitions
                newindex = i;
                %保存
                semg = matData.emg(oldindex:newindex,:);
                cyberglove = matData.glove(oldindex:newindex,:);
                imu =matData.acc(oldindex:newindex,:);
                frequncy = matData.frequency(1);
                issaved = savedFile(semg,cyberglove,imu,subject,repetitions,frequncy);
                if issaved
                    saveInfo = ['第',num2str(subject),'第',num2str(repetitions),'已经存储好了']
                    disp(saveInfo);
                else 
                    saveInfo = ['第',num2str(subject),'第',num2str(repetitions),'没存储好']
                    disp(saveInfo);
                end
                %更新
                repetitions = matData.repetition(i);
                oldindex = i;
            end
        end
    end
end

