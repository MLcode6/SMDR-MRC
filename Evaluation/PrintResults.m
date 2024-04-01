
function PrintResults(Result)
    [~,n] = size(Result);
    if n == 2
        fprintf('------------------------------------\n');
        fprintf('Evalucation Metric    Mean    Std\n');
        fprintf('------------------------------------\n');
        fprintf('1-HammingLoss         %.3f  %.4f\r',Result(1,1),Result(1,2));
        fprintf('1-RankingLoss         %.4f  %.4f\r',Result(2,1),Result(2,2));
        fprintf('Average_Precision     %.4f  %.4f\r',Result(3,1),Result(3,2));
        fprintf('1-OneError            %.4f  %.4f\r',Result(4,1),Result(4,2));
        fprintf('Macro_AvgF1           %.4f  %.4f\r',Result(5,1),Result(5,2));
        fprintf('MicroF1Measure        %.4f  %.4f\r',Result(6,1),Result(6,2));
        fprintf('Coverage              %.4f  %.4f\r',Result(7,1),Result(7,2));
        fprintf('------------------------------------\n');
    else
        fprintf('\n----------------------------\n');
        fprintf('Evalucation Metric    Mean\n');
        fprintf('----------------------------\n');
        fprintf('1-HammingLoss         %.3f  %.4f\r',Result(1,1));
        fprintf('1-RankingLoss         %.4f  %.4f\r',Result(2,1));
        fprintf('Average_Precision     %.4f  %.4f\r',Result(3,1));
        fprintf('1-OneError            %.4f  %.4f\r',Result(4,1));
        fprintf('Macro_AvgF1           %.4f  %.4f\r',Result(5,1));
        fprintf('MicroF1Measure        %.4f  %.4f\r',Result(6,1));
        fprintf('Coverage              %.4f  %.4f\r',Result(7,1));
        fprintf('----------------------------\n');
    end
end