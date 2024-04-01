function [MacroF1,MicroF1]=Mac_Mic_F1(Pre_Labels,test_target)
%This function calculates six measures based on instances.
%INPUT:
%  Pre_Labels: the predicted label (+1/-1) matrix of size Classes*Instances.
%  test_target: the actual label (+1/-1) matrix of size Classes*Instances.
%  where each column --> labels of a instance
%        each row    --> labels of different instances
%OUTPUT:
%  six label-based measures

    [num_class, num_instance]=size(Pre_Labels);
    
    TP=sum((Pre_Labels'>0).*(test_target'>0));
    FP=sum((Pre_Labels'>0).*(test_target'<0));
    FN=sum((Pre_Labels'<0).*(test_target'>0));
    
    for j=1:num_class
        if (TP(j)==0 & FP(j)==0)
            Pr(j)=0.0;
        else
            Pr(j)=TP(j)/(TP(j)+FP(j));
        end
        
        if (TP(j)==0 & FN(j)==0)
            Re(j)=0.0;
        else
            Re(j)=TP(j)/(TP(j)+FN(j));
        end
        
        if (Pr(j)==0 & Re(j)==0)     
            
            F1(j)=0.0;
        else
            F1(j)=2.0*(Pr(j)*Re(j))/(Pr(j)+Re(j));
        end
    end
    
%     MacroPrecision=mean(Pr);
%     MacroRecall=mean(Re);
    MacroF1=mean(F1);
    
    MicroPrecision=sum(TP)/sum(TP+FP);
    MicroRecall=sum(TP)/sum(TP+FN);
    MicroF1=2*MicroPrecision*MicroRecall/(MicroPrecision+MicroRecall);
end