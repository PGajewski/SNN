function Z = calc_jacobian(net, trainSet)
%calc_jacobian calculates jacobian for a NN with:
%   1 hidden tansig layer ('hiddenCount' neurons)
%   1 output linear neuron
%   1 input layer ('inputCount' inputs)

% author: Zbigniew Szymañski <z.szymanski@ii.pw.edu.pl>

    hiddenCount = net.layers{1}.dimensions;
    %NN weights
    w = net.IW{1}';              %weights inputs->hidden neurons
    bin=net.b{1};               %input bias
    v = net.LW{2,1}';           %weights hidden neurons->output
    bout = net.b{2};            %output bias
    
    inputCount= net.inputs{1}.size;
    paramsCount = numel(w)+numel(bin)+numel(v)+numel(bout);
    samplesCount = size(trainSet, 1);
    Z = zeros(samplesCount,paramsCount);
    
    for sample_no=1:samplesCount
        param_no=1;
        %partial derivatives of the output with respect to 
        %hidden neuron weights
        for hidden_no=1:hiddenCount
            tanh_param=0;
            for i=1:inputCount
               tanh_param=tanh_param+w(i,hidden_no)*...
                          trainSet(sample_no,i);
                end
            tanh_param=tanh_param+bin(hidden_no);
            
            for input_no=1:inputCount 
                Z(sample_no,param_no)=...
                    v(hidden_no)*(1-tanh(tanh_param).^2)*...
                    trainSet(sample_no,input_no);
                param_no=param_no+1;
            end
        end
        
        %partial derivatives of the output with respect to 
        %hidden neuron biases
        for hidden_no=1:hiddenCount
            tanh_param=0;
            for i=1:inputCount
               tanh_param=tanh_param+w(i,hidden_no)*...
                          trainSet(sample_no,i);
                end
            tanh_param=tanh_param+bin(hidden_no);
            
            Z(sample_no,param_no)=...
                v(hidden_no)*(1-tanh(tanh_param).^2);
            param_no=param_no+1;
        end

        %partial derivatives of the output with respect to 
        %output neuron weights
        for hidden_no=1:hiddenCount
            tanh_param=0;
            for i=1:inputCount
               tanh_param=tanh_param+w(i,hidden_no)*...
                          trainSet(sample_no,i);
                end
            tanh_param=tanh_param+bin(hidden_no);
            
            Z(sample_no,param_no)=tanh(tanh_param);
            param_no=param_no+1;
        end
        
        %partial derivatives of the output with respect to 
        %output neuron bias
        Z(sample_no,param_no)=1;
    end
    