function [net]= train_net(train_set,labels,hidden_neurons_count)
    %Opis: funkcja tworz�ca i ucz�ca sie� neuronow�
    %Parametry:
    %   train_set: zbi�r ucz�cy - kolejne punkty w kolejnych wierszach
    %   labels:    etykiety punkt�w - {-1,1}
    %   hidden_neurons_count: liczba neuron�w w warstwie ukrytej
    %Warto�� zwracana:
    %   net - obiekt reprezentuj�cy sie� neuronow�

    %inicjalizacja obiektu reprezentuj�cego sie� neuronow�
    %funkcja aktywacji: neuron�w z warstwy ukrytej - tangens hiperboliczny,
    %                   neuronu wyj�ciowego - liniowa
    %funkcja ucz�ca: gradient descent backpropagation - propagacja wsteczna
    %                   b��du    
    net=newff(train_set',labels',hidden_neurons_count,...
              {'tansig', 'purelin'},'traingd');
          
    rand('state',sum(100*clock));           %inicjalizacja generatora liczb 
                                            %pseudolosowych
	net=init(net);                          %inicjalizacja wag sieci
    net.trainParam.goal = 0.01;             %warunek stopu - poziom b��du
    net.trainParam.epochs = 100;            %maksymalna liczba epok
    net.trainParam.showWindow = false;      %nie pokazywa� okna z wykresami
                                            %w trakcie uczenia
%% Command section for default initialization                                            
    %Change weights and biases.
    bin = zeros(1, hidden_neurons_count);
    w = zeros(1, hidden_neurons_count);
    v = zeros(1, hidden_neurons_count);
    for i = 1: hidden_neurons_count
        w(1,i) = rand*0.3-0.15;
        bin(1,i) = rand*0.3-0.15;
        v(1,i) = rand*0.3-0.15;
    end
    net.IW{1} = w';
    net.LW{2,1} = v;
    net.b{1} = bin';
    net.b{2} = rand*0.3-0.15;
%%    
    %change subset sizes.
    net.divideParam.trainRatio = 0.8;
    net.divideParam.valRatio = 0.2;
     net.divideParam.testRatio = 0;
    
    net=train(net,train_set',labels');      %uczenie sieci
    
    %zmiana funkcji ucz�cej na: Levenberg-Marquardt backpropagation
    net.trainFcn = 'trainlm';
    net.trainParam.goal = 0.01;             %warunek stopu - poziom b��du
    net.trainParam.epochs = 200;            %maksymalna liczba epok
    net.trainParam.showWindow = false;      %nie pokazywa� okna z wykresami
                                            %w trakcie uczenia
    net=train(net,train_set',labels');      %uczenie sieci
    