clearvars; close all
miniBatch = 32;
maxEpochs = 15;
learnRate = 1e-4;

load("food11_pretraitees_data.mat", "augImdsTrain", "augImdsVal", "imdsTrain", "imdsVal");
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

net = imagePretrainedNetwork("mobilenetv2", NumClasses=numClasses);

headName = net.Learnables.Layer(end);
for i = 1:height(net.Learnables)
    if net.Learnables.Layer(i) ~= headName
        net = setLearnRateFactor(net, net.Learnables.Layer(i), net.Learnables.Parameter(i), 0);
    end
end

options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatch, ...
    Shuffle="every-epoch", ...
    ValidationData=augImdsVal, ...
    ValidationFrequency=100, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(augImdsTrain, net, "crossentropy", options);

save("transfert_mobilenetv2.mat", "net", "imdsVal");
