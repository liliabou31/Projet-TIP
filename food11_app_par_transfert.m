clearvars; close all
inputSize = [224 224 3];
miniBatch = 32;
maxEpochs = 15;
learnRate = 1e-4;

% Chargement du découpage commun du groupe (80 % entraînement / 20 % validation),
% produit par pretraitement_data.m avec rng(42).
% On ne charge que imdsTrain et imdsVal pour garder inputSize = 224x224.
load("food11_pretraitees_data.mat", "imdsTrain", "imdsVal");
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames);

% Redimensionnement en 224x224 RVB (sans augmentation de données)
augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ColorPreprocessing="gray2rgb");
augimdsVal   = augmentedImageDatastore(inputSize(1:2), imdsVal,   ColorPreprocessing="gray2rgb");

% 1. Charger MobileNetV2 et remplacer la sortie 1000 -> 11 classes
net = imagePretrainedNetwork("mobilenetv2");
net = replaceLayer(net, "Logits", fullyConnectedLayer(numClasses, Name="fc_food11"));

% 2. Équivalent de freezeParameters(params, "all") sur un dlnetwork
for i = 1:height(net.Learnables)
    net = setLearnRateFactor(net, net.Learnables.Layer(i), net.Learnables.Parameter(i), 0);
end

% 3. Équivalent de unfreezeParameters sur la dernière couche à 11 classes
net = setLearnRateFactor(net, "fc_food11", "Weights", 1);
net = setLearnRateFactor(net, "fc_food11", "Bias", 1);

net = initialize(net);

% Entraînement (sans régularisation L2)
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatch, ...
    Shuffle="every-epoch", ...
    ValidationData=augimdsVal, ...
    ValidationFrequency=100, ...
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(augimdsTrain, net, "crossentropy", options);

% Sauvegarde du réseau avec sa validation, pour evaluateModel (test_chaine_complete.m)
save("transfert_mobilenetv2.mat", "net", "imdsVal");