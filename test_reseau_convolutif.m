% =========================================================================
% ETAPE 2 : ARCHITECTURE CNN, ENTRAÎNEMENT ET ÉVALUATION
% =========================================================================
clear; clc;

% 1. Chargement des données pré-traitées
dataFile = 'food11_pretraitees_data.mat';
if ~exist(dataFile, 'file')
    error("Le fichier %s n'existe pas. Exécutez d'abord pretraitement_data.m.", dataFile);
end

fprintf("Chargement des données prétraitées depuis %s...\n", dataFile);
load(dataFile); 

% 2. Hyperparamètres d'entraînement
maxEpochs = 30;
miniBatchSize = 64;
smoothingWindow = 10;

% 3. Options d'entraînement et Callback
valFrequency = floor(numel(imdsTrain.Files) / miniBatchSize);

if exist('makeCourseProgressOutputFcn', 'file')
    progressFcn = makeCourseProgressOutputFcn( ...
        numel(imdsTrain.Files), miniBatchSize, maxEpochs, smoothingWindow);
    
    options = trainingOptions("adam", ...
        InitialLearnRate=0.001, ...
        MaxEpochs=maxEpochs, ...
        Shuffle="every-epoch", ...
        MiniBatchSize=miniBatchSize, ...
        L2Regularization=0.0001, ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropFactor=0.3, ...
        LearnRateDropPeriod=10, ...
        ValidationData=augImdsVal, ...
        ValidationFrequency=valFrequency, ...
        ValidationPatience=8, ...        % arrêt anticipé si la val. loss stagne
        CheckpointPath=pwd, ...            % sauvegarde en cas de plantage
        CheckpointFrequency=2, ...
        Verbose=false, ...
        Plots="training-progress", ...
        OutputFcn=progressFcn, ...
        Metrics="accuracy");
else
    options = trainingOptions("adam", ...
        InitialLearnRate=0.001, ...
        MaxEpochs=maxEpochs, ...
        Shuffle="every-epoch", ...
        MiniBatchSize=miniBatchSize, ...
        L2Regularization=0.0001, ...
        LearnRateSchedule="piecewise", ...
        LearnRateDropFactor=0.3, ...
        LearnRateDropPeriod=10, ...
        ValidationData=augImdsVal, ...
        ValidationFrequency=valFrequency, ...
        Verbose=false, ...
        Plots="training-progress", ...
        Metrics="accuracy");
end

% 4. Architecture du Réseau Convolutif (CNN)
layers = [
    imageInputLayer(inputSize, Name="input")
    
    % Bloc 1 : 128x128 -> 64x64
    convolution2dLayer(3, 32, Padding="same", Name="conv1")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")
    convolution2dLayer(3, 32, Padding="same", Name="conv1.2")
    batchNormalizationLayer(Name="bn1.2")
    reluLayer(Name="relu1.2")
    maxPooling2dLayer(2, Stride=2, Name="pool1")
    
    % Bloc 2 : 64x64 -> 32x32
    convolution2dLayer(3, 64, Padding="same", Name="conv2")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")
    convolution2dLayer(3, 64, Padding="same", Name="conv2.2")
    batchNormalizationLayer(Name="bn2.2")
    reluLayer(Name="relu2.2")
    maxPooling2dLayer(2, Stride=2, Name="pool2")

    
    % Bloc 3 : 32x32 -> 16x16
    convolution2dLayer(3, 128, Padding="same", Name="conv3")
    batchNormalizationLayer(Name="bn3")
    reluLayer(Name="relu3")
    convolution2dLayer(3, 128, Padding="same", Name="conv3.2")
    batchNormalizationLayer(Name="bn3.2")
    reluLayer(Name="relu3.2")
    maxPooling2dLayer(2, Stride=2, Name="pool3")

    
    % Bloc 4 : 16x16 -> 8x8
    convolution2dLayer(3, 256, Padding="same", Name="conv4")
    batchNormalizationLayer(Name="bn4")
    reluLayer(Name="relu4")
    convolution2dLayer(3, 256, Padding="same", Name="conv4.2")
    batchNormalizationLayer(Name="bn4.2")
    reluLayer(Name="relu4.2")
    
    
    % Tête de Classification
    globalAveragePooling2dLayer(Name="gap")
    dropoutLayer(0.3, Name="drop")
    fullyConnectedLayer(numClasses, Name="fc_out")
    softmaxLayer(Name="softmax")
];

% Inspection du réseau
analyzeNetwork(layers);

% 5. Lancement de l'Entraînement
fprintf("Début de l'entraînement avec trainnet...\n");
net = trainnet(augImdsTrain, layers, "crossentropy", options);
save("test_reseau_convolutif.mat", "net", "inputSize", "imdsVal");
% 6. Évaluation des Performances
fprintf("\n=========================================\n");
fprintf("      ÉVALUATION SUR FOOD-11             \n");
fprintf("=========================================\n");

% Précision Entraînement
trainScores = minibatchpredict(net, augImdsTrain);
trainLabelsPred = scores2label(trainScores, categories(imdsTrain.Labels));
trainAccuracy = mean(trainLabelsPred == imdsTrain.Labels);
fprintf("Précision Entraînement : %.2f%%\n", 100 * trainAccuracy);

% Précision Validation
valScores = minibatchpredict(net, augImdsVal);
valLabelsPred = scores2label(valScores, categories(imdsVal.Labels));
valAccuracy = mean(valLabelsPred == imdsVal.Labels);
fprintf("Précision Validation    : %.2f%%\n", 100 * valAccuracy);