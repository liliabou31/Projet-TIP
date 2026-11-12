% =========================================================================
% ETAPE 2 : ARCHITECTURE CNN, ENTRAÎNEMENT ET ÉVALUATION
% =========================================================================
clear; clc;

% 1. Chargement des données pré-traitées
dataFile = 'food11_preprocessed_data.mat';
if ~exist(dataFile, 'file')
    error("Le fichier %s n'existe pas. Exécutez d'abord pretraitement_data.m.", dataFile);
end

fprintf("Chargement des données prétraitées depuis %s...\n", dataFile);
load(dataFile); 

% 2. Hyperparamètres d'entraînement
maxEpochs = 5;
miniBatchSize = 500;
smoothingWindow = 10;

% 3. Options d'entraînement et Callback
valFrequency = floor(numel(imdsTrain.Files) / miniBatchSize);

if exist('makeCourseProgressOutputFcn', 'file')
    progressFcn = makeCourseProgressOutputFcn( ...
        numel(imdsTrain.Files), miniBatchSize, maxEpochs, smoothingWindow);
    
    options = trainingOptions("adam", ...
        InitialLearnRate=0.01, ...
        MaxEpochs=maxEpochs, ...
        Shuffle="every-epoch", ...
        MiniBatchSize=miniBatchSize, ...
        ValidationData=augImdsVal, ...
        ValidationFrequency=valFrequency, ...
        Verbose=false, ...
        Plots="training-progress", ...
        OutputFcn=progressFcn, ...
        Metrics="accuracy");
else
    options = trainingOptions("adam", ...
        InitialLearnRate=0.01, ...
        MaxEpochs=maxEpochs, ...
        Shuffle="every-epoch", ...
        MiniBatchSize=miniBatchSize, ...
        ValidationData=augImdsVal, ...
        ValidationFrequency=valFrequency, ...
        Verbose=false, ...
        Plots="training-progress", ...
        Metrics="accuracy");
end

% 4. Architecture du Réseau Convolutif (CNN)
layers = [
    imageInputLayer(inputSize, Name="input")
    
    % Bloc 1 : 64x64 -> 32x32
    convolution2dLayer(3, 32, Padding="same", Name="conv1")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")
    maxPooling2dLayer(2, Stride=2, Name="pool1")
    
    % Bloc 2 : 32x32 -> 16x16
    convolution2dLayer(3, 64, Padding="same", Name="conv2")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")
    maxPooling2dLayer(2, Stride=2, Name="pool2")
    
    % Bloc 3 : 16x16 -> 8x8
    convolution2dLayer(3, 128, Padding="same", Name="conv3")
    batchNormalizationLayer(Name="bn3")
    reluLayer(Name="relu3")
    maxPooling2dLayer(2, Stride=2, Name="pool3")
    
    % Bloc 4 : 8x8 -> 4x4
    convolution2dLayer(3, 256, Padding="same", Name="conv4")
    batchNormalizationLayer(Name="bn4")
    reluLayer(Name="relu4")
    maxPooling2dLayer(2, Stride=2, Name="pool4")
    
    % Tête de Classification
    fullyConnectedLayer(256, Name="fc1")
    reluLayer(Name="relu_fc")
    dropoutLayer(0.5, Name="drop")
    fullyConnectedLayer(numClasses, Name="fc_out")
    softmaxLayer(Name="softmax")
];

% Inspection du réseau
analyzeNetwork(layers);

% 5. Lancement de l'Entraînement
fprintf("Début de l'entraînement avec trainnet...\n");
net = trainnet(augImdsTrain, layers, "crossentropy", options);
save("test_reseau_convolutif.mat", "net");   % Sauvegarde du réseau entraîné (évaluation dans test_chaine_complete.m)

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