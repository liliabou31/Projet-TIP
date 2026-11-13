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
maxEpochs = 5;
miniBatchSize = 500;
smoothingWindow = 10;

% 3. Options d'entraînement et Callback
valFrequency = floor(numel(imdsTrain.Files) / miniBatchSize);

if exist('makeCourseProgressOutputFcn', 'file')
    progressFcn = makeCourseProgressOutputFcn( ...
        numel(imdsTrain.Files), miniBatchSize, maxEpochs, smoothingWindow);
    
    options = trainingOptions("adam", ...
        InitialLearnRate=0.0001, ...
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
        InitialLearnRate=0.0001, ...
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
    

    % Bloc 1 : 
    fullyConnectedLayer(256, Name="fc1")
    batchNormalizationLayer(Name="bn1")
    reluLayer(Name="relu1")
    dropoutLayer(0.3, Name="drop1")

    % Bloc 2 : 
    fullyConnectedLayer(128, Name="fc2")
    batchNormalizationLayer(Name="bn2")
    reluLayer(Name="relu2")
    dropoutLayer(0.3, Name="drop2")

    % Bloc 3 : 
    fullyConnectedLayer(64, Name="fc3")
    batchNormalizationLayer(Name="bn3")
    reluLayer(Name="relu3")
    dropoutLayer(0.3, Name="drop3")
    
    % Tête de Classification
    fullyConnectedLayer(numClasses, Name="fc_out")
    softmaxLayer(Name="softmax")
];

% Inspection du réseau
analyzeNetwork(layers);

% 5. Lancement de l'Entraînement
fprintf("Début de l'entraînement avec trainnet...\n");
net = trainnet(augImdsTrain, layers, "crossentropy", options);
save("reseau_simple.mat", "net"); 
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