%% FOOD-11 - APPRENTISSAGE PAR TRANSFER LEARNING (Documentation officielle imageDataAugmenter)
clearvars; close all
rng(42); % Garantit la même initialisation sur PC Intel et PC NVIDIA

%% 1. Paramètres
netName   = "efficientnetb0";
learnRate = 1e-4;
maxEpochs = 6;
miniBatch = 32;

%% 2. Données et Data Augmentation (selon doc MathWorks imageDataAugmenter)
load("food11_pretraitees_data.mat", "imdsTrain", "imdsVal", "inputSize", "numClasses");
classNames = categories(imdsTrain.Labels);

% Définition des transformations selon les propriétés officielles de imageDataAugmenter
augmenter = imageDataAugmenter( ...
    'FillValue', 128, ...
    'RandXReflection', true, ...
    'RandRotation', [-15 15], ...
    'RandScale', [0.9 1.1], ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8]);

% Création des datastores : redimensionnement + augmentation pour le train, redimensionnement seul pour la val
dsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, ...
    'ColorPreprocessing', 'gray2rgb');

dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 3. Réseau pré-entraîné (couche d'entrée adaptée à inputSize)
net = imagePretrainedNetwork(netName, NumClasses=numClasses);

oldInput  = net.Layers(1);
inputArgs = {"Name", oldInput.Name, "Normalization", oldInput.Normalization};
if oldInput.Normalization ~= "none"
    inputArgs = [inputArgs, {"Mean", mean(oldInput.Mean, [1 2])}];
end
if oldInput.Normalization == "zscore"
    inputArgs = [inputArgs, {"StandardDeviation", mean(oldInput.StandardDeviation, [1 2])}];
end
net = replaceLayer(net, oldInput.Name, imageInputLayer(inputSize, inputArgs{:}));

if ~net.Initialized
    net = initialize(net);
end

%% 4. Entraînement
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatch, ...
    L2Regularization=1e-5, ...
    Shuffle="every-epoch", ...
    ValidationData=dsVal, ...
    ValidationFrequency=40, ...
    Verbose=false, ...
    Plots="training-progress", ...
    Metrics="accuracy");

net = trainnet(dsTrain, net, "crossentropy", options);

%% 5. Sauvegarde (pour l'étape de test)
save("modele_food11_entraine.mat", "net", "classNames", "inputSize");
