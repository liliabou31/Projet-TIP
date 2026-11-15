%% FOOD-11 - Fine-tuning de MobileNetV2 (224x224) avec augmentation de données
% Pour chaque fonction, la page de documentation s'ouvre avec : doc nomDeLaFonction
% L'évaluation (matrice de confusion, F1, frugalité) est faite par evaluateModel.m,
% appelé depuis test_chaine_complete.m.
clearvars; close all

%% 1. Paramètres
inputSize = [224 224 3];   % taille des images donnée au réseau
miniBatch = 32;
maxEpochs = 15;
learnRate = 1e-4;          % petit pas : on ajuste un réseau déjà entraîné

%% 2. Données : un dossier par classe, séparation 80 % / 20 %
% Réf. MATLAB : imageDatastore (LabelSource="foldernames"), splitEachLabel
rng(0)                     % même séparation à chaque exécution
imds = imageDatastore("train", IncludeSubfolders=true, LabelSource="foldernames");
classNames = categories(imds.Labels);
numClasses = numel(classNames);
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");

%% 3. Augmentation (entraînement uniquement) et redimensionnement
% Réf. MATLAB : imageDataAugmenter, augmentedImageDatastore (ColorPreprocessing="gray2rgb")
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-15 15], ...
    RandScale=[0.9 1.1], ...
    FillValue=128);
dsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=augmenter, ColorPreprocessing="gray2rgb");
dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    ColorPreprocessing="gray2rgb");

%% 4. Réseau pré-entraîné
% Réf. MATLAB : imagePretrainedNetwork (NumClasses remplace la dernière couche)
net = imagePretrainedNetwork("mobilenetv2", NumClasses=numClasses);

% Couche d'entrée redéfinie à inputSize en gardant la normalisation du réseau
% pré-entraîné. Réf. MATLAB : imageInputLayer, replaceLayer, initialize
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

%% 5. Entraînement (fine-tuning de tout le réseau)
% Réf. MATLAB : trainingOptions, trainnet
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    LearnRateSchedule="piecewise", ...   % le pas est divisé par 2 ...
    LearnRateDropPeriod=3, ...           % ... toutes les 3 époques
    LearnRateDropFactor=0.5, ...
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatch, ...
    Shuffle="every-epoch", ...
    ValidationData=dsVal, ...
    ValidationFrequency=100, ...
    ValidationPatience=5, ...            % arrêt si la loss de validation ne s'améliore plus
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

net = trainnet(dsTrain, net, "crossentropy", options);

%% 6. Sauvegarde du réseau avec sa validation (pour test_chaine_complete.m)
save("mobilenetv2_food11.mat", "net", "classNames", "inputSize", "imdsVal");