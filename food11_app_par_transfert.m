clearvars; close all
%% 1. Paramètres
inputSize = [224 224 3];   % taille des images donnée au réseau
miniBatch = 32;            % voir justification en fin de fichier
maxEpochs = 8;
learnRate = 1e-4;          % petit pas : on ajuste un réseau déjà entraîné

%% 2. Charger les données à partir de food11_pretraitees_data.mat
load("food11_pretraitees_data.mat", "augImdsTrain", "augImdsVal", "imdsTrain", "imdsVal");
classNames = categories(imdsTrain.Labels);
numClasses = numel(classNames)
[imdsTrain, imdsVal] = splitEachLabel(imds, 0.8, "randomized");

%% 3. Augmentation de données
augmenter = imageDataAugmenter( ...
    RandXReflection=true, ...
    RandRotation=[-15 15], ...
    RandScale=[0.9 1.1], ...
    FillValue=128);
dsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    DataAugmentation=augmenter);
dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal);

%% 4. Réseau pré-entraîné
net = imagePretrainedNetwork("mobilenetv2", NumClasses=numClasses);


%% 5. Entraînement (fine-tuning de tout le réseau)
options = trainingOptions("adam", ...
    InitialLearnRate=learnRate, ...
    LearnRateSchedule="piecewise", ...   % le pas est divisé par 2 
    LearnRateDropPeriod=3, ...           %  toutes les 3 époques
    LearnRateDropFactor=0.5, …     % diviser par 2 le taux afin d’affiner l’apprentissage 
    MaxEpochs=maxEpochs, ...
    MiniBatchSize=miniBatch, …
    Shuffle="every-epoch", ...
    ValidationData=dsVal, ...
    ValidationFrequency=100, ...
    ValidationPatience=5, ...            % arrêt si la loss de validation ne s'améliore plus
    Metrics="accuracy", ...
    Plots="training-progress", ...
    Verbose=false);

%% 6. Lancement de l’entraînement
net = trainnet(dsTrain, net, "crossentropy", options);

%% 7. Sauvegarde pour l'étape de test
save("mobilenetv2_food11.mat", "net", "classNames", "inputSize");
