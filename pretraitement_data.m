% =========================================================================
% ETAPE 1 : PRÉTRAITEMENT ET PRÉPARATION DES DONNÉES FOOD-11
% =========================================================================
clear; clc;

% 1. Hyperparamètres de dimension
fprintf("les images sont redimensionnées en (64,64,3), peut être modifié\n")
inputSize = [64 64 3];
numClasses = 11;

% 2. Chargement du dossier principal
fprintf("Chargement du datastore complet...\n");
imdsFull = imageDatastore('train', ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% 3. Séparation stratifiée 80% Entraînement / 20% Validation
fprintf("Séparation 80/20...\n");
rng(42);    
[imdsTrain, imdsVal] = splitEachLabel(imdsFull, 0.8, 'randomized');

% 4. Redimensionnement automatique aux dimensions attendues (64x64x3)
fprintf("Redimension des images à la volée\n");
aug = imageDataAugmenter(RandXReflection=true, RandYReflection=true, ...
    RandRotation=[-20 20], RandScale=[0.9 1.1]); % data augmentation pour améliorer la généralisation de l'entrainement
augImdsTrain = augmentedImageDatastore(inputSize, imdsTrain, DataAugmentation=aug);
augImdsVal   = augmentedImageDatastore(inputSize, imdsVal);

% 5. Sauvegarde des structures de données pour le script d'entraînement
saveFile = 'food11_preprocessed_data.mat';
save(saveFile, 'augImdsTrain', 'augImdsVal', 'imdsTrain', 'imdsVal', 'inputSize', 'numClasses');

fprintf("Prétraitement terminé ! Données enregistrées sous : %s\n", saveFile);