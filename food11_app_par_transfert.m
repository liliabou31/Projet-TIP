%% FOOD-11 - APPRENTISSAGE PAR TRANSFER LEARNING (MOBILENETV2 - VERSION SIMPLE)
clearvars; close all            % Nettoie la mémoire et ferme les fenêtres graphiques pour partir sur une base saine
rng(42);                        % Fixe la graine aléatoire pour obtenir exactement les mêmes résultats sur PC Intel et NVIDIA

%% 1. Paramètres d'entraînement
netName   = "mobilenetv2";      % Modèle choisi : léger (~2.2M paramètres) et très efficace sur les textures de nourriture
learnRate = 1e-4;               % Petit taux d'apprentissage (1e-4) pour ajuster en douceur les couches pré-entraînées
maxEpochs = 6;                  % 6 passages complets sur les données : suffisant pour converger sans sur-apprendre
miniBatch = 32;                 % Paquets de 32 images : bon compromis entre vitesse de calcul et stabilité de l'apprentissage

%% 2. Données et Data Augmentation (Documentation officielle MATLAB)
load("food11_pretraitees_data.mat", "imdsTrain", "imdsVal", "inputSize", "numClasses"); % Charge les données déjà redimensionnées (128x128)
classNames = categories(imdsTrain.Labels); % Récupère la liste des 11 catégories de plats

% Création de l'outil d'augmentation géométrique pour diversifier artificiellement les images d'entraînement
augmenter = imageDataAugmenter( ...
    'FillValue', 128, ...           % Remplit les bords vides après rotation/décalage en gris neutre (évite les bords noirs)
    'RandXReflection', true, ...    % Miroir horizontal aléatoire (un plat vu en miroir reste le même plat)
    'RandRotation', [-15 15], ...   % Légère rotation entre -15° et +15° pour simuler différents angles de prise de vue
    'RandScale', [0.9 1.1]);        % Zoom léger (90% à 110%) pour s'adapter aux plats pris de près ou de loin

% Applique l'augmentation uniquement sur l'entraînement et force 3 canaux (RVB) si une image est en niveaux de gris
dsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');

% Prépare les données de validation SANS augmentation (pour évaluer le modèle sur des images réelles non modifiées)
dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ...
    'ColorPreprocessing', 'gray2rgb');

%% 3. Chargement, adaptation et gel partiel du réseau pré-entraîné
% Charge MobileNetV2 pré-entraîné sur ImageNet et remplace automatiquement la sortie par 11 classes (Food-11)
net = imagePretrainedNetwork(netName, NumClasses=numClasses);

% Adapte la taille d'entrée du réseau à nos images 128x128x3 tout en gardant la normalisation d'origine (zscore)
oldInput = net.Layers(1);
net = replaceLayer(net, oldInput.Name, imageInputLayer(inputSize, ...
    "Name", oldInput.Name, "Normalization", oldInput.Normalization, ...
    "Mean", mean(oldInput.Mean, [1 2]), "StandardDeviation", mean(oldInput.StandardDeviation, [1 2])));

% Gel de la première moitié du réseau : bloque les filtres de base (contours/couleurs) et laisse les couches profondes s'adapter aux plats
moitie = floor(height(net.Learnables) / 2);             % Calcule l'indice de la moitié des paramètres du réseau
for i = 1:moitie                                        % Parcourt uniquement la première moitié
    net.Learnables.LearnRateFactor(i) = 0;              % Bloque l'apprentissage de cette première moitié
end

%% 4. Configuration et lancement de l'entraînement
options = trainingOptions("adam", ...       % Optimiseur Adam : rapide et efficace pour le fine-tuning
    InitialLearnRate=learnRate, ...         % Applique le taux d'apprentissage défini plus haut (1e-4)
    MaxEpochs=maxEpochs, ...                % Nombre total d'époques
    MiniBatchSize=miniBatch, ...            % Taille des lots d'images traités simultanément
    L2Regularization=1e-4, ...              % Légère pénalité L2 pour éviter le surentraînement des couches dégelées
    Shuffle="every-epoch", ...              % Mélange l'ordre des images à chaque époque (évite d'apprendre par classe)
    ValidationData=dsVal, ...               % Évalue la précision sur le jeu de validation pendant l'entraînement
    ValidationFrequency=40, ...             % Fréquence de vérification sur la validation (toutes les 40 itérations)
    Verbose=false, ...                      % Désactive l'affichage texte détaillé dans la console
    Plots="training-progress", ...          % Affiche la courbe d'évolution de la précision en temps réel
    Metrics="accuracy");                    % Suit le pourcentage de bonnes prédictions

net = trainnet(dsTrain, net, "crossentropy", options); % Lance l'entraînement avec la fonction de perte standard de classification

%% 5. Sauvegarde
save("modele_food11_entraine.mat", "net", "classNames", "inputSize"); % Enregistre le modèle entraîné pour la phase de test
