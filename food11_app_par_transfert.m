%% FOOD-11 - APPRENTISSAGE PAR TRANSFER LEARNING (avec augmentation)
clearvars; close all

%% 1. Paramètres
netName   = "efficientnetb0";
learnRate = 1e-4;
maxEpochs = 8;
miniBatch = 64;

%% 2. Données
load("food11_pretraitees_data.mat", "imdsTrain", "imdsVal", "inputSize", "numClasses");
classNames = categories(imdsTrain.Labels);

% Entraînement : images augmentées à chaque lecture. Validation : redimensionnement seul.
imdsTrain.ReadSize = miniBatch;
dsTrain = transform(imdsTrain, ...
    @(data, info) augmentationFood(data, info, inputSize(1:2)), IncludeInfo=true);
dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ColorPreprocessing="gray2rgb");

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

%% 6. Sauvegarde (pour l'étape de test)
save("modele_food11_entraine.mat", "net", "classNames", "inputSize");

%% Fonction locale (doit rester à la fin du script)
function [dataOut, info] = augmentationFood(dataIn, info, tailleCible)
% Augmentation : on modifie l'image, le label reste inchangé.
dataOut = cell(size(dataIn, 1), 2);
for idx = 1:size(dataIn, 1)
    img = dataIn{idx};
    if size(img, 3) == 1
        img = repmat(img, [1 1 3]);            % gris -> 3 canaux
    end
    img = imresize(img, tailleCible);

    tform = randomAffine2d(XReflection=true, YReflection=true, ...
        Rotation=[-30 30], Scale=[0.9 1.1]);
    img = imwarp(img, tform, OutputView=affineOutputView(size(img), tform), ...
        FillValues=[128 128 128]);

    img = jitterColorHSV(img, Brightness=[-0.1 0.1], ...
        Contrast=[0.9 1.1], Saturation=[-0.1 0.1]);

    dataOut(idx, :) = {img, info.Label(idx)};
end
end