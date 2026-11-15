%% 1. Chargement du modèle entraîné et des métadonnées
if ~exist('modele_food11_entraine.mat', 'file')
    error("Le fichier 'modele_food11_entraine.mat' n'existe pas dans le dossier courant.");
end
load('modele_food11_entraine.mat'); % Charge 'net' et 'classNames'

% Récupération automatique de la taille d'entrée attendue par le réseau
targetInputSize = net.Layers(1).InputSize(1:2);

%% 2. Chargement du jeu de test (dossier 'test')
fprintf("Chargement des images de test...\n");
imdsTest = imageDatastore('test', ...
    'IncludeSubfolders', true, ...
    'FileExtensions', {'.jpg', '.png', '.jpeg'});

% Redimensionnement à la volée aux dimensions attendues par le réseau
augImdsTest = augmentedImageDatastore(targetInputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb');

%% 3. Prédiction par mini-batches
fprintf("Calcul des prédictions sur %d images de test...\n", numel(imdsTest.Files));
scores = minibatchpredict(net, augImdsTest);
testLabelsPred = scores2label(scores, classNames);

%% 4. Formattage direct en JSON avec ordre numérique croissant (1, 2, 3...)
fprintf("Formattage des résultats dans l'ordre numérique...\n");
filePaths = imdsTest.Files;
numImages = numel(filePaths);
imgNumbers = zeros(numImages, 1);

% Extraction des numéros pour le tri numérique
for i = 1:numImages
    [~, nameOnly, ~] = fileparts(filePaths{i});
    % Conversion en nombre (ex: "1" -> 1)
    val = str2double(nameOnly);
    if isnan(val)
        error("Le fichier '%s' ne contient pas un numéro valide.", filePaths{i});
    end
    imgNumbers(i) = val;
end

% Tri des indices par valeur numérique (1, 2, 3, ... N)
[~, sortedIndices] = sort(imgNumbers);

% Construction manuelle des lignes JSON (évite le re-tri alphabétique)
jsonEntries = cell(numImages, 1);
for i = 1:numImages
    idx = sortedIndices(i);
    [~, fileNameOnly, ~] = fileparts(filePaths{idx});
    jsonEntries{i} = sprintf('  "%s": "%s"', char(fileNameOnly), char(testLabelsPred(idx)));
end

% Assemblage du JSON final
jsonStr = sprintf('{\n%s\n}', strjoin(jsonEntries, sprintf(',\n')));

%% 5. Écriture du fichier JSON
outputJsonFile = 'predictions_test.json';
fid = fopen(outputJsonFile, 'w', 'native', 'UTF-8');
if fid == -1
    error("Impossible de créer le fichier JSON.");
end
fwrite(fid, jsonStr, 'char');
fclose(fid);

fprintf(" Fichier JSON généré avec succès dans l'ordre numérique exact : %s\n", outputJsonFile);