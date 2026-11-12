function res = evaluateModel(net, imdsVal, nom)
% EVALUATEMODEL  Évalue un réseau entraîné sur l'ensemble de validation.
%   Calcule l'accuracy, le F1 par classe et le F1 macro, le nombre de
%   paramètres et le score de frugalité ; affiche la matrice de confusion ;
%   ajoute une ligne au fichier resultats.csv (journal de tous les essais).
%   nom : nom de l'essai"
% net : réseau entraîné, imdsVal : ensemble de validation,
% nom : nom attribué à chaque essai

% --- Prédiction sur la validation (redimensionnement seul, sans augmentation)
classNames = categories(imdsVal.Labels);                 % Ordre des 11 sorties du réseau
inputSize  = net.Layers(1).InputSize;                    % Taille d'entrée propre à chaque réseau
dsVal = augmentedImageDatastore(inputSize(1:2), imdsVal, ColorPreprocessing="gray2rgb"); % redimensionnement des images 64x64
labelsPred = scores2label(minibatchpredict(net, dsVal), classNames);  % prédictions du réseau → noms des classes
labelsVrai = imdsVal.Labels; % vraies étiquettes

% --- F1 par classe à partir de la matrice de confusion
C  = confusionmat(labelsVrai, labelsPred, Order=categorical(classNames, classNames)); % matrice de comptage 11×11
tp = diag(C);                                            % Éléments correctement classés, diagonale = nombres corrects par classe
precision = tp ./ sum(C, 1)';                            % Parmi les prédictions de la classe, part de bien classées
rappel    = tp ./ sum(C, 2);                             % Parmi les vraies images de la classe, part retrouvées par le réseau
precision(isnan(precision)) = 0;                         % Classe jamais prédite → 0
f1 = 2 * precision .* rappel ./ (precision + rappel);
f1(isnan(f1)) = 0;                                       % Classe jamais reconnue → F1 = 0

% --- Métrique du serveur : précision et rappel moyens pondérés par l'inverse
%     de la fréquence des classes, puis F1 global = moyenne harmonique des deux
nParClasse = sum(C, 2);                                  % Nombre d'images réelles par classe
poids = (1 ./ nParClasse) / sum(1 ./ nParClasse);        % Classes rares = poids plus forts
Pw = sum(poids .* precision);                            % Précision moyenne pondérée
Rw = sum(poids .* rappel);                               % Rappel moyen pondéré


% --- Indicateurs globaux
res.Nom       = string(nom);
res.Accuracy  = mean(labelsPred == labelsVrai);  % taux de prédictions correctes
res.F1Macro   = mean(f1);                                          % Moyenne du F1 sur les 11 classes, comme pour le serveur
res.Precision = Pw;
res.Rappel    = Rw;
res.F1Serveur = 2 * Pw * Rw / (Pw + Rw);                           % F1 calculé comme le serveur
res.ParamsM   = sum(cellfun(@numel, net.Learnables.Value)) / 1e6;  % Paramètres en millions
res.Frugalite = res.F1Serveur - 0.3 / (1 + 1 / res.ParamsM);         % Score du challenge (w = 0.3)

disp(struct2table(res));
disp(table(string(classNames), precision, rappel, f1, nParClasse, ...
    VariableNames=["Classe", "Precision", "Rappel", "F1", "NbImages"]));

% --- Matrice de confusion (rappel à droite, précision en bas)
figure
cm = confusionchart(labelsVrai, labelsPred, ...
    RowSummary="row-normalized", ColumnSummary="column-normalized");
cm.Title = sprintf("%s — F1 serveur = %.3f | %.2f M param. | Frug = %.3f", ...
    res.Nom, res.F1Serveur, res.ParamsM, res.Frugalite);

% --- Journal : une ligne par essai (à ouvrir et trier dans Excel)
if isfile("resultats.csv")
    writetable(struct2table(res), "resultats.csv", WriteMode="append");
else
    writetable(struct2table(res), "resultats.csv");
end
end
