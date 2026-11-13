% TEST_CHAINE_COMPLETE  Évaluation des réseaux déjà entraînés par l'équipe (pas d'apprentissage ici).
clear; close all

dossierTest = "test";       % 3097 images sans label
genererJson = true;        % true uniquement pour le réseau retenu
modeleJson  = "mobilenetv2_food11";

% nom (unique)          fichier du réseau (contient net)   fichier .mat du split utilisé à l'entraînement
modeles = [
    %"test_reseau_convolutif",  "test_reseau_convolutif.mat",  "food11_pretraitees_data.mat"
    %"reseau_simple",           "reseau_simple.mat",           "food11_pretraitees_data.mat"
    "mobilenetv2_food11",  "mobilenetv2_food11.mat",  "food11_pretraitees_data.mat"
];


for k = 1:size(modeles, 1)
    nom = modeles(k, 1);
    S = load(modeles(k, 2), "net");
    D = load(modeles(k, 3), "imdsVal");
    res = evaluateModel(S.net, D.imdsVal, nom);            % ligne ajoutée à resultats.csv
    exportgraphics(gcf, "confusion_" + nom + ".png");       % figure pour le rapport
    if genererJson && nom == modeleJson
        classNames = categories(D.imdsVal.Labels);
        predictTest(S.net, classNames, dossierTest, "predictions_" + nom + ".json");
    end
end