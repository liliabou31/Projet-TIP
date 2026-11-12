% TEST_CHAINE_COMPLETE  Évaluation des réseaux déjà entraînés par l'équipe (pas d'apprentissage ici).
clear; close all

dossierTest = "test";       % 3097 images sans label
genererJson = false;        % true uniquement pour le réseau retenu
modeleJson  = "chờ xem kết quả csv rồi chọn 1 tên trùng với cột đầu của modeles";

% nom (unique)          fichier du réseau (contient net)   fichier .mat du split utilisé à l'entraînement
modeles = [
    "test_reseau_convolutif",  "test_reseau_convolutif.mat",  "food11_preprocessed_data.mat"
    "reseau_simple",           "reseau_simple.mat",           "food11_preprocessed_data.mat"
    %"modele_food11_entraine",  "modele_food11_entraine.mat",  "food11_preprocessed_data.mat"
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
