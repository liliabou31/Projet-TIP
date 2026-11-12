function predictTest(net, classNames, testDir, fichierJson)
% PREDICTTEST  Prédit la classe des images de test et écrit le JSON de soumission.
%   Format : { "0": "bread", "1": "fried", ... }
%   clé = nom de l'image sans extension, valeur = classe prédite en minuscules.

% --- Prédiction (même prétraitement que la validation : redimensionnement seul)
imdsTest  = imageDatastore(testDir);
inputSize = net.Layers(1).InputSize;
dsTest = augmentedImageDatastore(inputSize(1:2), imdsTest, ColorPreprocessing="gray2rgb");
labelsPred = scores2label(minibatchpredict(net, dsTest), classNames);

% --- Association nom d'image → classe, puis écriture du JSON
[~, noms] = cellfun(@fileparts, imdsTest.Files, UniformOutput=false);   % "123.jpg" → "123"
classes = cellstr(lower(string(labelsPred)));                           % Minuscules exigées
m = containers.Map(noms, classes);              % Encodé en objet JSON (clés numériques autorisées)

fid = fopen(fichierJson, "w", "n", "UTF-8");
fprintf(fid, "%s", jsonencode(m, PrettyPrint=true));
fclose(fid);

% --- Vérifications minimales avant soumission
fprintf("%d prédictions écrites dans %s\n", m.Count, fichierJson);
if m.Count ~= 3097
    warning("Le JSON devrait contenir 3097 entrées.");
end
disp(countcats(labelsPred)');   % Répartition des prédictions : aucune classe ne doit tout absorber
end
