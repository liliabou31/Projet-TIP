# Guide de Configuration : Classification Food-11 sur MATLAB

Ce projet est découpé en trois scripts M-file pour séparer la préparation des données et l'apprentissage des modèles :

1. **`pretraitement_data.m`** : Charge les images, effectue le split 80/20 et génère le fichier de données.
2. **`test_reseau_convolutif.m`** : Charge les données prétraitées, définit l'architecture du réseau convolutif entraîné de 0 et lance l'entraînement.
3. **`food11_app_par_transfert.m`** : Charge les données prétraitées, définit l'architecture du réseau entraîner par transfert d'apprentissage, et lance l'entraînement.
---

## 1. Modifiables dans `pretraitement_data.m`

Ce fichier prépare les images pour le réseau. Vous devez le ré-exécuter uniquement si vous modifiez la résolution d'entrée ou la répartition des jeux de données.

* **`inputSize = [64 64 3];`**
* **Rôle** : Redimensionne toutes les images du dataset à cette résolution (par exemple, $64 \times 64$ pixels sur 3 canaux RGB).
* `[64 64 3]` : Entraînement très rapide (idéal pour tester rapidement si le code fonctionne).
* `[224 224 3]` : Correspond à la taille des images sur lesquelles le réseau mobilenet a été entrainé. 


* ⚠️ *Attention* : Si vous modifiez `inputSize` ici, veillez à garder la même dimension dans la première couche `imageInputLayer` du second fichier.


* **`splitEachLabel(imdsFull, 0.8, 'randomized');`**
* **Rôle** : Répartition stratifiée du dataset. `0.8` signifie que **80%** des images de chaque classe vont dans l'entraînement et **20%** en validation.


* **`saveFile = 'food11_pretraitées_data.mat';`**
* **Rôle** : Nom du fichier sauvegardé contenant les datastores prêts à l'emploi.



---

## 2. Modifiables dans `test_reseau_convolutif.m`

Ce fichier contrôle la structure du modèle et les hyperparamètres d'apprentissage.

### Hyperparamètres d'entraînement

* **`maxEpochs = 30;`**
* **Rôle** : Nombre de passages complets du réseau sur l'ensemble du dataset d'entraînement.
* **Valeurs suggérées** : Augmenter à `15` ou `30` pour obtenir une meilleure précision finale.


* **`miniBatchSize = 64;`**
* **Rôle** : Nombre d'images traitées simultanément avant de mettre à jour les poids du réseau.




* **`InitialLearnRate = 0.001;`**
* **Rôle** : Pas d'apprentissage de l'optimiseur Adam.



### Architecture du réseau (`layers`)

* **`convolution2dLayer(3, 32, ...)`**
* Le premier paramètre (`3`) est la taille du filtre ($3 \times 3$).
* Le deuxième paramètre (`32`, `64`, `128`, `256`) représente le **nombre de filtres**. Vous pouvez augmenter ces valeurs pour donner plus de capacité au réseau (ex: passer à 64, 128, 256, 512), au prix d'un temps de calcul plus long.


* **`dropoutLayer(0.3)`**
* **Rôle** : Désactive aléatoirement 30% des neurones pendant la rétropropagation pour éviter le surapprentissage (*overfitting*).



---

## 🚀 Ordre d'exécution

1. Lancez `pretraitement_data.m` (à exécuter une première fois pour générer `food11_pretraitees_data.mat`).
2. a) Lancez `test_reseau_convolutif.m` (pour entraîner un réseau neuronal convolutif de zéro).
2. b) Lancez `food11_app_par_transfert.m` (pour entraîner un réseau via transfert d'apprentissage)
3. Modifiez `modeleJson` dans `test_chaine_complete.m` pour évaluer le modèle de votre choix (après l'avoir entrainé). Exemple: `modeleJson = "mobilenetv2_food11;`. Et mettez `genererJson = true` pour obtenir la matrice de confusion et json, sinon `false`.
4. Lancez `test_chaine_complete.m` pour évaluer le modèle sélectionné à l'étape 3. Ce script évalue les performances du modèle via différentes métriques (F1, accuracy, frugality...), génère la matrice de confusion, ainsi qu'un fichier json avec les résultats de la classification des images de test.

# ATTENTION : Pour lancer les programmes vous devez avoir les données de pré-enregistrées dans le dossier courant donc vous devez avoir les dossiers **`test`** et **`train`**