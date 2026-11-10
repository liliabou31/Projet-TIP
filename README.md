# Guide de Configuration : Classification Food-11 sur MATLAB

Ce projet est découpé en deux scripts M-file pour séparer la préparation des données et l'apprentissage automatique :

1. **`pretraitement_data.m`** : Charge les images, effectue le split 80/20 et génère le fichier de données.
2. **`test_reseau_convolutif.m`** : Charge les données prétraitées, définit l'architecture du réseau et lance l'entraînement. C'EST UN EXEMPLE IL NE MARCHE PAS BIEN MAIS VOUS POUVEZ L'UTILISER POUR VOUS EN INSPIRER

---

## 1. Modifiables dans `pretraitement_data.m`

Ce fichier prépare les images pour le réseau. Vous devez le ré-exécuter uniquement si vous modifiez la résolution d'entrée ou la répartition des jeux de données.

* **`inputSize = [64 64 3];`**
* **Rôle** : Redimensionne toutes les images du dataset à cette résolution ($64 \times 64$ pixels sur 3 canaux RGB).
* **Valeurs suggérées** :
* `[64 64 3]` : Entraînement très rapide (idéal pour tester rapidement si le code fonctionne).
* `[128 128 3]` ou `[160 160 3]` : Meilleur compromis entre précision et temps de calcul (recommandé pour le rendu final).


* ⚠️ *Attention* : Si vous modifiez `inputSize` ici, veillez à garder la même dimension dans la première couche `imageInputLayer` du second fichier.


* **`splitEachLabel(imdsFull, 0.8, 'randomized');`**
* **Rôle** : Répartition stratifiée du dataset. `0.8` signifie que **80%** des images de chaque classe vont dans l'entraînement et **20%** en validation.


* **`saveFile = 'food11_pretraitées_data.mat';`**
* **Rôle** : Nom du fichier sauvegardé contenant les datastores prêts à l'emploi.



---

## 2. Modifiables dans `test_reseau_convolutif.m`

Ce fichier contrôle la structure du modèle et les hyperparamètres d'apprentissage.

### Hyperparamètres d'entraînement

* **`maxEpochs = 5;`**
* **Rôle** : Nombre de passages complets du réseau sur l'ensemble du dataset d'entraînement.
* **Valeurs suggérées** : Augmenter à `15` ou `30` pour obtenir une meilleure précision finale.


* **`miniBatchSize = 500;`**
* **Rôle** : Nombre d'images traitées simultanément avant de mettre à jour les poids du réseau.
* **Valeurs suggérées** :
* `32` ou `64` : Si vous manquez de mémoire RAM ou si vous souhaitez une meilleure généralisation.
* `128` à `500` : Pour accélérer le calcul si votre machine le permet.




* **`InitialLearnRate = 0.01;`**
* **Rôle** : Pas d'apprentissage de l'optimiseur Adam.
* **Valeurs suggérées** : Si la courbe de perte (*loss*) oscille trop fortement ou ne descend pas, diminuez cette valeur à `0.001` ou `0.0001`.



### Architecture du réseau (`layers`)

* **`convolution2dLayer(3, 32, ...)`**
* Le premier paramètre (`3`) est la taille du filtre ($3 \times 3$).
* Le deuxième paramètre (`32`, `64`, `128`, `256`) représente le **nombre de filtres**. Vous pouvez augmenter ces valeurs pour donner plus de capacité au réseau (ex: passer à 64, 128, 256, 512), au prix d'un temps de calcul plus long.


* **`dropoutLayer(0.5)`**
* **Rôle** : Désactive aléatoirement 50% des neurones pendant la rétropropagation pour éviter le surapprentissage (*overfitting*). Vous pouvez ajuster entre `0.2` et `0.5`.



---

## 🚀 Ordre d'exécution

1. Lancez **`pretraitement_data.m`** (à exécuter une première fois pour générer `food11_preprocessed_data.mat`).
2. Lancez **`test_reseau_convolutif.m`** (pour ajuster vos paramètres, entraîner et évaluer le réseau autant de fois que souhaité).

# ATTENTION : Pour lancer les programme vous devez avoir les données de pré-enregistrées dans le dossier courant donc vous devez avoir les dossiers **`test`** et **`train`**