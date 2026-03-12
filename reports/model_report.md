# Rapport de Modélisation — Scoring de Churn Client

## Contexte

Prédiction du churn (résiliation) des clients d'un opérateur télécom à partir du dataset Telco Customer Churn. L'objectif est d'identifier les clients à risque pour déclencher des actions de rétention ciblées.

- **Dataset** : 7 043 clients, 20 variables (démographiques, services souscrits, facturation)
- **Classe cible** : `Churn` (Yes/No) — ~26 % de churners (déséquilibre modéré)
- **Split** : 80/20 stratifié (`RANDOM_STATE=1204`)

---

## Protocole d'Analyse

Le protocole suit une séparation stricte entre entraînement et test. Le jeu de test (`X_test`) n'est utilisé qu'**une seule fois**, à la toute fin du processus, pour garantir des scores non biaisés.

| Étape | Données utilisées | Objectif |
|-------|:-:|---------|
| **Étapes 2-4** — Finetuning | `X_train` (CV interne 5-fold) | Optimisation des hyperparamètres (GridSearchCV / RandomizedSearchCV) |
| **Étape 5** — Nested CV | `X_train` (5 folds ext. × 5 folds int.) | Estimation non biaisée de la généralisation |
| **Étape 6** — Comparaison Baseline vs Tuned | `X_train` (`cross_val_predict`) | Prédictions out-of-fold, tableau comparatif, matrices de confusion |
| **Étape 7** — Seuil de décision | `X_train` (prédictions out-of-fold) | Calibrage du seuil optimal (max F1), courbes ROC, lift marketing |
| **Étape 8** — Évaluation finale | `X_test` (**unique utilisation**) | Scores non biaisés, sélection du modèle, export `.joblib` |

> **Pourquoi calibrer le seuil sur le train ?** Optimiser le seuil sur le jeu de test biaiserait l'évaluation finale. Le seuil est un hyperparamètre de décision qui doit être fixé avant la confrontation au test.

---

## Modèle Retenu

>    **À compléter après ré-exécution du notebook** avec la nouvelle méthodologie (seuil calibré sur train).

| | |
|---|---|
| **Algorithme** | Sélectionné automatiquement (meilleur F1 test avec seuil calibré sur train) |
| **Optimisation** | RandomizedSearchCV (5-fold stratifié) |
| **Rééquilibrage** | `class_weight` ou `scale_pos_weight` (optimisé par la recherche) |
| **Seuil de décision** | Calibré par balayage F1-score sur prédictions out-of-fold (train) |
| **Pipeline** | StandardScaler + OneHotEncoder → Classifieur |

---

## Performance sur le Jeu de Test

> Les métriques finales seront disponibles après ré-exécution (ÉTAPE 8 du notebook).

### Matrice de Confusion — Interprétation Métier

- **TP (Vrais Positifs)** : churners détectés → ciblage rétention
- **FN (Faux Négatifs)** : churners manqués → coût critique (perte de revenu)
- **FP (Faux Positifs)** : fausses alertes → coût faible (action marketing inutile)
- **TN (Vrais Négatifs)** : non-churners correctement identifiés

---

## Validation de Généralisation

- **Nested cross-validation** (5 folds externes × 5 folds internes) sur `X_train` uniquement
- Confirme l'absence de surapprentissage si l'écart entre F1 test et F1 nested CV est modéré
- L'écart positif éventuel s'explique par l'optimisation du seuil (non appliquée dans la nested CV)

---

## Fichier Exporté

`models/finetunes.joblib` contient :
- Pipeline complet (préprocesseur + classifieur)
- Seuil de décision optimal (calibré sur prédictions out-of-fold du train)
- Hyperparamètres optimaux
- Métriques de performance (évaluées sur le jeu de test)

```python
import joblib
data = joblib.load("models/finetunes.joblib")
pipeline = data['pipeline']
threshold = data['threshold']
probas = pipeline.predict_proba(X_new)[:, 1]
predictions = (probas >= threshold).astype(int)
```
