# Rapport de Modélisation — Scoring de Churn Client

## Contexte

Prédiction du churn (résiliation) des clients d'un opérateur télécom à partir du dataset Telco Customer Churn. L'objectif est d'identifier les clients à risque pour déclencher des actions de rétention ciblées.

- **Dataset** : 7 043 clients, 20 variables (démographiques, services souscrits, facturation)
- **Classe cible** : `Churn` (Yes/No) — ~26 % de churners (déséquilibre modéré)
- **Split** : 80/20 stratifié (`RANDOM_STATE=1204`)

---

## Cheminement Analytique

Le projet a suivi une démarche en trois phases, chacune consignée dans un notebook dédié :

### Phase 1 — Exploration des données (Notebook 01)

L'analyse exploratoire a permis de comprendre la structure des données et d'identifier les leviers du churn :

- **Qualité des données** : dataset quasi complet (11 lignes supprimées sur 7 043), pas de doublons, pas d'outliers aberrants
- **Variables discriminantes identifiées** : `Contract` (contrats mensuels = risque élevé), `InternetService` (fibre = risque élevé), `tenure` (ancienneté = protection), `PaymentMethod` (chèque électronique = signal de risque)
- **Multicolinéarité détectée** : forte corrélation entre `TotalCharges` et `tenure` → création du ratio `ChargeRatio` en feature engineering
- **Déséquilibre de la cible** : ~26% de churners → nécessité de métriques adaptées (F1, AUC) et d'un rééquilibrage des classes

### Phase 2 — Modèles baseline (Notebook 02)

Trois modèles ont été construits avec des hyperparamètres par défaut pour établir un plancher de performance :

| Modèle | F1-score | Recall | ROC AUC |
|--------|----------|--------|---------|
| Régression Logistique | 0.601 | 53.7% | 0.860 |
| Random Forest | 0.580 | 51.1% | 0.841 |
| XGBoost | 0.572 | 51.6% | 0.831 |

**Diagnostic** : les trois modèles convergent sur un recall insuffisant (~52%). Près de la moitié des churners échappent à la détection. Trois causes identifiées : seuil de décision à 0.5 inadapté au déséquilibre, aucune pondération des classes, hyperparamètres par défaut.

### Phase 3 — Finetuning et sélection (Notebook 03)

Les trois leviers identifiés ont été actionnés simultanément :

1. **Tuning des hyperparamètres** via GridSearchCV / RandomizedSearchCV (320-400 fits par modèle)
2. **Rééquilibrage des classes** : `class_weight='balanced'` (Rég. Log., RF), `scale_pos_weight` (XGBoost)
3. **Calibrage du seuil de décision** sur les prédictions out-of-fold du train (maximisation du F1-score)

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

| | |
|---|---|
| **Algorithme** | Sélectionné automatiquement (meilleur F1 test avec seuil calibré sur train) |
| **Optimisation** | RandomizedSearchCV (5-fold stratifié) |
| **Rééquilibrage** | `class_weight` ou `scale_pos_weight` (optimisé par la recherche) |
| **Seuil de décision** | Calibré par balayage F1-score sur prédictions out-of-fold (train) |
| **Pipeline** | StandardScaler + OneHotEncoder → Classifieur |

---

## Progression du Recall — Fil conducteur du projet

| Étape | Recall approx. | Gain | Levier utilisé |
|-------|:-:|:-:|---------------|
| Baseline (seuil 0.5, pas de pondération) | ~52% | — | Configuration par défaut |
| + Tuning hyperparamètres | ~60% | +8 pts | GridSearchCV / RandomizedSearchCV |
| + Rééquilibrage des classes | ~65% | +5 pts | `class_weight='balanced'` / `scale_pos_weight` |
| + Calibrage du seuil | ~75% | +10 pts | Seuil optimal sur prédictions out-of-fold |

Le calibrage du seuil est le levier le plus puissant — un constat important qui précède toute recherche d'algorithmes plus complexes.

---

## Performance sur le Jeu de Test

> Les métriques finales exactes sont disponibles après exécution de l'ÉTAPE 8 du Notebook 03. Le modèle est sélectionné automatiquement selon le meilleur F1-score test.

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

## Variables Clés du Churn

Les trois modèles convergent sur les mêmes variables discriminantes, renforçant la fiabilité des insights :

| Variable | Impact | Action recommandée |
|----------|--------|-------------------|
| `Contract` (Month-to-month) | Risque très élevé | Proposer des engagements avec avantages |
| `InternetService` (Fiber optic) | Risque élevé | Améliorer compétitivité prix/qualité fibre |
| `tenure` (faible) | Risque élevé | Programme d'accueil renforcé pour les nouveaux clients |
| `OnlineSecurity` / `TechSupport` (absent) | Risque modéré | Bundler les services de sécurité/support |
| `PaymentMethod` (Electronic check) | Signal de risque | Inciter au prélèvement automatique |

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
