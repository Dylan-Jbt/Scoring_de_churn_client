# Rapport de Modélisation — Scoring de Churn Client

## Contexte

Prédiction du churn (résiliation) des clients d'un opérateur télécom à partir du dataset Telco Customer Churn. L'objectif est d'identifier les clients à risque pour déclencher des actions de rétention ciblées.

- **Dataset** : 7 043 clients, 20 variables (démographiques, services souscrits, facturation)
- **Classe cible** : `Churn` (Yes/No) — ~26 % de churners (déséquilibre modéré)
- **Split** : 80/20 stratifié (`RANDOM_STATE=1204`)

---

## Modèle Retenu

| | |
|---|---|
| **Algorithme** | Random Forest |
| **Optimisation** | RandomizedSearchCV (80 iter × 5 folds) |
| **Rééquilibrage** | `class_weight='balanced'` |
| **Seuil de décision** | 0.48 (optimisé par balayage F1-score) |
| **Pipeline** | StandardScaler + OneHotEncoder → RandomForestClassifier |

---

## Performance sur le Jeu de Test (1 407 clients)

| Métrique | Valeur |
|----------|--------|
| **F1-score** | **0.6667** |
| **Recall** | 0.7754 (77.5 %) |
| **Precision** | 0.5847 (58.5 %) |
| **Accuracy** | 0.7939 (79.4 %) |
| **ROC AUC** | 0.8591 |

### Matrice de Confusion

|  | Prédit Non-Churn | Prédit Churn |
|--|:--:|:--:|
| **Réel Non-Churn** | 827 | 206 |
| **Réel Churn** | 84 | 290 |

- **290 churners détectés** sur 374 (taux de détection : 77.5 %)
- **84 churners manqués** (coût critique : perte de revenu)
- **206 faux positifs** (coût faible : action marketing inutile)

---

## Gains par rapport au Baseline

| Métrique | Baseline (NB02) | Finetuné (NB03) | Gain |
|----------|:--:|:--:|:--:|
| F1-score | 0.580 | 0.667 | +15 % |
| Recall | 51.1 % | 77.5 % | +26.4 pts |
| ROC AUC | 0.841 | 0.859 | +0.018 |

Le finetuning a permis de détecter ~100 churners supplémentaires sur le jeu de test.

---

## Validation de Généralisation

Nested cross-validation (5 folds externes × 5 folds internes) confirmant l'absence de surapprentissage. L'écart positif entre F1 test et F1 nested CV s'explique par l'optimisation du seuil non appliquée pendant la nested CV.

---

## Fichier Exporté

`models/finetunes.joblib` contient :
- Pipeline complet (préprocesseur + classifieur)
- Seuil de décision optimal (0.48)
- Hyperparamètres optimaux
- Métriques de performance

```python
import joblib
data = joblib.load("models/finetunes.joblib")
pipeline = data['pipeline']
threshold = data['threshold']
probas = pipeline.predict_proba(X_new)[:, 1]
predictions = (probas >= threshold).astype(int)
```
