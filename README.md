=======
# Scoring de churn client (Classification)

## 1. Contexte métier & objectif

L'opérateur télécom européen **TelcoWave** (mobile + fibre) fait face à un taux de résiliation d'environ **26 %**. Acquérir un nouveau client coûte 5 à 7 fois plus cher que d'en retenir un existant : chaque point de churn évité a un impact direct sur le chiffre d'affaires.

**Objectif** — Construire un modèle de **scoring** qui estime, pour chaque client, la probabilité qu'il résilie son abonnement au prochain trimestre. Ce score permet de :

- **Prioriser** les actions de rétention sur les clients les plus à risque (top K %).
- **Optimiser le budget** marketing en concentrant les offres sur les déciles à fort lift.
- **Anticiper** les départs et agir en amont (offres de fidélisation, accompagnement).

---

## 2. Description des données

| Élément | Détail |
|---------|--------|
| **Source** | `data/Telco-Customer-Churn.csv` |
| **Lignes** | 7 043 clients |
| **Colonnes** | 21 (20 features + 1 cible) |
| **Cible** | `Churn` — *Yes / No* (binaire) |
| **Déséquilibre** | ≈ 26 % de churners (modéré) |

### Variables disponibles

| Catégorie | Colonnes |
|-----------|----------|
| **Identifiant** | `customerID` |
| **Démographiques** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Ancienneté** | `tenure` (mois) |
| **Services souscrits** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Contrat & facturation** | `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Montants** | `MonthlyCharges`, `TotalCharges` |

### Qualité

- **Valeurs manquantes** : 11 lignes où `TotalCharges` est vide (clients à tenure = 0) → supprimées.
- **Doublons** : aucun.
- **Outliers** : pas de valeur aberrante détectée (confirmé par IQR).
- **Feature engineering** : création de `ChargeRatio = TotalCharges / MonthlyCharges` (proxy de fidélité, décorrélé de `tenure`).

---

## 3. Installation

### Prérequis

- Python ≥ 3.10

### Mise en place

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd "Scoring de churn client (Classification)"

# Créer et activer l'environnement virtuel
python -m venv env
# Windows
env\Scripts\activate
# Linux / macOS
source env/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## 4. Reproduire l'analyse

Exécuter les notebooks dans l'ordre depuis Jupyter :

```bash
jupyter lab
```

| Étape | Notebook | Contenu |
|:-----:|----------|---------|
| **1** | `notebooks/01_data_exploration.ipynb` | Chargement, nettoyage, EDA univariée/bivariée, tests Chi², corrélations, synthèse qualité |
| **2** | `notebooks/02_baseline_models.ipynb` | Préparation (pipeline `StandardScaler` + `OneHotEncoder`), entraînement Logistic Regression / Random Forest / XGBoost avec hyperparamètres par défaut, comparaison, export `models/baseline.joblib` |
| **3** | `notebooks/03_finetuned_model.ipynb` | GridSearchCV & RandomizedSearchCV, rééquilibrage (`class_weight`, `scale_pos_weight`), calibration du seuil de décision, évaluation finale sur le jeu test, export `models/finetunes.joblib` |

### Inférence sur de nouvelles données

```python
import joblib

data = joblib.load("models/finetunes.joblib")
pipeline  = data["pipeline"]
threshold = data["threshold"]

probas      = pipeline.predict_proba(X_new)[:, 1]
predictions = (probas >= threshold).astype(int)
```

---

## 5. Résultats

### Baselines (seuil = 0.5, sans rééquilibrage)

| Modèle | F1-score | Recall | ROC AUC |
|--------|:--------:|:------:|:-------:|
| Régression logistique | 0.601 | 53.7 % | 0.860 |
| Random Forest | 0.580 | 51.1 % | 0.841 |
| XGBoost | 0.572 | 51.6 % | 0.831 |

**Diagnostic** : recall ≈ 52 % insuffisant → 1 churner sur 2 passe entre les mailles du filet.

### Après finetuning

Trois leviers activés simultanément :

| Levier | Gain recall |
|--------|:-----------:|
| Optimisation hyperparamètres (GridSearchCV / RandomizedSearchCV) | +8 pts |
| Rééquilibrage des classes (`class_weight` / `scale_pos_weight`) | +5 pts |
| Calibration du seuil de décision | +10 pts |

**Recall final ≈ 75 %** — le modèle identifie désormais 3 churners sur 4.

### Décision de seuil & top K %

- Le **seuil optimal** est calibré sur les prédictions out-of-fold du jeu d'entraînement en maximisant le F1-score (balayage 0.1 → 0.9).
- L'analyse par **déciles** (lift / gain cumulé) montre que cibler le **top 20–30 %** des clients les plus risqués capture la majorité des churners, pour un budget marketing maîtrisé.

### Variables les plus discriminantes (consensus des 3 modèles)

| Variable | Impact | Action recommandée |
|----------|--------|-------------------|
| `Contract` (Month-to-month) | Très fort | Proposer des engagements avec avantages |
| `InternetService` (Fiber optic) | Fort | Améliorer la compétitivité prix/qualité fibre |
| `tenure` (faible) | Fort | Renforcer l'onboarding des nouveaux clients |
| `OnlineSecurity` / `TechSupport` (absent) | Modéré | Offrir des bundles sécurité/support |
| `PaymentMethod` (Electronic check) | Signal | Encourager le prélèvement automatique |

---

## 6. Limites, risques & pistes d'amélioration

### Limites & risques

| Risque | Détail |
|--------|--------|
| **Fuite de données (data leakage)** | `TotalCharges` est un proxy quasi-linéaire de `tenure` × `MonthlyCharges`. Le feature engineering (`ChargeRatio`) et un split strict train/test avant toute transformation atténuent le risque, mais une vigilance reste nécessaire. |
| **Biais de sélection** | Le jeu de données ne contient que les clients encore actifs ou récemment résiliés ; les clients partis bien avant la fenêtre d'observation sont absents, ce qui peut sous-estimer certains profils de churn. |
| **Biais socio-démographiques** | Les variables `gender` et `SeniorCitizen` sont présentes. Vérifier que le modèle ne discrimine pas sur ces critères (audit d'équité). |
| **Stabilité temporelle** | Le modèle est entraîné sur un snapshot statique. En production, un suivi de la dérive (data drift / concept drift) est indispensable. |
| **Seuil figé** | Le seuil optimal dépend du ratio coût d'acquisition / coût de rétention ; il doit être recalibré si la stratégie marketing évolue. |
---

## Arborescence du projet

```
├── README.md
├── requirements.txt
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── baseline.joblib
│   └── finetunes.joblib
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_models.ipynb
│   └── 03_finetuned_model.ipynb
├── reports/
│   └── figures/
│   └── model_report.md
└── utils/
    ├── data_prep.py
    ├── infer.py
    ├── metrics.py
    └── train.py
```
>>>>>>> develop
