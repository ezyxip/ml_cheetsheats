```markdown
# 12. Обучение и тюнинг гиперпараметров

Цель: уметь обучать модели, настраивать их гиперпараметры через кросс‑валидацию (GridSearch/RandomizedSearch) и выбирать корректную метрику `scoring`.

---

## 1. Базовый шаблон обучения

### 1.1. fit / predict / predict_proba

```python
# Для регрессии
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Для классификации (метки)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Для классификации (вероятности)
y_proba = model.predict_proba(X_test)[:, 1]  # вероятность класса 1
```

Пример с логистической регрессией:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
```

Всегда:
- обучаем только на `train`,
- оцениваем качество на `val`/`test`.

---

## 2. Кросс‑валидация в тюнинге

### 2.1. GridSearchCV (полный перебор)

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X, y)

print("Лучшие параметры:", grid.best_params_)
print("Лучший score:", grid.best_score_)
best_model = grid.best_estimator_
```

---

### 2.2. RandomizedSearchCV (случайный поиск)

Полный перебор бывает дорогим; случайный поиск берёт случайные комбинации гиперпараметров.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

param_dist = {
    "n_estimators": randint(100, 600),
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rand_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30,               # сколько случайных комбинаций проверить
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

rand_search.fit(X, y)
print("Лучшие параметры:", rand_search.best_params_)
print("Лучший score:", rand_search.best_score_)
best_model = rand_search.best_estimator_
```

---

### 2.3. GridSearch/RandomizedSearch с Pipeline

Важно: тюнить модель внутри `Pipeline`, чтобы не словить leakage.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])

param_grid = {
    "logreg__C": [0.01, 0.1, 1, 10],
    "logreg__penalty": ["l2"],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
```

Обрати внимание на синтаксис имени параметров:
- `step__param` (двойное подчёркивание).

---

## 3. Байесовская оптимизация (BayesOpt) — общая схема

Полно задавать код не обязательно, но важно понимать идею:

- вместо полного/случайного перебора → строится **модель** функции качества по гиперпараметрам;
- на каждом шаге эта модель подсказывает, какие гиперпараметры попробовать дальше (exploration vs exploitation).

Типичные библиотеки:

- `scikit-optimize` → `skopt.BayesSearchCV`,
- `optuna`,
- `hyperopt`,
- `bayes_opt` и др.

### 3.1. skopt.BayesSearchCV (общая идея)

```python
# pip install scikit-optimize
from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

rf = RandomForestClassifier(random_state=42)

search_spaces = {
    "n_estimators": (100, 600),
    "max_depth": (3, 20),
    "min_samples_split": (2, 10),
    "min_samples_leaf": (1, 5),
    "max_features": ["sqrt", "log2", None],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

opt = BayesSearchCV(
    estimator=rf,
    search_spaces=search_spaces,
    n_iter=30,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
)

opt.fit(X, y)
print(opt.best_params_, opt.best_score_)
```

(На экзамене обычно достаточно знать, что BayesOpt существует и чем отличается от Grid/Random.)

### 3.2. Optuna (очень кратко)

Идея:
- пишем функцию `objective(trial)`, которая:
  - берёт гиперпараметры `trial.suggest_*`,
  - обучает модель с CV,
  - возвращает метрику (с минусом, если нужно минимизировать ошибку).
- Optuna сама делает байесовскую оптимизацию.

---

## 4. Типовые сетки гиперпараметров

Ниже — разумные стартовые гриды (их можно сокращать/расширять под размер данных).

### 4.1. RandomForest

**Классификация**:

```python
param_grid_rf_clf = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True, False],
}
```

**Регрессия** (почти то же):

```python
param_grid_rf_reg = {
    "n_estimators": [100, 300, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
}
```

---

### 4.2. XGBoost (классификация)

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",   # при больших данных
    random_state=42,
)

param_grid_xgb = {
    "n_estimators": [200, 500],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_lambda": [1, 5, 10],
    "reg_alpha": [0, 0.1, 1],
}
```

---

### 4.3. LightGBM (классификация)

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    objective="binary",
    random_state=42,
)

param_grid_lgbm = {
    "n_estimators": [200, 500, 800],
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 63, 127],
    "max_depth": [-1, 5, 10],
    "subsample": [0.7, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "min_child_samples": [10, 20, 50],
}
```

---

### 4.4. CatBoost (классификация)

```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
)

param_grid_cat = {
    "depth": [4, 6, 8],
    "learning_rate": [0.03, 0.1],
    "n_estimators": [300, 500, 800],
    "l2_leaf_reg": [1, 3, 5, 10],
}
```

(У CatBoost тюнинг часто делают через встроенный `GridSearch` или внешние либы, но подход тот же.)

---

### 4.5. SVM (SVC)

```python
from sklearn.svm import SVC

svc = SVC(probability=True, random_state=42)

param_grid_svc = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["rbf"],      # можно добавить "linear"
    "gamma": ["scale", 0.01, 0.001],
}
```

Обязательно масштабирование (`StandardScaler` в Pipeline).

---

### 4.6. KNN

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

param_grid_knn = {
    "n_neighbors": [3, 5, 7, 11, 15],
    "weights": ["uniform", "distance"],
    "p": [1, 2],              # 1 = манхэттен, 2 = евклид
}
```

Тоже обязательно масштабирование признаков.

---

## 5. Как выбирать метрику для `scoring`

### 5.1. Регрессия

Наиболее популярные метрики:

- MAE: `neg_mean_absolute_error`
- MSE: `neg_mean_squared_error`
- RMSE: `neg_root_mean_squared_error` (или sqrt(MSE вручную)
- R²: `r2`

Важно: во многих метриках для регрессии `sklearn` использует **знак минус**:

- `scoring="neg_mean_absolute_error"` → **чем больше (менее отрицательно), тем лучше**.

Пример:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")
print("MAE:", -scores.mean())
```

Выбор:

- если важна абсолютная ошибка → MAE,
- если хотим сильнее штрафовать большие ошибки → MSE/RMSE,
- R² — для интерпретации доли объяснённой дисперсии.

---

### 5.2. Классификация (сбалансированная)

Основные:

- `accuracy` — доля правильных,
- `f1` — баланс Precision/Recall,
- `roc_auc` — площадь под ROC‑кривой.

Пример:

```python
GridSearchCV(..., scoring="accuracy")
GridSearchCV(..., scoring="f1")
GridSearchCV(..., scoring="roc_auc")
```

---

### 5.3. Классификация (сильный дисбаланс)

`accuracy` почти всегда бесполезна.

Используем:

- `roc_auc` — хорошая общая метрика,
- `average_precision` — PR‑AUC,
- `f1`, `f1_macro`, `f1_weighted`,
- `recall` — если важно не пропустить позитивные случаи.

Примеры:

```python
# ROC-AUC
GridSearchCV(..., scoring="roc_auc")

# PR-AUC (average precision)
GridSearchCV(..., scoring="average_precision")

# F1
GridSearchCV(..., scoring="f1")
```

---

### 5.4. Временные ряды

Выбор метрики как для обычной регрессии/классификации, но:

- важнее **валидировать по TimeSeriesSplit**,
- для прогноза значений чаще используют:
  - `neg_mean_absolute_error`,
  - `neg_mean_squared_error`,
  - иногда MAPE/SMAPE (несколько сложнее, часто реализуют сами).

---

## 6. Мини‑рецепт тюнинга “под ключ”

1. Выбрать модель и метрику:
   - регрессия → RF + `neg_mean_absolute_error`,
   - сбалансированная классификация → RF + `roc_auc` или `accuracy`,
   - дисбаланс → RF/бустинг + `roc_auc`/`average_precision`/`f1`.

2. Разбить данные:
   - train/test (отложенный test),
   - внутри train — CV (KFold/StratifiedKFold/GroupKFold/TimeSeriesSplit).

3. Собрать Pipeline:
   - препроцессинг (импутация, скейлинг, encoding),
   - модель.

4. Задать `param_grid` или `param_distributions`.

5. Запустить `GridSearchCV` или `RandomizedSearchCV`.

6. Взять `best_estimator_`, обучить на всём train, проверить на test.

Этот набор шагов покрывает типовой процесс обучения и тюнинга моделей на экзамене и в практических задачах.
```