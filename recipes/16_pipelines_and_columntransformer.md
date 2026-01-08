```markdown
# 16. Полный sklearn‑пайплайн (Pipeline + ColumnTransformer)

Цель: собрать **один объект**, который:
- делает весь препроцессинг (импутация, масштабирование, кодирование),
- обучает модель,
- корректно используется в CV / GridSearch без утечек.

---

## 1. Общая схема пайплайна

1. Разделить признаки на:
   - числовые (`num_cols`),
   - категориальные (`cat_cols`).
2. Для числовых:
   - `SimpleImputer` (median/mean),
   - `StandardScaler` / другой скейлер.
3. Для категориальных:
   - `SimpleImputer` (most_frequent / constant),
   - `OneHotEncoder` (или другой энкодер).
4. Объединить ветки через `ColumnTransformer`.
5. Добавить модель как последний шаг в `Pipeline`.
6. Использовать этот пайплайн в:
   - `cross_val_score`,
   - `GridSearchCV` / `RandomizedSearchCV`.

---

## 2. Базовый шаблон: разделение признаков

```python
import pandas as pd

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("num_cols:", num_cols)
print("cat_cols:", cat_cols)
```

---

## 3. Пайплайн для классификации

### 3.1. Полный Pipeline + ColumnTransformer (пример с RandomForest)

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# 1. train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 2. Ветка для числовых признаков
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 3. Ветка для категориальных признаков
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# 4. ColumnTransformer: объединяем ветки
preprocess = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# 5. Полный пайплайн: препроцессинг + модель
clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42)),
])

# 6. Кросс-валидация
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")
print("CV ROC-AUC:", scores.mean(), "+-", scores.std())

# 7. Финальное обучение и оценка на test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
```

---

### 3.2. Тюнинг гиперпараметров модели и препроцессинга в одном GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    # гиперпараметры RandomForest
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    # гиперпараметры препроцессинга (пример)
    "preprocess__num__imputer__strategy": ["median", "mean"],
    # можно тюнить и OHE, и scaler при желании
}

grid = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1,
    verbose=1,
)

grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)
print("Лучший ROC-AUC (CV):", grid.best_score_)

best_clf = grid.best_estimator_
y_pred_test = best_clf.predict(X_test)
```

Обрати внимание на синтаксис:
- `step1__step2__param`:
  - `preprocess` → `num` → `imputer` → `strategy`,
  - `model` → `max_depth` и т.д.

---

## 4. Пайплайн для регрессии

### 4.1. Полный Pipeline (пример с RandomForestRegressor)

```python
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

reg = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(random_state=42)),
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(reg, X_train, y_train, cv=kf, scoring="neg_mean_absolute_error")
print("CV MAE:", -scores.mean(), "+-", scores.std())

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### 4.2. GridSearch для регрессии

```python
from sklearn.model_selection import GridSearchCV

param_grid_reg = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [None, 5, 10],
    "preprocess__num__imputer__strategy": ["median", "mean"],
}

grid_reg = GridSearchCV(
    estimator=reg,
    param_grid=param_grid_reg,
    cv=kf,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=1,
)

grid_reg.fit(X_train, y_train)
print("Лучшие параметры:", grid_reg.best_params_)
print("Лучший MAE (CV):", -grid_reg.best_score_)

best_reg = grid_reg.best_estimator_
y_pred_test = best_reg.predict(X_test)
```

---

## 5. Пайплайн для временных рядов (TimeSeriesSplit)

Основные принципы те же:
- препроцессинг + модель в `Pipeline`,
- но CV → `TimeSeriesSplit`,
- **без shuffle**, без `train_test_split(shuffle=True)`.

Предположим, у нас уже есть `df_feat` с:
- признаками (включая лаги/rolling/календарные),
- таргетом `target`,
- индекс — дата.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

X = df_feat.drop(columns=["target"])
y = df_feat["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

ts_model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(random_state=42)),
])

tscv = TimeSeriesSplit(n_splits=5)

scores = cross_val_score(ts_model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print("TimeSeries CV MAE:", -scores.mean(), "+-", scores.std())
```

**Важно**:
- все лаги/rolling/разности должны быть заранее построены так, чтобы не использовать будущее;
- `TimeSeriesSplit` не перемешивает данные, каждый фолд — “старое → новое”.

---

## 6. Объединение: как это выглядит в реальном проекте

1. В ноутбуке/скрипте:
   - делаем EDA,
   - определяем `num_cols`, `cat_cols`,
   - при необходимости строим lag/rolling/бинаризуем и т.п.
2. Собираем `ColumnTransformer`:
   - разные пайплайны для числовых / категориальных.
3. Собираем `Pipeline`:
   - `("preprocess", preprocess), ("model", <ваша_модель>)`.
4. Используем этот пайплайн везде:
   - `cross_val_score`,
   - `GridSearchCV` / `RandomizedSearchCV`,
   - `.fit` на train и `.predict` на test/production.
5. Сохраняем **весь пайплайн целиком** (через `joblib.dump`), а не только модель:
   - так в проде будет один объект, который делает и препроцессинг, и предсказание.

---

## 7. Мини‑шаблон “скопировать-вставить”

```python
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42)),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# CV
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")
print("CV ROC-AUC:", scores.mean())

# Fit + predict
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Этот шаблон можно адаптировать под любую модель (LogReg, XGBoost, LGBM, CatBoost) и под регрессию/TimeSeries, меняя только модель и CV‑схему.
```