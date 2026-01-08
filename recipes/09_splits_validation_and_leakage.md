```markdown
# 09. Разбиения, валидация и утечка данных

Цель: правильно делить данные на train/validation/test, выбирать тип кросс‑валидации под задачу и избегать утечки данных (data leakage).

---

## 1. Базовые разбиения

### 1.1. train_test_split

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

# Классификация: важно stratify (особенно при дисбалансе)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,         # 20% на test
    random_state=42,
    stratify=y,            # только для классификации
)
```

Регрессия → `stratify=None`:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
)
```

---

### 1.2. Отложенный test set (hold‑out)

Идея:

1. Разделить данные сразу на:
   - `train_val` (80–90%),
   - `test` (10–20%) — **отложенный и не трогаем**.
2. На `train_val` делать:
   - кросс‑валидацию,
   - подбор гиперпараметров,
   - выбор моделей.
3. В самом конце:
   - один раз запустить финальную модель на `test`,
   - получить честную оценку качества.

```python
X = df.drop(columns=["target"])
y = df["target"]

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # если классификация
)

# Дальше на X_train_val / y_train_val — CV, GridSearch и т.п.
```

---

## 2. Кросс‑валидация (CV)

### 2.1. Обычный k‑Fold (регрессия / сбалансированная классификация)

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=["target"])
y = df["target"]

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

model = RandomForestRegressor(random_state=42)

scores = cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
print("CV MAE:", -scores.mean(), "+-", scores.std())
```

---

### 2.2. Stratified k‑Fold (классификация, особенно при дисбалансе)

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

clf = RandomForestClassifier(random_state=42)

scores = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
print("CV ROC-AUC:", scores.mean(), "+-", scores.std())
```

Стратификация → в каждом фолде сохраняется распределение классов.

---

### 2.3. GroupKFold / GroupShuffleSplit (группы)

Использовать, когда есть **группы** (user_id, patient_id, company_id и т.п.), и нельзя, чтобы объекты из одной группы были и в train, и в val/test (иначе leakage по пользователю).

```python
from sklearn.model_selection import GroupKFold, cross_val_score

X = df.drop(columns=["target"])
y = df["target"]
groups = df["user_id"]   # один и тот же user_id не должен попадать и в train, и в val

gkf = GroupKFold(n_splits=5)

clf = RandomForestClassifier(random_state=42)

scores = cross_val_score(clf, X, y, cv=gkf.split(X, y, groups=groups), scoring="roc_auc")
print("GroupKFold ROC-AUC:", scores.mean(), "+-", scores.std())
```

---

### 2.4. TimeSeriesSplit (временные ряды)

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=["target"])
y = df["target"]

tscv = TimeSeriesSplit(
    n_splits=5,
    # можно задать test_size/gap при необходимости
)

model = RandomForestRegressor(random_state=42)

scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print("TimeSeriesSplit MAE:", -scores.mean(), "+-", scores.std())
```

Особенности:
- train‑фолды всегда раньше по времени, чем test‑фолды,
- нельзя использовать `shuffle=True`.

---

## 3. Как выбирать схему CV под задачу

### 3.1. Регрессия / классификация без особых ограничений

- **Данных много**, классы примерно сбалансированы:
  - `KFold(shuffle=True)`.
- **Классификация с дисбалансом**:
  - `StratifiedKFold(shuffle=True)`.

### 3.2. Есть естественные группы (пользователи, компании, пациенты)

- Использовать `GroupKFold`:
  - каждая группа целиком в одном фолде,
  - предотвращает утечку по пользователю/пациенту.

### 3.3. Временные ряды

- **Всегда**:
  - разбиения только “по времени” (no shuffle),
  - либо ручной split по дате,
  - либо `TimeSeriesSplit`.
- Никаких `StratifiedKFold`/`KFold(shuffle=True)`.

### 3.4. Очень маленький датасет

- Лучше больше фолдов (например, 5–10),
- Для экстремально малого — `LeaveOneOut` (дорого, но максимум информации),
- Но помнить про вариативность метрик.

---

## 4. Data Leakage (утечка данных)

### 4.1. Определение

Data leakage — использование **информации из валидaционных / тестовых / будущих данных** при обучении модели или подготовке признаков.

Итог:
- на трейне/валидации модель кажется “очень хорошей”,
- на реальных данных → качество резко падает.

---

### 4.2. Типичные кейсы утечки

1. **Масштабирование/импутация/кодирование на всём датасете** до split:
   - `scaler.fit(X)` на всех данных, затем `train_test_split`.
   - Итог: параметры скейлера “видят” распределение test.

2. **Imputer / encoder / PCA на всём датасете**:
   - `SimpleImputer.fit(df)` до разбиения,
   - `OneHotEncoder.fit(df)` до разбиения,
   - `PCA.fit(df)` на всём датасете.

3. **Target leakage в feature engineering**:
   - использование таргета для расчёта признаков (например, mean target per user),
     без корректной CV‑схемы,
   - использование будущих значений в временных рядах (например, средний чек клиента за все будущие транзакции).

4. **Использование test для выбора модели/гиперпараметров**:
   - многократное “подглядывание” в test и тюнинг под него,
   - test превращается в ещё один validation, нет честной оценки.

5. **Смешивание данных из разных временных периодов**:
   - train содержит более “старые” данные, val/test — более “новые”,
   - но в фичах используются агрегаты, посчитанные по всей истории, включая будущие периоды.

---

## 5. Как избежать утечки: пайплайны и правильный порядок

### 5.1. Золотое правило

**Любой** объект, у которого есть `.fit()` (scaler, imputer, encoder, PCA, модель и т.п.), должен:

- `fit` ТОЛЬКО на train (или train‑fold в CV),
- `transform`/`predict` на val/test.

Лучший способ это гарантировать — использовать `Pipeline` + `ColumnTransformer` + CV.

---

### 5.2. Пример: классификация с полноценным пайплайном

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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
scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="roc_auc")
print("CV ROC-AUC:", scores.mean())

# Финальное обучение и оценка на test
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

Все шаги препроцессинга “подстраиваются” только под train/фолды внутри CV, а не под весь датасет.

---

### 5.3. Пример: GridSearchCV без утечки

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
}

grid = GridSearchCV(
    estimator=clf,             # наш Pipeline
    param_grid=param_grid,
    cv=skf,
    scoring="roc_auc",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

print("Лучшие параметры:", grid.best_params_)
print("Лучший score:", grid.best_score_)

best_model = grid.best_estimator_

# Оценка на отложенном test set
y_pred = best_model.predict(X_test)
```

GridSearch:
- внутри использует CV,
- для каждого набора гиперпараметров **заново** делает `.fit()` пайплайна на train‑fold → без утечки.

---

### 5.4. Специальный случай: Target Encoding и утечка

Target/Mean Encoding **очень легко** делает утечку, если:

```python
# ПЛОХО: считаем средний таргет по категориям на всём датасете
mean_target = df.groupby("category")["target"].mean()
df["cat_te"] = df["category"].map(mean_target)
# дальше split → test уже “подсмотрел” в свои же таргеты
```

Правильно:

- использовать `category_encoders.TargetEncoder` **внутри** пайплайна,
- и обязательно CV (он сам внутри считает средние только по train‑fold).

```python
import category_encoders as ce
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=["target"])
y = df["target"]

te = ce.TargetEncoder(cols=["category"])
logreg = LogisticRegression(max_iter=1000)

pipe = Pipeline([
    ("te", te),
    ("model", logreg),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
print("CV ROC-AUC:", scores.mean())
```

---

## 6. Мини‑чек‑лист “Без утечки”

1. Разбить данные на `train` и `test` **в самом начале** и больше не трогать test до финала.
2. Любые операции с `.fit()`:
   - не вызывать до `train_test_split`,
   - всегда использовать внутри `Pipeline`/CV.
3. Использовать правильный тип CV:
   - `StratifiedKFold` для классификации,
   - `GroupKFold` при наличи групп (user_id и т.п.),
   - `TimeSeriesSplit` для временных рядов.
4. Не использовать таргет для построения фич (кроме специальных методов с корректной CV‑схемой).
5. Не тюнить гиперпараметры на test:
   - test — только один финальный прогон.
```