```markdown
# 06. Общий Feature Engineering

Цель: научиться осознанно добавлять новые признаки и проверять, действительно ли они улучшают модель, а не просто усложняют её.

---

## 1. Общий чек‑лист по Feature Engineering

1. Сделать **baseline**:
   - простая модель,
   - минимум препроцессинга,
   - без сложных фичей.
2. По EDA понять:
   - какие зависимости есть,
   - какие признаки явно “просятся” к преобразованию.
3. Сгенерировать кандидатов:
   - арифметика (суммы/отношения/разности),
   - признаки из предметной области (формулы),
   - агрегаты по группам (`groupby`),
   - биннинг числовых признаков,
   - нелинейные преобразования (`log`, `sqrt`, `exp`).
4. Оценить:
   - через кросс‑валидацию,
   - через feature importance / permutation importance.
5. Оставить только **полезные** признаки.
6. Строго:
   - не использовать **будущее** или таргет при построении признаков,
   - строить все признаки **только по train внутри CV**, без утечек.

---

## 2. Baseline: с чего начать

### 2.1. Идея

Baseline — самая простая разумная модель:
- минимальный препроцессинг (импутация, OHE),
- без хитрых фичей,
- служит точкой отсчёта: новые фичи → лучше / хуже / без изменений.

### 2.2. Пример baseline для классификации

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
print("Baseline ROC-AUC:", scores.mean(), "+-", scores.std())
```

Дальше: добавляем новые признаки к `df` и повторяем процедуру — сравниваем ROC‑AUC.

---

## 3. Формулы предметной области

### 3.1. Идея

Использовать знания о задаче/физике/экономике:
- плотность = масса / объём,
- BMI = вес / рост²,
- выручка = цена × количество,
- маржа = (выручка − себестоимость) / выручка.

### 3.2. Пример

```python
# Пример: e-commerce
df["revenue"] = df["price"] * df["quantity"]
df["margin"] = (df["revenue"] - df["cost"]) / df["revenue"]

# Пример: BMI
df["BMI"] = df["weight_kg"] / (df["height_m"] ** 2)
```

Требования:
- формула должна быть **логичной** в контексте задачи;
- не использовать таргет (`target`) в формуле.

---

## 4. Арифметические комбинации признаков

### 4.1. Типовые комбинации

- Суммы / разности:
  - `total_income = salary + bonus`,
  - `debt_diff = total_debt - secured_debt`.
- Отношения (ratio):
  - `debt_to_income = debt / income`,
  - `price_per_unit = price / quantity`.
- Произведения:
  - `feature = f1 * f2` (интеракции признаков).
- Разности во времени (если не time series → аккуратно).

### 4.2. Примеры кода

```python
# Сумма
df["total_income"] = df["salary"] + df["bonus"]

# Отношение с защитой от деления на 0
eps = 1e-6
df["debt_to_income"] = df["debt"] / (df["income"] + eps)

# Произведение
df["area_price"] = df["area"] * df["price_per_m2"]
```

Проверка:
- сделать `.describe()` для новых фичей,
- наложить гистограммы / scatter с таргетом.

---

## 5. Агрегаты по группам (`groupby`)

Очень мощная техника: строим признаки на уровне “объектов” из статистик по “группам”.

Примеры:
- средний чек клиента по всем его покупкам,
- количество покупок клиента за всё время,
- средняя цена товара в категории,
- среднее поведение юзера по сессиям.

### 5.1. `groupby().transform()` (сохранение размера)

`transform` возвращает Series той же длины, что и исходный DataFrame → удобно для фичей.

```python
# Средний чек клиента
df["client_mean_amount"] = df.groupby("client_id")["amount"].transform("mean")

# Количество покупок клиента
df["client_purchase_count"] = df.groupby("client_id")["amount"].transform("count")

# Стандартное отклонение по клиенту
df["client_amount_std"] = df.groupby("client_id")["amount"].transform("std")
```

### 5.2. `groupby().agg()` + `merge`

Полезно, если хотим сжать данные (одна строка на клиента, товар и т.п.).

```python
# Построить таблицу клиентов
clients = df.groupby("client_id")["amount"].agg(
    client_mean_amount="mean",
    client_purchase_count="count",
    client_amount_sum="sum",
).reset_index()

# Затем подмерджить к “основному” датасету (если нужно)
df = df.merge(clients, on="client_id", how="left")
```

Важно:
- в задачах с **временными рядами** и **онлайн‑сценариях** нельзя использовать будущие транзакции для построения признаков (см. чек‑лист ниже).

---

## 6. Биннинг числовых признаков

### 6.1. Зачем

- Превращает числовой признак в категориальный (часто — порядковый):
  - например, возраст → “18–25”, “26–35”, “36–50”, “50+”.
- Может стабилизировать зависимость и сделать её более интерпретируемой.

### 6.2. `pd.cut` — равные интервалы

```python
# Возрастные группы по фиксированным интервалам
bins = [0, 25, 35, 50, 100]
labels = ["0-25", "26-35", "36-50", "50+"]

df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=True, include_lowest=True)

df["age_bin"].value_counts()
```

### 6.3. `pd.qcut` — по квантилям

```python
# Делим признак на 4 квантиля (примерно равные группы по числу объектов)
df["income_qbin"] = pd.qcut(df["income"], q=4, labels=False, duplicates="drop")
```

Особенности:
- `qcut` создаёт примерно одинаковое количество объектов в каждом бине,
- после биннинга можно применить OHE или OrdinalEncoding.

---

## 7. Нелинейные преобразования: log, sqrt, exp

### 7.1. log

Используем для:
- сильно скошенных распределений (скошенность вправо),
- положительных значений (доход, цена, количество).

```python
import numpy as np

col = "income"

# Защита от 0 (log(1 + x))
df[f"log_{col}"] = np.log1p(df[col])

# Сравнение распределений
df[[col, f"log_{col}"]].hist(bins=30, figsize=(10, 4))
```

### 7.2. sqrt

Иногда лог “слишком сильно” сжимает, sqrt — мягче:

```python
df["sqrt_income"] = np.sqrt(df["income"].clip(lower=0))
```

### 7.3. exp

Используется реже — в основном, чтобы “вернуть” значение после логарифма:
- если обучали модель на `log(target)`, то предсказание возвращаем через `np.expm1`.

```python
y_log = np.log1p(y)
# ...
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)   # обратно из лог‑масштаба
```

---

## 8. Внешние данные (external data)

### 8.1. Шаги

1. Загрузить external‑датасет (`df_ext`), где есть:
   - ключ (id, регион, дата и т.п.),
   - дополнительные признаки.
2. Проверить:
   - размер,
   - уникальность ключа (`.duplicated`).
3. Сделать `merge` с основным датасетом по ключу.
4. Пересчитать EDA / baseline.

### 8.2. Пример: доп.данные по регионам

```python
# df: основной датасет, содержит "region_id"
# df_ext: таблица с дополнительной инфой по регионам
#         ["region_id", "population", "gdp_per_capita", "unemployment_rate"]

df_ext = pd.read_csv("regions.csv")

# Проверка дубликатов ключа
df_ext["region_id"].duplicated().sum()  # должно быть 0

# Merge
df = df.merge(df_ext, on="region_id", how="left")
```

### 8.3. Временные ряды и внешние данные

Если ключ = дата:
- **нельзя** использовать данные из будущего (например, будущий курс валюты).
- Для каждого момента времени t используем только то, что было известно на момент t.

Пример (упрощённо):
- внешний датасет с курсом доллара по дням,
- целевой признак — что‑то, зависящее от курсов в прошлом,
- join по дате допустим, если на момент t курс уже был известен.

---

## 9. Проверка полезности фич: feature importance

### 9.1. Feature importance из деревьев / бустинга

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feat_names = X_train.columns

feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print(feat_imp.head(20))
```

Смотрим:
- попали ли новые фичи в топ,
- есть ли совсем неиспользуемые.

### 9.2. Permutation Importance (более честная оценка)

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    rf, X_test, y_test,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",   # или нужная метрика
)

perm_imp = pd.Series(result.importances_mean, index=feat_names).sort_values(ascending=False)
print(perm_imp.head(20))
```

---

## 10. Чек‑лист безопасности Feature Engineering

1. **Не использовать будущее**:
   - не смотреть на транзакции/события после момента предсказания,
   - в `groupby` для time series — аккуратно (лучше отдельная шпора по временнЫм фичам).
2. **Не использовать таргет** для построения фич:
   - кроме специальных методов (Target Encoding) с правильной CV‑схемой.
3. **Не смотреть на тест**:
   - не строить признаки, глядя на test set,
   - все операции (включая агрегации) должны быть “обучены” только на train.
4. **Добавлять фичи поэтапно**:
   - добавили набор фич → померили метрику через CV → оставили/выкинули.
5. **Следить за размерностью**:
   - много фич → риск переобучения,
   - использовать регуляризацию, importance, отбор признаков при необходимости.

---

## 11. Минимальный рецепт “добавили фичу — проверили”

```python
# 1. Baseline (до фич)
scores_base = cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc")
print("Baseline:", scores_base.mean())

# 2. Добавляем новые признаки в df
df["debt_to_income"] = df["debt"] / (df["income"] + 1e-6)

# Обновляем X
X_new = df.drop(columns=["target"])
X_train_new, X_test_new, y_train, y_test = train_test_split(
    X_new, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Обучаем ту же модель с новыми признаками
scores_new = cross_val_score(clf, X_train_new, y_train, cv=5, scoring="roc_auc")
print("With new feature:", scores_new.mean())
```

Если метрика стабильно выросла → фича, скорее всего, полезна.
```