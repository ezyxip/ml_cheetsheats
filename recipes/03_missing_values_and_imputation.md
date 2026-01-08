```markdown
# 03. Пропуски: стратегии и импутация

Цель: понять, где и почему пропуски, выбрать стратегию обработки (удаление / логическая замена / импутация) и реализовать её без утечек.

---

## 1. Чек‑лист работы с пропусками

1. Найти признаки с пропусками, посчитать долю пропусков.
2. Понять возможную причину пропусков (по данным/логике задачи).
3. Решить по каждому признаку:
   - логическая замена,
   - удаление строк/столбцов,
   - простая импутация (mean/median/mode),
   - продвинутая импутация (KNN, MICE).
4. Реализовать импутацию **только по train** (через `sklearn` пайплайн).
5. Проверить, как изменилась структура данных (распределения, связи).

---

## 2. Быстрая диагностика пропусков

```python
# Сколько пропусков в каждом столбце
df.isna().sum().sort_values(ascending=False)

# Доля пропусков
missing_frac = df.isna().mean().sort_values(ascending=False)
print(missing_frac)

# Строки с пропусками
df_with_na = df[df.isna().any(axis=1)]
df_with_na.head()
```

---

## 3. Основные стратегии

### 3.1. Логическая замена

Когда пропуск = отдельная “ситуация” с понятным смыслом.

Примеры:
- пропуск в признаке “количество детей” → 0 (если уверены, что NaN = 0),
- пропуск в “тип документа” → “Unknown”,
- пропуск в “доход” → возможно, отдельная категория “Не указано” (для моделей, умеющих категории).

```python
# Пример: логическая замена для числового признака
df["children"] = df["children"].fillna(0)

# Пример: логическая замена для категориального признака
df["doc_type"] = df["doc_type"].fillna("Unknown")
```

Важно: логическая замена требует обоснования в предметной области.

---

### 3.2. Удаление (Drop)

1. Удалить признаки (столбцы), где пропусков “слишком много”:
   - например, > 60–80% пропусков и признак не критичен.
2. Удалить строки, если:
   - пропусков мало и удаление не сильно уменьшит датасет,
   - или строка испорчена по многим признакам.

```python
# Удаление столбцов с долей пропусков > 0.8
missing_frac = df.isna().mean()
cols_to_drop = missing_frac[missing_frac > 0.8].index
df = df.drop(columns=cols_to_drop)

# Удаление строк с пропусками в конкретных колонках
df = df.dropna(subset=["col1", "col2"])
```

---

## 4. Простая импутация (SimpleImputer)

Используем `sklearn.impute.SimpleImputer`.

### 4.1. Импутация числовых признаков

Обычно:
- медиана (`strategy="median"`) — устойчивее к выбросам,
- среднее (`strategy="mean"`) — если распределение близко к нормальному.

```python
from sklearn.impute import SimpleImputer

num_cols = ["age", "income"]  # пример

num_imputer = SimpleImputer(strategy="median")
df[num_cols] = num_imputer.fit_transform(df[num_cols])
```

### 4.2. Импутация категориальных признаков

Чаще всего:
- `most_frequent` — мода,
- `constant` с некоторым значением (например, `"Unknown"`).

```python
cat_cols = ["city", "gender"]

cat_imputer = SimpleImputer(strategy="most_frequent")
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# или константа
cat_imputer_const = SimpleImputer(strategy="constant", fill_value="Unknown")
df[cat_cols] = cat_imputer_const.fit_transform(df[cat_cols])
```

---

## 5. KNNImputer (импутация по похожим объектам)

Идея: заменить пропуск средним/медианой среди ближайших соседей (k‑NN) по другим признакам.

```python
from sklearn.impute import KNNImputer

num_cols = ["age", "income", "score"]

knn_imputer = KNNImputer(
    n_neighbors=5,          # число соседей
    weights="uniform",      # или "distance"
)

df[num_cols] = knn_imputer.fit_transform(df[num_cols])
```

Особенности:
- работает только с числовыми признаками (категориальные нужно закодировать заранее),
- чувствителен к масштабам признаков → лучше сначала `StandardScaler`, но через пайплайн.

---

## 6. MICE / IterativeImputer

MICE (Multiple Imputation by Chained Equations) ≈ `sklearn.impute.IterativeImputer`.

Идея: каждый признак с пропусками предсказывается по остальным признакам (циклически).

```python
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

num_cols = ["age", "income", "score"]

iter_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=42,
)

df[num_cols] = iter_imputer.fit_transform(df[num_cols])
```

Особенности:
- тяжёлый по времени,
- может дать более “умные” замены, но легко переусложнить,
- для экзамена достаточно знать идею и базовый шаблон.

---

## 7. Импутация без утечки: через Pipeline / ColumnTransformer

**Важное правило**: все `fit` (в т.ч. импутация) делаются **только на train**.

Лучший способ — использовать `Pipeline` и `ColumnTransformer`.

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Разбиваем данные
X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Пайплайн для числовых признаков
num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# 3. Пайплайн для категориальных признаков
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

# 4. Общий препроцессор
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# 5. Полный пайплайн (препроцессинг + модель)
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=42)),
])

# 6. Обучение — все imputer'ы и encoder'ы обучаются ТОЛЬКО на X_train
clf.fit(X_train, y_train)

# 7. Предсказание
y_pred = clf.predict(X_test)
```

---

## 8. Проверка влияния импутации на распределения

### 8.1. Визуально: гистограммы до/после

```python
col = "income"

# До импутации
df_before = df_original.copy()  # копия до любых операций
df_after = df.copy()            # после импутации

plt.figure(figsize=(8, 4))
sns.kdeplot(df_before[col].dropna(), label="до импутации")
sns.kdeplot(df_after[col].dropna(), label="после импутации")
plt.legend()
plt.title(col)
plt.show()
```

### 8.2. KL‑дивергенция (Kullback–Leibler) между распределениями

Оценка, насколько изменилось распределение признака.

```python
import numpy as np
from scipy.stats import entropy  # KL‑дивергенция

def kl_divergence(p, q, eps=1e-9):
    """KL(P || Q) для дискретных распределений, p и q — массивы вероятностей."""
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()
    q = q / q.sum()
    return entropy(p, q)  # по умолчанию — натуральный логарифм

col = "income"

# Оценим распределения на гистограмме
hist_before, bin_edges = np.histogram(
    df_before[col].dropna(), bins=30, density=True
)
hist_after, _ = np.histogram(
    df_after[col].dropna(), bins=bin_edges, density=True
)

kl = kl_divergence(hist_before, hist_after)
print(f"KL-дивергенция для {col}: {kl:.4f}")
```

Интерпретация:
- KL ≈ 0 → распределение почти не изменилось.
- Большое значение KL → импутация сильно деформировала распределение.

---

## 9. Частые паттерны / советы

1. **Числовые признаки**:
   - медиана — безопасный дефолт,
   - mean — если распределение симметричное, без тяжёлых хвостов.
2. **Категориальные признаки**:
   - `most_frequent` + при необходимости отдельная категория “Unknown”.
3. **Очень много пропусков** (например, > 80%):
   - чаще признак удаляют, если нет сильных доменных аргументов его сохранить.
4. **KNNImputer / IterativeImputer**:
   - использовать, если есть время и много “информативных” признаков,
   - обязательно через пайплайн (с масштабированием для KNN).
5. **Никогда**:
   - не считать импутер (`fit`) на всём датасете перед `train_test_split`,
   - не использовать таргет в импутации признаков.

---

## 10. Минимальный шаблон: простая импутация для числовых и категориальных

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

model = Pipeline([
    ("preprocess", preprocess),
    ("reg", RandomForestRegressor(random_state=42)),
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

Этот шаблон можно адаптировать под классификацию, сложные имутеры и т.д.
```