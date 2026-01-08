```markdown
# 07. Feature Engineering для временных рядов

Цель: аккуратно подготовить признаки для временных рядов так, чтобы модель видела только прошлое, а не будущее, и при этом использовала лаги, разности, скользящие статистики и сезонность.

---

## 1. Общий чек‑лист по временным рядам

1. Привести данные к виду:
   - есть столбец с датой/временем (`datetime`),
   - либо индекс — DatetimeIndex.
2. Отсортировать по времени (и по id, если много рядов).
3. Сделать `train/test split` **по времени**, без перемешивания.
4. В кросс‑валидации использовать `TimeSeriesSplit` или аналоги.
5. Генерировать признаки:
   - лаги (`shift`),
   - разности (`diff`),
   - скользящие статистики (`rolling`),
   - сезонные / календарные (`month`, `dayofweek`, `hour`, и т.п.).
6. Строгий запрет:
   - никакого использования будущих значений для признаков (даже в агрегатах).
7. Масштабирование/encoding:
   - `fit` скейлеров/энкодеров только на train,
   - потом `transform` на test и на будущих точках.

---

## 2. Подготовка временного индекса

```python
import pandas as pd

df = pd.read_csv("data.csv")

# Преобразуем столбец в datetime
df["date"] = pd.to_datetime(df["date"], dayfirst=True)  # dayfirst при необходимости

# Сортировка по времени (и, при наличии нескольких рядов, по id)
df = df.sort_values(["id", "date"])  # если один ряд, можно просто ["date"]

# Установка индекса по дате (опционально)
df = df.set_index("date")
df.head()
```

---

## 3. Train/Test split “по времени”

### 3.1. Разделить по дате

```python
# Допустим, хотим train до 2022-01-01, остальное — test
split_date = "2022-01-01"

train = df.loc[df.index < split_date].copy()
test = df.loc[df.index >= split_date].copy()

print(train.shape, test.shape)
```

### 3.2. Разделить по последним N точкам

```python
test_size = 100  # последних 100 наблюдений на test

train = df.iloc[:-test_size].copy()
test = df.iloc[-test_size:].copy()
```

**Главное**:
- **не** использовать `train_test_split(shuffle=True)` для временных рядов,
- никакой перемешки в процессе.

---

## 4. Кросс‑валидация: TimeSeriesSplit

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=["target"])
y = df["target"]

tscv = TimeSeriesSplit(
    n_splits=5,         # количество фолдов
    test_size=None,     # можно задать явно, либо пусть делит сам
    # gap=0             # при необходимости пропуск между train и test
)

model = RandomForestRegressor(random_state=42)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")

scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print("CV MAE:", -scores.mean())
```

TimeSeriesSplit гарантирует:
- train‑фолды всегда “раньше” во времени, чем test‑фолды.

---

## 5. Лаги (Lags)

Лаг — значение признака (или таргета) в прошлом шаге(шаги).

### 5.1. Простой лаг для одного ряда

```python
# df с индексом по дате, столбец 'target'
df["lag_1"] = df["target"].shift(1)
df["lag_2"] = df["target"].shift(2)
df["lag_7"] = df["target"].shift(7)    # лаг на неделю (если дневные данные)

# Удаляем строки, где лаги NaN (самое начало ряда)
df_lagged = df.dropna().copy()
```

### 5.2. Лаги для нескольких рядов (по id)

```python
# Если есть несколько временных рядов по клиентам/товарам и т.п.
df["lag_1"] = df.groupby("id")["target"].shift(1)
df["lag_7"] = df.groupby("id")["target"].shift(7)

df_lagged = df.dropna().copy()
```

**Важно**:
- `shift(1)` — всегда “смотрим назад” (t−1),
- ни в коем случае не `shift(-1)` (это будущее).

---

## 6. Разности (differences)

Разность помогает убрать тренд и сделать ряд более стационарным:
- `diff_1 = y_t − y_{t−1}`,
- сезонная разность: `y_t − y_{t−season}`.

### 6.1. Первая разность

```python
df["diff_1"] = df["target"].diff(1)          # y_t - y_{t-1}
df = df.dropna().copy()
```

Для нескольких рядов:

```python
df["diff_1"] = df.groupby("id")["target"].diff(1)
df = df.dropna().copy()
```

### 6.2. Сезонная разность (например, дневной ряд с недельной сезонностью)

```python
season = 7   # сезонность = неделя (для дневных данных)
df["diff_season"] = df["target"].diff(season)
df = df.dropna().copy()
```

---

## 7. Скользящие статистики (rolling)

Скользящие средние/стд/макс и т.п. по окну из последних N наблюдений.

### 7.1. Скользящее среднее и стандартное отклонение

```python
window = 7   # окно в 7 дней, если данные дневные

df["roll_mean_7"] = df["target"].rolling(window=window, min_periods=1).mean()
df["roll_std_7"]  = df["target"].rolling(window=window, min_periods=1).std()
```

**Чтобы не залезать в будущее**, часто используют сдвиг:

```python
df["roll_mean_7"] = df["target"].rolling(window=7, min_periods=1).mean().shift(1)
df["roll_std_7"]  = df["target"].rolling(window=7, min_periods=1).std().shift(1)
```

Так модель на момент t использует статистику по [t−7, …, t−1], а не включая текущий t.

### 7.2. Rolling для нескольких рядов (по id)

```python
df["roll_mean_7"] = (
    df.groupby("id")["target"]
      .rolling(window=7, min_periods=1)
      .mean()
      .shift(1)
      .reset_index(level=0, drop=True)
)
```

---

## 8. Сезонность и календарные признаки

Для DatetimeIndex или столбца с датой (`datetime64[ns]`):

```python
dt = df.index.to_series()  # если дата в индексе
# или dt = df["date"]      # если дата в колонке

df["year"]       = dt.dt.year
df["month"]      = dt.dt.month
df["day"]        = dt.dt.day
df["dayofweek"]  = dt.dt.dayofweek   # 0=понедельник, 6=воскресенье
df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
df["dayofyear"]  = dt.dt.dayofyear

df["is_month_start"] = dt.dt.is_month_start.astype(int)
df["is_month_end"]   = dt.dt.is_month_end.astype(int)
```

Если есть часы/минуты:

```python
df["hour"]       = dt.dt.hour
df["minute"]     = dt.dt.minute
df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
```

### 8.1. Циклическое кодирование (sin/cos)

Для периодических признаков (день недели, месяц) разумно использовать sin/cos:

```python
import numpy as np

# День недели: 0..6
df["dow"] = dt.dt.dayofweek

df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

# Месяц: 1..12
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

Так модель понимает, что понедельник (0) “близок” к воскресенью (6), а декабрь — к январю.

---

## 9. Масштабирование и кодирование в задачах временных рядов

Правило то же, что и всегда, но особенно важно:

- все `Scaler`/`Encoder`/`Imputer` **fit только на train**,
- для валидации используем TimeSeriesSplit,
- никаких `.fit` на всём датасете.

### 9.1. Пример пайплайна для временного ряда

Допустим, у нас уже есть DataFrame `df_feat` с признаками (включая лаги и т.п.) и таргетом.

```python
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import numpy as np

X = df_feat.drop(columns=["target"])
y = df_feat["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

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

model = Pipeline([
    ("preprocess", preprocess),
    ("rf", RandomForestRegressor(random_state=42)),
])

tscv = TimeSeriesSplit(n_splits=5)

from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=tscv, scoring="neg_mean_absolute_error")
print("CV MAE:", -scores.mean())
```

---

## 10. Минимальный пример: из ряда — в supervised‑датасет

### 10.1. Одномерный ряд: предсказываем `target_t` по прошлым значениям

```python
import pandas as pd

# df: индекс = дата, столбец "target"
df = df.sort_index()

# Создаем лаги
df["lag_1"] = df["target"].shift(1)
df["lag_2"] = df["target"].shift(2)
df["lag_3"] = df["target"].shift(3)

# Скользящее среднее
df["roll_mean_3"] = df["target"].rolling(window=3, min_periods=1).mean().shift(1)

# Календарные признаки
dt = df.index.to_series()
df["month"] = dt.dt.month
df["dow"]   = dt.dt.dayofweek

# Убираем NaN из-за лагов/rolling
df = df.dropna().copy()

X = df.drop(columns=["target"])
y = df["target"]

# Разделение по времени: последние 100 наблюдений на test
test_size = 100
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 10.2. Прогноз на будущее

Чтобы сделать прогноз на будущий момент:
1. Взять последние значения признаков (включая лаги).
2. Сгенерировать “новую строку” для времени t+1:
   - обновить календарные признаки (дата, месяц, день недели),
   - использовать известные на будущее exogenous‑признаки,
   - лаги — это просто прошлые значения `target` (и других фичей).
3. Подать эту строку в модель.

(Для экзамена достаточно понимать идею, детальная реализация — отдельная тема.)

---

## 11. Чек‑лист “без утечки” для временных рядов

1. **Сначала сортировка по времени**, потом всё остальное.
2. `train/test split` — только “слева‑на‑право” по времени.
3. В кросс‑валидации:
   - использовать `TimeSeriesSplit` (или аналог),
   - не использовать `shuffle=True`.
4. Все признаки (лаги, rolling, разности, агрегаты) должны:
   - зависеть только от значений **до текущего момента**,
   - для rolling — добавлять `.shift(1)` при необходимости.
5. Групповые агрегаты (`groupby`) по объектам/клиентам:
   - если учитывают несколько временных точек → делать их на “прошлых” данных (через `expanding`, `rolling`, `shift`, а не по всему датасету).
6. Масштабирование и энкодинг:
   - `.fit` — только на train,
   - `.transform` — на test и на будущих данных.
```