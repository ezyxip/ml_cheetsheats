```markdown
# 05. Выбросы и аномалии

Цель: обнаружить выбросы/аномалии, понять — удалять, трансформировать или оставить, и уметь применять базовые алгоритмы (IQR, Z‑score, Boxplot, Isolation Forest, One‑Class SVM, DBSCAN).

---

## 1. Чек‑лист работы с выбросами / аномалиями

1. Визуализировать распределения:
   - гистограммы, boxplot, scatter (признаки vs таргет).
2. Оценить влияние выбросов:
   - искажают ли они масштаб/распределения/модель?
3. Для табличных задач:
   - попробовать логарифмирование/обрезку (winsorizing),
   - либо удалить явно ошибочные значения (например, возраст = 500).
4. Для задач “поиск аномалий” (без таргета):
   - обучить детектор (IsolationForest, One‑Class SVM, DBSCAN),
   - пометить аномалии и решать, как с ними работать (удалять/выделять).
5. Всегда:
   - все `fit` (в т.ч. детекторы аномалий) — только на train,
   - никогда не “подглядывать” в test.

---

## 2. Визуализация: Boxplot и логарифмирование

### 2.1. Boxplot для одного признака

```python
import matplotlib.pyplot as plt
import seaborn as sns

col = "income"

plt.figure(figsize=(6, 3))
sns.boxplot(x=df[col])
plt.title(f"Boxplot: {col}")
plt.show()
```

### 2.2. Boxplot с таргетом (регрессия)

```python
col = "income"
target_col = "target"

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x=col, y=target_col, alpha=0.5)
plt.title(f"{col} vs {target_col}")
plt.show()
```

### 2.3. Логарифмирование “тяжёлых хвостов”

Часто распределение признака “с хвостом” (до очень больших значений). Лог-преобразование помогает:

```python
import numpy as np

col = "income"

df[f"log_{col}"] = np.log1p(df[col])  # log(1 + x), устойчиво к нулям

# Сравнить распределения
sns.histplot(df[col], bins=50)
plt.title(col)
plt.show()

sns.histplot(df[f"log_{col}"], bins=50)
plt.title(f"log_{col}")
plt.show()
```

---

## 3. Простые правила: IQR и Z‑score

### 3.1. IQR (межквартильный размах)

Формулы:

- Q1 = 25‑й персентиль,
- Q3 = 75‑й персентиль,
- IQR = Q3 − Q1,
- “подозрительные” значения:  
  `x < Q1 − 1.5 * IQR` или `x > Q3 + 1.5 * IQR`.

```python
import numpy as np

col = "income"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
df_outliers = df[outliers_mask]
df_no_outliers = df[~outliers_mask]

print("Выбросов:", outliers_mask.sum())
```

**Дальше выбор:**
- удалить `df = df_no_outliers`,
- или обрезать значения до границ (“winsorize”):

```python
df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
```

---

### 3.2. Z‑score (стандартизованное отклонение)

Формула:

\[
z_i = \frac{x_i - \mu}{\sigma}
\]

Где:
- `μ` — среднее,
- `σ` — стандартное отклонение.

Обычно выбросы: `|z| > 3` (или 2,5).

```python
from scipy import stats
import numpy as np

col = "income"

z = np.abs(stats.zscore(df[col].dropna()))
threshold = 3

outliers_mask = z > threshold
print("Выбросов:", outliers_mask.sum())
```

**Минусы:**
- чувствителен к уже существующим выбросам,
- лучше для примерно нормально распределённых признаков.

---

## 4. Workflow для табличных задач (регрессия/классификация)

1. На этапе EDA:
   - построить boxplot / scatterplots,
   - оценить, какие значения явно **ошибочные** (отрицательный возраст, невозможные величины).
2. Для “очевидных ошибок”:
   - либо исправить (если понятен источник),
   - либо удалить строки.
3. Для экстремально больших, но возможных значений (например, очень высокий доход):
   - попробовать логарифмирование,
   - рассмотреть клиппинг (clip) на уровне разумных квантилей (1% / 99%).
4. Проверить влияние:
   - обучить простую модель “до” и “после” обработки выбросов,
   - сравнить метрики.

---

## 5. Isolation Forest — детекция аномалий

Isolation Forest — ансамбль деревьев, который “изолирует” аномалии за меньшее число разбиений.

### 5.1. Базовый пример

```python
from sklearn.ensemble import IsolationForest

features = ["feature1", "feature2", "feature3"]
X = df[features]

iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,   # доля аномалий (если знаем примерно)
    random_state=42,
)

iso.fit(X)

# Предсказание: 1 = нормальный, -1 = аномалия
labels = iso.predict(X)
df["anomaly_iso"] = (labels == -1).astype(int)

df["anomaly_iso"].value_counts()
```

### 5.2. Использование в supervised‑задаче

```python
# 1. Делим на train/test
from sklearn.model_selection import train_test_split

X_full = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Детектор аномалий обучаем ТОЛЬКО на train
iso = IsolationForest(
    n_estimators=100,
    contamination=0.01,
    random_state=42,
)

iso.fit(X_train[features])
train_labels = iso.predict(X_train[features])
test_labels = iso.predict(X_test[features])

X_train_clean = X_train[train_labels == 1]
y_train_clean = y_train[train_labels == 1]

# 3. Обучаем модель на “очищенных” данных и сравниваем с исходной
```

---

## 6. One‑Class SVM — ещё один детектор аномалий

One‑Class SVM учится описывать “нормальное” множество точек, остальное — аномалии.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

features = ["feature1", "feature2", "feature3"]
X = df[features]

ocsvm = Pipeline(steps=[
    ("scaler", StandardScaler()),        # SVM чувствителен к масштабу
    ("svm", OneClassSVM(
        kernel="rbf",
        nu=0.01,        # приблизительная доля аномалий (0 < nu <= 1)
        gamma="scale",
    ))
])

ocsvm.fit(X)

labels = ocsvm.predict(X)      # 1 = нормальный, -1 = аномалия
df["anomaly_svm"] = (labels == -1).astype(int)
df["anomaly_svm"].value_counts()
```

Особенности:
- часто тяжелее по времени, чем IsolationForest,
- требует масштабирования признаков,
- чувствителен к выбору `nu`, `gamma`.

---

## 7. DBSCAN и аномалии через кластеризацию

DBSCAN — алгоритм кластеризации, который:
- находит плотные области (кластеры),
- точки вне плотных областей помечает как шум (noise) → аномалии.

### 7.1. Основные понятия

- `eps` — радиус окрестности (ε‑окрестность).
- `min_samples` (часто `minPts`) — минимальное число точек в ε‑окрестности, чтобы точка была **core point**.
- **Core point** — точка с >= `min_samples` соседей в радиусе `eps`.
- **Border point** — точка, которая сама не core, но лежит в eps‑окрестности core‑точки.
- **Noise** (шум) — точка, которая не попала ни в один кластер (ни core, ни border).

### 7.2. Применение DBSCAN

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

features = ["feature1", "feature2"]
X = df[features]

# Масштабируем, чтобы признаки были сопоставимы
X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(
    eps=0.5,          # радиус окрестности
    min_samples=5,    # minPts
    n_jobs=-1,
)

dbscan.fit(X_scaled)
labels = dbscan.labels_   # -1 = шум, 0,1,2,... = номера кластеров

df["cluster"] = labels
df["is_noise"] = (labels == -1).astype(int)

df["cluster"].value_counts()    # сколько объектов в каждом кластере и шума
```

### 7.3. Визуализация кластеров и шума

```python
plt.figure(figsize=(6, 4))
sns.scatterplot(
    x=X.iloc[:, 0],
    y=X.iloc[:, 1],
    hue=df["cluster"],
    palette="tab10",
    alpha=0.7,
)
plt.title("DBSCAN кластеры (шум = -1)")
plt.show()
```

Подбор `eps`:
- небольшой `eps` → много шума, мало точек в кластерах,
- большой `eps` → почти всё становится одним кластером.
Часто используют эвристику “k‑distance plot” (для экзамена достаточно знать идею).

---

## 8. Когда удалять аномалии, а когда — нет

**Удалять/корректировать**, если:
- значения очевидно неверные (ошибки ввода/логирования),
- аномалии — результат технических проблем (сломанный датчик),
- модели сильно страдают (очевидное переобучение/нестабильность).

**Не удалять (или аккуратно помечать)**, если:
- аномалии — реальная часть процесса (fraud, аварии, редкие события),
- вы решаете задачу поиска аномалий/долгих хвостов,
- объём данных небольшой, и удаление сильно меняет распределения.

Возможный компромисс:
- обучить модель **без** аномалий,
- потом отдельно обучить детектор аномалий и использовать оба сигнала.

---

## 9. Минимальные шаблоны

### 9.1. IQR‑фильтр для одного признака

```python
col = "feature"

Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df[col] >= lower) & (df[col] <= upper)]
```

### 9.2. Isolation Forest для нескольких признаков

```python
from sklearn.ensemble import IsolationForest

features = ["f1", "f2", "f3"]
X = df[features]

iso = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
)
iso.fit(X)

df["anomaly"] = (iso.predict(X) == -1).astype(int)
```

### 9.3. DBSCAN для поиска “шума”

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

features = ["f1", "f2"]
X = df[features]

X_scaled = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X_scaled)

df["cluster"] = labels
df["is_noise"] = (labels == -1).astype(int)
```

Эти три блока покрывают 90% практических сценариев по теме “выбросы и аномалии” на экзамене.
```