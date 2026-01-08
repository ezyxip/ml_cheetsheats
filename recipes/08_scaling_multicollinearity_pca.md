```markdown
# 08. Масштабирование, мультиколлинеарность и PCA

Цель: понять, когда нужно масштабирование, как диагностировать и лечить мультиколлинеарность, и как использовать PCA для снижения размерности (борьба с “проклятием размерности”).

---

## 1. Быстрый чек‑лист

1. Определить, какие модели будете использовать:
   - **нужен скейлинг**: линейные модели (LinReg/LogReg), SVM, KNN, k‑Means, PCA, нейросети;
   - **можно без скейлинга**: деревья, RandomForest, бустинг (XGBoost/LightGBM/CatBoost).
2. Построить:
   - корреляционную матрицу (heatmap),
   - посчитать VIF для числовых признаков.
3. Если есть сильная мультиколлинеарность:
   - удалить лишние признаки,
   -/или использовать регуляризацию (Ridge/Lasso/ElasticNet),
   -/или снизить размерность (PCA).
4. Проверить влияние:
   - метрики модели (CV),
   - интерпретируемость (устойчивость коэффициентов, feature importance).

---

## 2. Масштабирование признаков

### 2.1. Когда масштабирование критично

**Нужно масштабировать**, если:

- используете:
  - линейную регрессию / логистическую регрессию с регуляризацией,
  - SVM (особенно с RBF/полиномиальными ядрами),
  - KNN (расстояния),
  - k‑Means (расстояния),
  - PCA,
  - многие нейросети;
- признаки имеют сильно разные масштабы (например, `доход` в тысячах, `возраст` в годах).

**Можно не масштабировать**, если:

- используете модели на деревьях:
  - DecisionTree, RandomForest,
  - GradientBoosting, XGBoost, LightGBM, CatBoost.
  
Но даже для деревьев скейлинг не вредит, иногда облегчает регуляризацию/оптимизацию.

---

### 2.2. StandardScaler

Приводит признак к нулевому среднему и единичному стандартному отклонению:

\[
z = \frac{x - \mu}{\sigma}
\]

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   # fit ТОЛЬКО на train
```

**Типичный случай**: линейные модели с L1/L2, SVM, PCA, KNN.

---

### 2.3. MinMaxScaler

Линейно масштабирует признак в интервал [0, 1] (или [min, max]):

\[
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
\]

```python
from sklearn.preprocessing import MinMaxScaler

mm = MinMaxScaler(feature_range=(0, 1))
X_scaled = mm.fit_transform(X)
```

**Когда полезно**:

- когда алгоритм чувствителен к абсолютным значениям в ограниченном диапазоне (нейросети, некоторые distance‑методы),
- когда признаки имеют “разумные” минимумы/максимумы (нет экстремальных выбросов).

---

### 2.4. RobustScaler

Масштабирование, устойчивое к выбросам (использует медиану и IQR):

\[
x' = \frac{x - \text{median}}{\text{IQR}}
\]

```python
from sklearn.preprocessing import RobustScaler

rb = RobustScaler()
X_scaled = rb.fit_transform(X)
```

**Типичный случай**:

- признаки с тяжёлыми хвостами и выбросами (доход, цена),
- когда StandardScaler слишком чувствителен к выбросам.

---

### 2.5. Масштабирование в пайплайне (правильно, без утечек)

```python
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

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
    ("logreg", LogisticRegression(max_iter=1000)),
])

clf.fit(X_train, y_train)
```

---

## 3. Мультиколлинеарность

### 3.1. Что это и почему плохо

Мультиколлинеарность = сильная линейная зависимость между признаками:

- признаки сильно коррелированы (один можно линейно выразить через другой),
- приводит к:
  - нестабильным и “случайным” коэффициентам в линейных моделях,
  - большой дисперсии оценок (коэффициенты “скачут” от выборки к выборке),
  - сложности интерпретации (непонятно, какой признак действительно важен).

Для деревьев/бустинга — не так критично (они сами выбирают, какой признак сплитить), но тоже может немного ухудшать устойчивость.

---

### 3.2. Корреляционная матрица и heatmap

```python
import seaborn as sns
import matplotlib.pyplot as plt

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

corr = df[num_cols].corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
plt.title("Матрица корреляций (Pearson)")
plt.show()
```

Смотрите на:
- пары признаков с |corr| > 0.8–0.9 — возможная сильная мультиколлинеарность.

---

## 4. VIF (Variance Inflation Factor)

### 4.1. Формула

Для признака \(X_j\):

- строим регрессию \(X_j\) на все остальные признаки \(X_{-j}\),
- считаем \(R_j^2\) — коэффициент детерминации этой регрессии,
- тогда:

\[
\text{VIF}_j = \frac{1}{1 - R_j^2}
\]

Интерпретация (правила большого пальца):

- VIF ~ 1 — нет мультиколлинеарности,
- VIF > 5 — заметная,
- VIF > 10 — сильная, стоит задуматься.

### 4.2. Расчёт VIF в Python

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd

# Берём только числовые признаки
X_num = df[num_cols].dropna().copy()

# Можно добавить константу (интерсепт), но для VIF обычно смотрим только на сами признаки
X_vif = add_constant(X_num)

vif_data = []
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    vif_data.append((col, vif))

vif_df = pd.DataFrame(vif_data, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
print(vif_df)
```

---

## 5. Борьба с мультиколлинеарностью

### 5.1. Удаление признаков

Подход:

1. Найти сильно коррелирующие пары (|corr| > 0.9).
2. Для каждой пары удалить один признак:
   - менее интерпретируемый,
   - или тот, у которого VIF больше,
   - либо оставить тот, что даёт лучшую метрику модели.
3. Пересчитать VIF/метрики.

Пример:

```python
# Скажем, area и rooms сильно коррелированы
df = df.drop(columns=["area"])  # оставили rooms, если он более “важен”
```

---

### 5.2. Регуляризация: Ridge, Lasso, ElasticNet

#### Ridge (L2)

Добавляет штраф \(\lambda \sum w_j^2\) к MSE:

- уменьшает по модулю все коэффициенты,
- особенно стабилизирует ситуацию при мультиколлинеарности,
- **не** зануляет коэффициенты (оставляет все признаки, но с “усреднёнными” весами).

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train_scaled, y_train)
```

#### Lasso (L1)

Добавляет штраф \(\lambda \sum |w_j|\):

- может занулять некоторые коэффициенты → **отбор признаков**,
- полезно, если признаков много и хотим sparse‑решение.

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_train_scaled, y_train)

coef = pd.Series(lasso.coef_, index=X_train_scaled.columns)
print(coef[coef != 0].sort_values())
```

#### ElasticNet

Комбинация L1 и L2:

\[
\lambda \left(\alpha \sum |w_j| + (1 - \alpha) \sum w_j^2\right)
\]

```python
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42)
enet.fit(X_train_scaled, y_train)
```

**Где использовать регуляризацию**:

- линейная регрессия / логистическая регрессия,
- много признаков, есть подозрение на мультиколлинеарность,
- хотим:
  - более устойчивые оценки (Ridge),
  - либо автоматический отбор признаков (Lasso/ElasticNet).

---

## 6. PCA (Principal Component Analysis)

### 6.1. Идея

PCA = линейное преобразование признаков:

- строит новые ортогональные компоненты (PC1, PC2, …),
- первая компонента — направление наибольшей дисперсии,
- вторая — направление наибольшей дисперсии, ортогональное первой, и т.д.

Зачем:

- **снижение размерности** (оставляем только первые k компонент),
- частичное устранение мультиколлинеарности (компоненты некоррелированы),
- ускорение обучения, борьба с “проклятием размерности”.

Важно:
- PCA **не учитывает таргет** — чисто unsupervised метод,
- можно потерять информацию, важную для предсказания.

---

### 6.2. Обязательное масштабирование перед PCA

PCA чувствителен к масштабу:

- если один признак в больших числах, а другой — в маленьких, первый доминирует,
- поэтому обычно применяют `StandardScaler`.

---

### 6.3. Применение PCA

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X_num = df[num_cols].dropna().copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_num)

pca = PCA(n_components=None, random_state=42)  # пока возьмём все
X_pca = pca.fit_transform(X_scaled)

explained = pca.explained_variance_ratio_
print("Доля объяснённой дисперсии по компонентам:", explained)
print("Накопленная доля:", explained.cumsum())
```

---

### 6.4. Как выбрать число компонент

Подходы:

1. По накопленной дисперсии:
   - выбрать минимальное k, для которого `cumsum[k] >= 0.9` (например, 90% вариации).
2. По графику “локтя” (scree plot):
   - посмотреть, где кривая объяснённой дисперсии начинает выравниваться.

Пример:

```python
import numpy as np
import matplotlib.pyplot as plt

pca = PCA().fit(X_scaled)

plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel("Число компонент")
plt.ylabel("Накопленная доля объяснённой дисперсии")
plt.grid(True)
plt.show()
```

---

### 6.5. PCA в пайплайне

Пример: сначала скейлинг, потом PCA, потом модель.

```python
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipe_pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=10, random_state=42)),
    ("logreg", LogisticRegression(max_iter=1000)),
])

pipe_pca.fit(X_train, y_train)
y_pred = pipe_pca.predict(X_test)
```

Можно тюнить `n_components` через `GridSearchCV`.

---

## 7. Связь с “проклятием размерности”

**Проклятие размерности**:

- при росте числа признаков:
  - точки становятся “разреженными” в пространстве,
  - расстояния между объектами выравниваются (все примерно одинаково далеко),
  - для хорошей оценки нужно **экспоненциально больше** данных,
  - модели (особенно KNN, SVM, линейные без регуляризации) переобучаются.

Что помогает:

1. **Осмысленный feature selection**:
   - удаление неинформативных/избыточных фич,
   - Lasso / ElasticNet.
2. **Снижение размерности**:
   - PCA, автоэнкодеры (для нейросетей), другие методы.
3. **Регуляризация**:
   - Ridge/Lasso/ElasticNet уменьшают эффективную сложность модели.
4. **Уменьшение “раздува” признаков**:
   - аккуратнее с One‑Hot Encoding (особенно для признаков с тысячами категорий),
   - использовать Target/Binary/Hashing Encoding, группировку редких категорий.

---

## 8. Минимальные шаблоны

### 8.1. Стандартное масштабирование числовых + логистическая регрессия

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num_pipe = Pipeline([
    ("scaler", StandardScaler()),
])

cat_pipe = Pipeline([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

clf = Pipeline([
    ("preprocess", preprocess),
    ("logreg", LogisticRegression(penalty="l2", C=1.0, max_iter=1000)),
])
```

---

### 8.2. Расчёт VIF и удаление признаков с сильной мультиколлинеарностью

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import pandas as pd

X_num = df[num_cols].dropna().copy()
X_vif = add_constant(X_num)

vif_list = []
for i, col in enumerate(X_vif.columns):
    if col == "const":
        continue
    vif = variance_inflation_factor(X_vif.values, i)
    vif_list.append((col, vif))

vif_df = pd.DataFrame(vif_list, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)
print(vif_df)

# Пример: удалить признаки с VIF > 10
high_vif_features = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
df_reduced = df.drop(columns=high_vif_features)
```

---

### 8.3. PCA + классификация

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),  # выбираем число компонент так, чтобы объяснить 95% дисперсии
    ("logreg", LogisticRegression(max_iter=1000)),
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

(Если `n_components` — число < 1, PCA подберёт минимальное k, для которого накопленная дисперсия ≥ это число.)
```