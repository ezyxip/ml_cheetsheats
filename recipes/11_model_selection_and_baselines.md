```markdown
# 11. Выбор моделей и базовые решения (Baselines)

Цель: быстро построить разумный baseline, понимать, какую модель выбрать под задачу, и знать плюсы/минусы основных алгоритмов.

---

## 1. Зачем нужен baseline

Baseline:

- простейшая модель, которую легко побить;
- даёт “нижнюю планку” качества;
- помогает понять, вообще ли есть сигнал в данных.

---

## 2. Baseline для регрессии

### 2.1. DummyRegressor

```python
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

X = df.drop(columns=["target"])
y = df["target"]

dummy_mean = DummyRegressor(strategy="mean")
scores = cross_val_score(dummy_mean, X, y, cv=5, scoring="neg_mean_absolute_error")
print("Baseline (mean) MAE:", -scores.mean())
```

Основные стратегии:

- `strategy="mean"` — предсказывать среднее по train;
- `strategy="median"` — предсказывать медиану;
- `strategy="constant", constant=C` — предсказывать константу C.

### 2.2. Ручной baseline

```python
y_train_mean = y_train.mean()
y_pred_baseline = np.full_like(y_test, fill_value=y_train_mean, dtype=float)
```

---

## 3. Baseline для классификации

### 3.1. DummyClassifier

```python
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

X = df.drop(columns=["target"])
y = df["target"]

dummy = DummyClassifier(strategy="most_frequent")  # всегда предсказывает самый частый класс
scores = cross_val_score(dummy, X, y, cv=5, scoring="accuracy")
print("Baseline accuracy:", scores.mean())
```

Стратегии:

- `most_frequent` — всегда самый частый класс;
- `stratified` — случайно, сохраняя распределение классов;
- `uniform` — случайно с равными вероятностями;
- `constant` — всегда один фиксированный класс.

### 3.2. Быстрая оценка “на бумаге”

- Если класс 1 — 5%, класс 0 — 95%:
  - baseline accuracy ≈ 0.95 (всегда 0),
  - если модель даёт 0.96 accuracy — это почти ничего не значит → надо смотреть ROC‑AUC, F1.

---

## 4. Как выбирать модель: основные вопросы

Перед выбором алгоритма ответьте:

1. **Тип задачи**:
   - регрессия / классификация / временные ряды.
2. **Размер датасета (N)**:
   - очень мало (N < 1–2k),
   - средне (несколько тысяч / десятков тысяч),
   - очень много (100k+ / миллионы).
3. **Число признаков (p)**:
   - немного (до 50),
   - много (сотни/тысячи),
   - p >> N (high dimensional).
4. **Линейность зависимости**:
   - видите почти линейные зависимости → можно начать с линейных моделей;
   - сильная нелинейность → деревья/ансамбли.
5. **Интерпретируемость**:
   - нужна ли “читаемая” модель (коэффициенты, простые правила)?
6. **Время/ресурсы**:
   - нужны быстрые обучения/предсказания или можно ждать?

---

## 5. Обзор моделей (кратко: когда, плюсы/минусы)

### 5.1. Линейные модели (LinReg, LogisticRegression)

**Когда использовать**:

- зависимость близка к линейной,
- много наблюдений, относительно немного признаков,
- нужна интерпретация (коэффициенты),
- быстрый baseline.

**Пример (регрессия)**:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

**Пример (классификация)**:

```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(
    penalty="l2",      # L2 по умолчанию
    C=1.0,
    max_iter=1000,
)
clf.fit(X_train_scaled, y_train)
```

**Плюсы**:

- быстрые,
- хорошо масштабируются,
- понятная интерпретация (знаки/величины коэффициентов).

**Минусы**:

- плохо ловят нелинейные зависимости,
- чувствительны к мультиколлинеарности, выбросам, масштабу.

---

### 5.2. KNN (k‑Nearest Neighbors)

**Когда**:

- мало признаков (p небольшое),
- данные не слишком большие (N до ~10–50k),
- форма зависимости сложная, но гладкая.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
)
knn.fit(X_train_scaled, y_train)
```

**Плюсы**:

- простая идея,
- нелинейность “из коробки”,
- мало предположений о данных.

**Минусы**:

- медленный при больших N (предсказание),
- чувствителен к масштабу и “проклятию размерности”,
- нет хорошей интерпретации.

---

### 5.3. Naive Bayes

**Когда**:

- текст, bag‑of‑words,
- много признаков (p большое), но сильные независимости,
- нужен очень быстрый baseline.

```python
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_train_counts, y_train)
```

**Плюсы**:

- очень быстрый,
- хорошо работает на текстах,
- robust при p >> N.

**Минусы**:

- на табличных данных часто проигрывает деревьям/бустингу,
- сильное предположение о независимости признаков.

---

### 5.4. SVM (Support Vector Machines)

**Когда**:

- N не очень велико (до ~10k, иногда больше с линейным ядром),
- много признаков,
- сложные нелинейные границы классов.

```python
from sklearn.svm import SVC

svm = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale",
    probability=True,
)
svm.fit(X_train_scaled, y_train)
```

**Плюсы**:

- мощен на “сложных” задачах с небольшим/средним N,
- хорошо работает при p >> N,
- устойчив к переобучению (при правильном подборе C, gamma).

**Минусы**:

- требует скейлинга,
- плохо масштабируется по N (особенно RBF‑ядро),
- интерпретация сложная.

---

### 5.5. Деревья решений (DecisionTree)

**Когда**:

- нужен простой интерпретируемый набор правил,
- немного признаков и объектов,
- baseline для моделей на деревьях.

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    max_depth=3,       # ограничить глубину для интерпретируемости
    random_state=42,
)
tree.fit(X_train, y_train)
```

**Плюсы**:

- интерпретируемость (если дерево маленькое),
- работает с числовыми и категориальными (после OHE),
- не требует скейлинга.

**Минусы**:

- одиночное дерево легко переобучается,
- нестабильное (чувствительно к небольшим изменениям данных).

---

### 5.6. RandomForest (RF)

**Когда**:

- универсальная табличная задача (часто “go‑to” модель),
- есть нелинейности, взаимодействия признаков,
- N и p от малых до средних/крупных.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
```

**Плюсы**:

- хорошо работает “out‑of‑the‑box”,
- устойчив к переобучению по сравнению с одиночным деревом,
- умеет работать с разными типами признаков,
- меньше чувствителен к масштабированию и выбросам.

**Минусы**:

- медленнее, чем линейные модели,
- хуже, чем градиентный бустинг в “тонкой настройке”,
- интерпретация сложнее (но есть feature importance).

---

### 5.7. Градиентный бустинг (XGBoost, LightGBM, CatBoost)

**Когда**:

- серьёзные табличные задачи,
- нужно высокое качество,
- данные от малых до очень больших.

---

#### 5.7.1. XGBoost

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)
xgb.fit(X_train, y_train)
```

**Плюсы**:

- мощный классический бустинг,
- много возможностей тонкой настройки.

**Минусы**:

- чувствителен к параметрам,
- требует осторожного препроцессинга категориальных фич (OHE/LabelEnc).

---

#### 5.7.2. LightGBM

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
lgbm.fit(X_train, y_train)
```

**Плюсы**:

- очень быстрый и эффективный на больших датасетах,
- умеет работать с категориальными признаками (через `categorical_feature`),
- хорошее качество на табличных данных.

**Минусы**:

- много гиперпараметров,
- требует внимательности с категориальными (но проще, чем XGBoost).

---

#### 5.7.3. CatBoost

```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    depth=6,
    learning_rate=0.05,
    n_estimators=500,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    verbose=0,
)

cat.fit(
    X_train, y_train,
    cat_features=cat_cols_idx,  # индексы категориальных признаков
)
```

**Плюсы**:

- нативная поддержка категориальных фич (без OHE),
- часто даёт одно из лучших качеств “из коробки”,
- устойчива к пропускам.

**Минусы**:

- “чёрный ящик” по сравнению с линейными/маленькими деревьями,
- дольше обучается, чем RF/простые модели.

---

## 6. Как выбирать модель в типичных ситуациях

### 6.1. “Первый подход к снаряду”

1. Базовый EDA → понять тип задачи и данные.
2. Baseline:
   - DummyRegressor/DummyClassifier,
   - простая линейная модель (LinReg/LogReg) или маленький RandomForest.
3. Если качество явно плохое → думать над фичами + переходить к ансамблям.

---

### 6.2. Малый датасет (N < ~2000)

- Начать:
  - линейные модели (с регуляризацией),
  - KNN,
  - небольшие деревья.
- Ансамбли (RF/бустинг) тоже подойдут, но осторожно с переобучением (n_estimators не слишком большой, сильная регуляризация).
- CV с большим количеством фолдов (5–10).

---

### 6.3. Средний датасет (N ~ 2k–100k, p до сотен/тысяч)

- Очень хороший вариант:
  - RandomForest,
  - градиентный бустинг (XGBoost / LightGBM / CatBoost).
- Линейные модели как baseline и для интерпретации.
- При p >> N → линейные модели с регуляризацией, Naive Bayes (для текстов), SVM.

---

### 6.4. Требуется интерпретируемость

- Линейные модели (LinReg/LogReg) с хорошо подготовленными фичами.
- Небольшие деревья (ограничить `max_depth`).
- Для ансамблей:
  - feature importance,
  - permutation importance,
  - SHAP.

---

## 7. Мини‑рецепт выбора модели (для шпоры)

1. Сделать Dummy baseline (регрессия/классификация).
2. Простая модель:
   - регрессия → LinearRegression / Ridge,
   - классификация → LogisticRegression / маленький RandomForest.
3. Посмотреть:
   - хватает ли линейности? если нет — переходить к деревьям/ансамблям.
4. Для “боевого” решения на табличных данных:
   - RandomForest как быстрый и устойчивый вариант,
   - затем попробовать CatBoost/LightGBM/XGBoost.
5. Для отчёта/экзамена:
   - линейная модель + одна “мощная” (RF/бустинг),
   - сравнение метрик,
   - объяснение выбора (интерпретируемость vs качество).

Эти шаги покрывают большинство практических задач по выбору модели и построению baseline.
```