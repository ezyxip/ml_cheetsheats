```markdown
# 10. Дисбаланс классов: рецепты

Цель: понять, что в данных есть дисбаланс классов, выбрать корректные метрики и применить методы борьбы (class_weight, oversampling/undersampling, настройка порога), не допустив утечек.

---

## 1. Как обнаружить дисбаланс

```python
y = df["target"]

# Абсолютные количества
print(y.value_counts())

# Доли классов
print(y.value_counts(normalize=True))
```

Примеры:
- 0: 9500, 1: 500 → сильный дисбаланс (класс 1 — редкий).
- 0: 5200, 1: 4800 → почти нет дисбаланса.

Чем более редкий “позитивный” класс (обычно интересующий: fraud, дефолт, болезнь), тем сильнее нужно думать о специальных методах.

---

## 2. Метрики при несбалансированных данных

### 2.1. Базовые: Precision, Recall, F1‑score

Предположим, класс `1` — позитивный (интересующий).

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = y_test
y_pred = model.predict(X_test)

precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
```

Интуиции:
- **Precision** — “какая доля предсказанных 1 действительно 1?”
- **Recall** — “какую долю всех настоящих 1 мы нашли?”
- **F1** — гармоническое среднее precision и recall, баланс между ними.

При сильном дисбалансе `Accuracy` почти бесполезна (“все нули” дают высокую accuracy).

---

### 2.2. ROC‑AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_true, y_proba)
print("ROC-AUC:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr)
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title(f"ROC-кривая (AUC={roc_auc:.3f})")
plt.grid(True)
plt.show()
```

ROC‑AUC:
- хорошо работает и при дисбалансе,
- но иногда слишком оптимистичен, когда положительных очень мало.

---

### 2.3. PR‑AUC (Precision‑Recall AUC)

Более информативная при **сильном дисбалансе**, концентрируется на позитивном классе.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_true, y_proba)
print("PR-AUC (average precision):", pr_auc)

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall кривая (AP={pr_auc:.3f})")
plt.grid(True)
plt.show()
```

---

## 3. Методы борьбы с дисбалансом

### 3.1. class_weight='balanced'

Многие модели sklearn поддерживают параметр `class_weight`:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

logreg = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

rf = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=200,
    random_state=42,
)
```

Идея:
- увеличивает вес ошибок на миноритарном классе,
- модель “боится” пропускать позитивные объекты,
- помогает, не меняя сами данные.

---

### 3.2. Oversampling: SMOTE

SMOTE (Synthetic Minority Over-sampling Technique):

- создаёт **синтетические** объекты миноритарного класса,
- эффективно увеличивает количество примеров `class=1`.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(
    sampling_strategy="auto",   # по умолчанию выравнивает до мажоритарного
    k_neighbors=5,
    random_state=42,
)

X_res, y_res = smote.fit_resample(X_train, y_train)
print("До SMOTE:", y_train.value_counts())
print("После SMOTE:", y_res.value_counts())
```

Важно:
- **никогда** не применять SMOTE к test/validation,
- oversampling делаем только на train (или внутри CV‑фолдов через Pipeline).

---

### 3.3. Undersampling: RandomUnderSampler

Undersampling — выбрасываем часть объектов мажоритарного класса.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(
    sampling_strategy="auto",  # по умолчанию делает классы равными
    random_state=42,
)

X_res, y_res = rus.fit_resample(X_train, y_train)
print("До:", y_train.value_counts())
print("После:", y_res.value_counts())
```

Плюсы:
- быстро, просто, уменьшает размер обучающего набора.

Минусы:
- выкидываем данные (можем потерять информацию).

---

### 3.4. Изменение порога (threshold tuning)

Большинство классификаторов по умолчанию используют порог 0.5:
- `proba >= 0.5` → класс 1,
- `proba < 0.5` → класс 0.

При дисбалансе это часто не оптимально:
- хотим сдвинуть порог, чтобы увеличить Recall или F1.

```python
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

y_proba = model.predict_proba(X_val)[:, 1]

thresholds = np.linspace(0.0, 1.0, 101)

best_thr = 0.5
best_f1 = 0

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    f1 = f1_score(y_val, y_pred_thr)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr

print("Лучший threshold по F1:", best_thr, "F1:", best_f1)
```

Дальше:
- использовать `best_thr` для предсказаний на test/проде:

```python
y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred_test = (y_proba_test >= best_thr).astype(int)
```

Можно оптимизировать по `Recall`, `Precision` или кастомному скору.

---

## 4. Использование imblearn: Pipeline со SMOTE/RandomUnderSampler

### 4.1. Oversampling (SMOTE) в Pipeline

Чтобы не допустить утечек, нужно, чтобы SMOTE:

- вызывал `fit_resample` **отдельно для каждого train‑fold**,
- не видел валидационные/тестовые данные.

Для этого используем `imblearn.pipeline.Pipeline`:

```python
from imblearn.pipeline import Pipeline  # ВАЖНО: не sklearn.pipeline
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

X = df.drop(columns=["target"])
y = df["target"]

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

clf = RandomForestClassifier(random_state=42)

pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", clf),
])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc")
print("CV ROC-AUC (SMOTE):", scores.mean(), "+-", scores.std())
```

SMOTE в таком пайплайне:
- делает oversampling только на train‑части каждого фолда.

---

### 4.2. Undersampling в Pipeline

```python
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

pipe_rus = Pipeline(steps=[
    ("preprocess", preprocess),
    ("rus", RandomUnderSampler(random_state=42)),
    ("clf", clf),
])

scores = cross_val_score(pipe_rus, X, y, cv=skf, scoring="roc_auc")
print("CV ROC-AUC (RandomUnderSampler):", scores.mean(), "+-", scores.std())
```

Можно также комбинировать oversampling и undersampling (например, SMOTE + TomekLinks), но для экзамена достаточно SMOTE и RandomUnderSampler.

---

## 5. Мини‑рецепт “под ключ”

1. Проверяем дисбаланс:

```python
print(y.value_counts(normalize=True))
```

2. Выбираем метрику:
   - при сильном дисбалансе: ROC‑AUC + PR‑AUC, F1, Recall.

3. Строим базовую модель с `class_weight='balanced'`:

```python
clf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
)
```

4. Пробуем Pipeline со SMOTE:

```python
pipe = Pipeline([
    ("preprocess", preprocess),
    ("smote", SMOTE(random_state=42)),
    ("clf", clf),
])
```

5. Оцениваем через StratifiedKFold на ROC‑AUC / F1.

6. На валидации подбираем порог:
   - максимизируем F1 или Recall при ограничении на Precision.

7. Финально:
   - обучаем пайплайн на всём train,
   - применяем к test с выбранным threshold,
   - считаем ROC‑AUC, PR‑AUC, Precision, Recall, F1 на test.

Этот сценарий покрывает большинство задач с дисбалансом классов.
```