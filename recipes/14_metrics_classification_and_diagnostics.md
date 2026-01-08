```markdown
# 14. Классификация: метрики и диагностика

Цель: уметь считать и интерпретировать метрики классификации, строить ROC / PR‑кривые, матрицу ошибок, настраивать порог и особенно правильно работать с дисбалансом классов.

---

## 1. Базовые понятия

Для бинарной классификации (класс 1 — “позитивный”):

- **TP** (True Positive) — предсказали 1, и это 1.
- **FP** (False Positive) — предсказали 1, а это 0.
- **TN** (True Negative) — предсказали 0, и это 0.
- **FN** (False Negative) — предсказали 0, а это 1.

---

## 2. Основные метрики

### 2.1. Accuracy

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

- доля правильных предсказаний;
- **опасна** при сильном дисбалансе (можно получить высокую accuracy, просто предсказывая всегда 0).

```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
```

---

### 2.2. Precision, Recall, F1‑score

**Precision** (точность):

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

“Из всех предсказанных 1, какая доля действительно 1?”

**Recall** (полнота, TPR, sensitivity):

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

“Из всех реальных 1, какую долю мы нашли?”

**F1‑score** (гармоническое среднее Precision и Recall):

\[
F1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
f1        = f1_score(y_true, y_pred)
```

---

### 2.3. ROC‑AUC

ROC‑кривая — график TPR против FPR при изменении порога.

- **TPR** (True Positive Rate) — то же самое, что Recall:

\[
\text{TPR} = \frac{TP}{TP + FN}
\]

- **FPR** (False Positive Rate):

\[
\text{FPR} = \frac{FP}{FP + TN}
\]

**ROC‑AUC** — площадь под ROC‑кривой.

- 0.5 — случайный предсказатель;
- 1.0 — идеальный классификатор;
- больше → лучше.

```python
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_true, y_proba)
print("ROC-AUC:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_true, y_proba)
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="random")
plt.xlabel("FPR")
plt.ylabel("TPR (Recall)")
plt.title("ROC-кривая")
plt.legend()
plt.grid(True)
plt.show()
```

---

### 2.4. PR‑AUC (Precision‑Recall AUC)

Precision‑Recall кривая — график Precision vs Recall при разных порогах.

**PR‑AUC** (часто считают как *Average Precision*).

Особенно важна при **сильном дисбалансе**, так как фокусируется на позитивном классе.

```python
from sklearn.metrics import average_precision_score, precision_recall_curve

pr_auc = average_precision_score(y_true, y_proba)
print("PR-AUC (Average Precision):", pr_auc)

precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
plt.plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall кривая")
plt.grid(True)
plt.legend()
plt.show()
```

---

## 3. Матрица ошибок (Confusion Matrix)

### 3.1. Построение и вывод

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_true, y_pred)  # по умолчанию порядок классов [0,1]
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
```

Матрица вида (для классов [0, 1]):

|           | Pred 0 | Pred 1 |
|-----------|--------|--------|
| True 0    |   TN   |   FP   |
| True 1    |   FN   |   TP   |

Интерпретация:
- много FN → пропускаем много позитивных объектов (низкий Recall);
- много FP → много ложных тревог (низкий Precision).

---

## 4. Работа с порогом (threshold tuning)

По умолчанию `model.predict` использует порог 0.5 для бинарной классификации (если есть `predict_proba`).

При дисбалансе/особых требованиях:
- нужно **двигать порог**,
- и смотреть, как меняются Precision, Recall, F1.

### 4.1. Изменение порога вручную

```python
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

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

---

### 4.2. Графики зависимости метрик от порога

```python
precisions = []
recalls = []
f1s = []

for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    precisions.append(precision_score(y_val, y_pred_thr, zero_division=0))
    recalls.append(recall_score(y_val, y_pred_thr))
    f1s.append(f1_score(y_val, y_pred_thr, zero_division=0))

plt.figure(figsize=(8, 4))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls,    label="Recall")
plt.plot(thresholds, f1s,        label="F1")
plt.xlabel("Threshold")
plt.ylabel("Metric value")
plt.title("Метрики vs порог")
plt.grid(True)
plt.legend()
plt.show()
```

Интерпретация:

- при низких порогах (0.1–0.2):
  - Recall высокий (почти все 1 поймали),
  - Precision низкий (много FP).
- при высоких порогах (0.8–0.9):
  - Precision высокий,
  - Recall низкий (пропускаем много 1).
- выбираем порог в зависимости от бизнес‑требований:
  - “важнее не пропустить позитив” → максимизируем Recall при приемлемом Precision,
  - “важнее мало ложных тревог” → держим Precision выше порога.

---

## 5. Особый фокус: дисбаланс классов

При сильном дисбалансе:

- Accuracy ≈ “доля мажоритарного класса” → почти бесполезна.
- Лучше использовать:
  - ROC‑AUC,
  - PR‑AUC,
  - F1 (или F1 для интересующего класса),
  - Recall или Precision, в зависимости от задачи.

### 5.1. Пример: оценка при дисбалансе

```python
from sklearn.metrics import classification_report

print("Распределение классов в тесте:")
print(y_test.value_counts(normalize=True))

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=3))
```

Смотрим:

- `precision`/`recall` именно для класса `1`,
- `macro`/`weighted` F1.

---

## 6. Быстрый чек‑лист по диагностике классификатора

1. **Матрица ошибок**:
   - сколько FN и FP,
   - какая ошибка критичнее (FN или FP?) для задачи.
2. **Метрики**:
   - для сбалансированных данных: Accuracy, F1, ROC‑AUC,
   - для дисбаланса: ROC‑AUC + PR‑AUC + F1/Recall/Precision по позитивному классу.
3. **ROC‑кривая**:
   - если ROC‑AUC ≈ 0.5 → модель почти случайная,
   - сравнивать несколько моделей на одном графике.
4. **PR‑кривая** (особенно при дисбалансе):
   - важна форма кривой и PR‑AUC (Average Precision).
5. **Порог**:
   - подобрать порог под бизнес‑критерий (например, Recall ≥ 0.9).

---

## 7. Мини‑рецепт “под ключ”

После обучения модели:

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Базовые метрики
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc     = accuracy_score(y_test, y_pred)
prec    = precision_score(y_test, y_pred)
rec     = recall_score(y_test, y_pred)
f1      = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)

print(f"Accuracy={acc:.3f}, Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")
print(f"ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")

# 2. Матрица ошибок
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# 3. ROC-кривая
fpr, tpr, thr_roc = roc_curve(y_test, y_proba)
plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("FPR")
plt.ylabel("TPR (Recall)")
plt.title("ROC-кривая")
plt.grid(True)
plt.legend()
plt.show()

# 4. PR-кривая
precision, recall, thr_pr = precision_recall_curve(y_test, y_proba)
plt.plot(recall, precision, label=f"PR (AP={pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall кривая")
plt.grid(True)
plt.legend()
plt.show()

# 5. Подбор порога по F1 (или по Recall/Precision)
thresholds = np.linspace(0.0, 1.0, 101)

f1s = []
for thr in thresholds:
    y_pred_thr = (y_proba >= thr).astype(int)
    f1s.append(f1_score(y_test, y_pred_thr, zero_division=0))

best_thr = thresholds[np.argmax(f1s)]
print("Лучший threshold по F1:", best_thr, "F1:", max(f1s))

plt.plot(thresholds, f1s)
plt.xlabel("Threshold")
plt.ylabel("F1")
plt.title("F1 vs Threshold")
plt.grid(True)
plt.show()
```

Эти шаги дают полную картину поведения классификатора, особенно на несбалансированных данных.
```