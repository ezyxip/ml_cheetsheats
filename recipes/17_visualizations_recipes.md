```markdown
# 17. Рецепты визуализаций (matplotlib + seaborn)

Цель: иметь под рукой готовые куски кода для EDA, диагностики и отчётов.

---

## 0. Базовый шаблон импорта и настроек

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={
    "figure.figsize": (8, 4),
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
```

---

## 1. Распределения и выбросы

### 1.1. Гистограммы (histogram)

```python
col = "feature"

plt.figure(figsize=(6, 4))
sns.histplot(df[col].dropna(), bins=30, kde=False)
plt.title(f"Гистограмма: {col}")
plt.xlabel(col)
plt.ylabel("Count")
plt.show()
```

### 1.2. Гистограмма + KDE (плотность)

```python
col = "feature"

plt.figure(figsize=(6, 4))
sns.histplot(df[col].dropna(), bins=30, kde=True)
plt.title(f"Распределение + KDE: {col}")
plt.xlabel(col)
plt.ylabel("Count")
plt.show()
```

---

### 1.3. Boxplot (поиск выбросов)

```python
col = "feature"

plt.figure(figsize=(6, 3))
sns.boxplot(x=df[col])
plt.title(f"Boxplot: {col}")
plt.xlabel(col)
plt.show()
```

### 1.4. Несколько boxplot’ов для числовых признаков

```python
num_cols = ["feature1", "feature2", "feature3"]

plt.figure(figsize=(8, 4))
sns.boxplot(data=df[num_cols], orient="h")
plt.title("Boxplot для нескольких признаков")
plt.show()
```

---

## 2. Scatter plot, линейность/нелинейность

### 2.1. Scatter plot: признак vs таргет (регрессия)

```python
feature = "feature"
target = "target"

plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x=feature, y=target, alpha=0.5)
plt.title(f"{feature} vs {target}")
plt.xlabel(feature)
plt.ylabel(target)
plt.show()
```

### 2.2. Scatter + регрессионная линия (regplot)

```python
feature = "feature"
target = "target"

plt.figure(figsize=(6, 4))
sns.regplot(data=df, x=feature, y=target, scatter_kws={"alpha": 0.4})
plt.title(f"{feature} vs {target} (regplot)")
plt.show()
```

---

### 2.3. Pairplot (матрица рассеяния)

```python
cols = ["feature1", "feature2", "feature3", "target"]

sns.pairplot(df[cols], diag_kind="hist")
plt.show()
```

С цветом по классу (классификация):

```python
cols = ["feature1", "feature2", "feature3", "target"]
sns.pairplot(df[cols], hue="target", diag_kind="hist")
plt.show()
```

---

## 3. Корреляции: heatmap

```python
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
corr = df[num_cols].corr(method="pearson")

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0,
    annot=False,       # True, если нужны числа
    fmt=".2f",
    square=True,
)
plt.title("Матрица корреляций (Pearson)")
plt.show()
```

Только верхний треугольник (по желанию, чтобы не дублировать):

```python
mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    mask=mask,
    cmap="coolwarm",
    center=0,
    annot=False,
    fmt=".2f",
    square=True,
)
plt.title("Матрица корреляций (верхний треугольник)")
plt.show()
```

---

## 4. Визуализация групп и категорий

### 4.1. Barplot: частоты категорий

```python
col = "category_feature"

plt.figure(figsize=(8, 4))
df[col].value_counts().head(20).plot(kind="bar")
plt.title(f"Распределение категорий: {col}")
plt.xlabel(col)
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Версия через seaborn:

```python
col = "category_feature"

plt.figure(figsize=(8, 4))
sns.countplot(data=df, x=col, order=df[col].value_counts().index)
plt.title(f"Распределение категорий: {col}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

### 4.2. Barplot: средний таргет по категориям (классификация/регрессия)

```python
cat_col = "category_feature"
target = "target"

group_stats = df.groupby(cat_col)[target].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 4))
group_stats.head(20).plot(kind="bar")
plt.title(f"Средний {target} по категориям {cat_col}")
plt.xlabel(cat_col)
plt.ylabel(f"Mean {target}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

Тоже через seaborn:

```python
plt.figure(figsize=(8, 4))
sns.barplot(
    data=df,
    x=cat_col,
    y=target,
    estimator=np.mean,
    order=group_stats.index,  # сортировка по среднему
)
plt.title(f"Средний {target} по категориям {cat_col}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

### 4.3. Boxplot по категориям (распределение таргета)

```python
cat_col = "category_feature"
target = "target"

plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x=cat_col, y=target)
plt.title(f"{target} по категориям {cat_col}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 5. Дополнительные полезные графики

### 5.1. Распределение таргета (регрессия)

```python
target = "target"

plt.figure(figsize=(6, 4))
sns.histplot(df[target].dropna(), bins=30, kde=True)
plt.title(f"Распределение таргета: {target}")
plt.show()
```

---

### 5.2. Распределение таргета (классы)

```python
target = "target"

plt.figure(figsize=(6, 4))
df[target].value_counts(normalize=True).plot(kind="bar")
plt.title("Распределение классов (доли)")
plt.ylabel("Fraction")
plt.xticks(rotation=0)
plt.show()
```

---

### 5.3. Логарифмирование + сравнение распределений

```python
col = "feature"

df[f"log_{col}"] = np.log1p(df[col])  # log(1 + x)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(df[col].dropna(), bins=30, kde=True)
plt.title(col)

plt.subplot(1, 2, 2)
sns.histplot(df[f"log_{col}"].dropna(), bins=30, kde=True)
plt.title(f"log1p({col})")

plt.tight_layout()
plt.show()
```

---

## 6. Шаблоны для компоновки графиков (subplot)

### 6.1. Несколько графиков в одной фигуре

```python
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

sns.histplot(df["feature1"], bins=30, ax=axes[0])
axes[0].set_title("feature1")

sns.histplot(df["feature2"], bins=30, ax=axes[1])
axes[1].set_title("feature2")

plt.tight_layout()
plt.show()
```

---

### 6.2. FacetGrid (seaborn) по категориям

```python
g = sns.FacetGrid(df, col="category_feature", col_wrap=3, height=3, sharex=False, sharey=False)
g.map(sns.histplot, "numeric_feature", bins=20)
g.fig.suptitle("Распределение numeric_feature по категориям", y=1.02)
plt.show()
```

---

## 7. Быстрый словарь “что использовать, когда”

- **Посмотреть форму распределения признака**:
  - `sns.histplot` (с/без KDE),
  - `sns.kdeplot` (если нужно только плотность),
  - `sns.boxplot` (выбросы).
- **Проверить линейность связи с таргетом (регрессия)**:
  - `sns.scatterplot(x=feature, y=target)`,
  - `sns.regplot(x=feature, y=target)` (с линией).
- **Мульти‑фичевые взаимодействия**:
  - `sns.pairplot` для нескольких признаков.
- **Корреляции**:
  - heatmap: `sns.heatmap(df[num_cols].corr())`.
- **Категории**:
  - `sns.countplot` / `value_counts().plot("bar")` — частоты,
  - `sns.barplot(x=cat, y=target)` — средний таргет по категориям,
  - `sns.boxplot(x=cat, y=target)` — распределение таргета по категориям.

---

## 8. Мини‑шаблон полного EDA‑набора графиков

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid", rc={"figure.figsize": (8, 4)})

# 1. Распределение числовых
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

# 2. Boxplot для числовых
plt.figure(figsize=(8, 4))
sns.boxplot(data=df[num_cols], orient="h")
plt.title("Boxplot для числовых признаков")
plt.show()

# 3. Частоты категориальных
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
for col in cat_cols:
    plt.figure(figsize=(8, 3))
    df[col].value_counts().head(20).plot(kind="bar")
    plt.title(col)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# 4. Корреляции
corr = df[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Корреляции")
plt.show()
```

Этот файл — как “словарь” готовых графиков: можно просто копировать нужные куски в ноутбук под конкретную задачу.
```