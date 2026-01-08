```markdown
# 02. Загрузка данных и базовый EDA

Цель: быстро загрузить данные, понять их структуру, тип задачи, качество данных и наметить план препроцессинга/моделирования.

---

## 1. Быстрый чек‑лист EDA

1. Загрузить данные (`pd.read_csv` и т.п.).
2. Посмотреть размер датасета (число объектов и признаков).
3. Посмотреть типы столбцов, пропуски, дубликаты.
4. Определить таргет и тип задачи (регрессия / классификация / time series).
5. Разбить признаки на числовые / категориальные / даты.
6. Посмотреть распределения числовых признаков, выбросы.
7. Посмотреть частоты категориальных признаков.
8. Посмотреть распределение таргета (и дисбаланс классов).
9. Посмотреть связи:
   - числовые с таргетом (scatter, corr),
   - категориальные с таргетом (boxplot / barplot / groupby).

---

## 2. Импорт библиотек (шаблон)

```python
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={"figure.figsize": (8, 4)})
```

---

## 3. Загрузка данных

### 3.1. Базовый рецепт

```python
df = pd.read_csv("data.csv")  # путь к файлу
df.shape  # (n_objects, n_features)
```

### 3.2. Типовые варианты `read_csv`

```python
df = pd.read_csv(
    "data.csv",
    sep=",",               # разделитель: ",", ";", "\t" и т.п.
    encoding="utf-8",      # "cp1251" для многих русских CSV
    na_values=["NA", "NaN", ""],  # что считать пропусками
)
```

### 3.3. Работа с датами и индексом

```python
df = pd.read_csv(
    "data.csv",
    parse_dates=["date_col"],     # парсить столбец как дату
    dayfirst=True,               # если формат dd/mm/yyyy
)

df = df.sort_values("date_col")  # важно для временных рядов
df = df.set_index("date_col")    # по необходимости
```

### 3.4. Большие файлы (опционально)

```python
# Прочитать только первые N строк
df = pd.read_csv("data.csv", nrows=100000)

# Чтение по частям
chunks = pd.read_csv("data.csv", chunksize=50000)
for chunk in chunks:
    # обработка chunk
    pass
```

---

## 4. Первичный осмотр датасета

```python
# Размер
df.shape         # (n_rows, n_cols)

# Несколько строк
df.head()
df.tail()
df.sample(5, random_state=42)

# Общая информация
df.info()

# Типы данных
df.dtypes

# Базовые статистики для числовых
df.describe()

# Для категориальных (object/category)
df.describe(include=["object"])
```

---

## 5. Пропуски и дубликаты (только диагностика)

```python
# Количество пропусков по столбцам
df.isna().sum()

# Доля пропусков по столбцам
df.isna().mean().sort_values(ascending=False)

# Строки без/с пропусками
df[df.isna().any(axis=1)].head()

# Дубликаты
df.duplicated().sum()          # сколько дублирующихся строк
df_no_dupl = df.drop_duplicates()
```

(Подробная обработка пропусков — в отдельной шпоре.)

---

## 6. Определение таргета и типа задачи

```python
target_col = "target"  # заменить на реальное имя
y = df[target_col]
X = df.drop(columns=[target_col])

y.head()
y.dtype
```

- **Классификация**: `y` — категориальный / дискретный (классы).
- **Регрессия**: `y` — вещественный признак.
- **Временной ряд**: `y` + упорядочивание по времени (`date_col`).

### 6.1. Распределение таргета

```python
# Для регрессии
y.hist(bins=30)
plt.title("Распределение таргета")
plt.show()

# Для классификации
y.value_counts(normalize=True)  # доли классов
y.value_counts().plot(kind="bar")
plt.title("Распределение классов")
plt.show()
```

---

## 7. Разделение признаков по типам

```python
# Числовые признаки
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Категориальные признаки
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

# Если есть даты (кроме индекса)
datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()
```

(Позже эти списки пригодятся для скейлинга / кодирования / пайплайна.)

---

## 8. Распределения признаков и выбросы

### 8.1. Гистограммы для всех числовых

```python
df[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### 8.2. Boxplot (поиск выбросов)

```python
for col in num_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(col)
    plt.show()
```

---

## 9. Категориальные признаки: частоты и редкие категории

```python
for col in cat_cols:
    print(f"=== {col} ===")
    print(df[col].value_counts().head(10))           # топ частот
    print("nunique:", df[col].nunique())             # число уникальных
    print()

    # График частот (ТОП-10 значений)
    plt.figure()
    df[col].value_counts().head(10).plot(kind="bar")
    plt.title(col)
    plt.show()
```

---

## 10. Связь признаков с таргетом

### 10.1. Числовые признаки vs таргет (регрессия)

```python
for col in num_cols:
    if col == target_col:
        continue
    plt.figure()
    sns.scatterplot(data=df, x=col, y=target_col, alpha=0.5)
    plt.title(f"{col} vs {target_col}")
    plt.show()
```

### 10.2. Категориальные признаки vs таргет (регрессия)

```python
for col in cat_cols:
    plt.figure(figsize=(10, 4))
    sns.boxplot(data=df, x=col, y=target_col)
    plt.title(f"{target_col} по категориям {col}")
    plt.xticks(rotation=45)
    plt.show()
```

### 10.3. Категориальные признаки vs таргет (классификация)

```python
for col in cat_cols:
    # Средняя доля "позитивного" класса по категориям
    # (если бинарная классификация и позитивный класс = 1)
    rates = df.groupby(col)[target_col].mean().sort_values(ascending=False)
    print(f"=== {col} ===")
    print(rates.head(10))
    plt.figure(figsize=(10, 4))
    rates.head(10).plot(kind="bar")
    plt.title(f"Доля класса 1 по категориям {col}")
    plt.xticks(rotation=45)
    plt.show()
```

---

## 11. Корреляции и heatmap (числовые признаки)

```python
corr = df[num_cols].corr(method="pearson")  # корреляция Пирсона
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Матрица корреляций (Pearson)")
plt.show()
```

Полезно:
- искать сильно коррелирующие признаки (мультиколлинеарность),
- смотреть связь признаков с таргетом (если таргет числовой).

---

## 12. Примеры полезных groupby‑агрегаций

```python
# Среднее и медиана таргета по категории
df.groupby("some_category")[target_col].agg(["mean", "median", "count"]).sort_values("mean", ascending=False)

# Среднее нескольких признаков по группам
df.groupby("some_category")[["feature1", "feature2"]].mean()
```

---

## 13. Минимальный шаблон EDA «под ключ»

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={"figure.figsize": (8, 4)})

# 1. Загрузка
df = pd.read_csv("data.csv")

# 2. Общая информация
print(df.shape)
print(df.head())
print(df.info())
print(df.describe())

# 3. Пропуски и дубликаты
print(df.isna().sum())
print("Duplicates:", df.duplicated().sum())

# 4. Таргет
target_col = "target"
y = df[target_col]
X = df.drop(columns=[target_col])

# 5. Типы признаков
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# 6. Распределения
X[num_cols].hist(bins=30, figsize=(12, 8))
plt.tight_layout()
plt.show()

for col in cat_cols:
    plt.figure(figsize=(8, 3))
    X[col].value_counts().head(10).plot(kind="bar")
    plt.title(col)
    plt.show()

# 7. Корреляции (если регрессия)
corr = df[num_cols + [target_col]].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.show()
```

(На экзамене можно не писать всё подряд, но этот шаблон легко резать на нужные куски.)
```