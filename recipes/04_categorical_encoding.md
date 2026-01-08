```markdown
# 04. Категории и кодирование

Цель: правильно закодировать категориальные признаки (порядковые и номинальные), не устроив утечку данных и не раздув размерность.

---

## 1. Чек‑лист работы с категориальными признаками

1. Найти категориальные признаки, посчитать:
   - число уникальных значений (cardinality),
   - наличие редких категорий.
2. Определить тип:
   - **порядковые** (есть естественный порядок: низкий/средний/высокий),
   - **номинальные** (нет порядка: город, цвет, модель).
3. Выбрать кодирование:
   - порядковые → OrdinalEncoding / mapping,
   - номинальные с малым числом категорий → One‑Hot Encoding (OHE),
   - номинальные с большим числом категорий → Target/Binary Encoding и т.п.
4. Настроить пайплайн:
   - fit энкодеров только на train,
   - избегать утечки при target encoding.
5. Проверить:
   - размерность признаков,
   - наличие Dummy Trap / мультиколлинеарности (если важно).

---

## 2. Быстрая диагностика категориальных признаков

```python
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
cat_cols
```

### 2.1. Cardinality и редкие категории

```python
for col in cat_cols:
    print(f"=== {col} ===")
    print("nunique:", df[col].nunique())
    print(df[col].value_counts(dropna=False).head(10))
    print()
```

Полезно:
- отметить признаки с **очень большим** числом уникальных значений (100+, 1000+),
- посмотреть хвост распределения: редкие категории можно объединить в “Other”.

---

## 3. Порядковые категории (Ordinal Encoding, mapping)

### 3.1. Простой mapping вручную

Пример: `education = ["primary", "secondary", "higher"]`.

```python
order = {
    "primary": 0,
    "secondary": 1,
    "higher": 2,
}

df["education_ord"] = df["education"].map(order)
```

Особенности:
- использовать **только если порядок имеет смысл**,
- аккуратно с пропусками (`NaN` останется `NaN` → потом импутируем).

### 3.2. OrdinalEncoder из sklearn

```python
from sklearn.preprocessing import OrdinalEncoder

ord_cols = ["education", "grade"]  # признаки с порядком

ordinal_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1,  # код для неизвестных категорий
)

df[ord_cols] = ordinal_encoder.fit_transform(df[ord_cols])
```

---

## 4. Номинальные категории и One‑Hot Encoding (OHE)

### 4.1. Базовый OHE через pandas

Подходит для очень простых случаев (НЕ через пайплайн → есть риск утечки).

```python
df_ohe = pd.get_dummies(
    df,
    columns=["city", "gender"],
    drop_first=False,   # drop_first=True, если боимся Dummy Trap
    dummy_na=False,     # добавить отдельную колонку для NaN
)
```

### 4.2. Официальный путь: OneHotEncoder + ColumnTransformer

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_pipe = Pipeline(steps=[])  # здесь можно добавить imputer, scaler

cat_pipe = Pipeline(steps=[
    ("ohe", OneHotEncoder(
        handle_unknown="ignore",  # не падать на новых категориях
        drop=None,                # или 'first', если нужно убрать одну колонку
        sparse_output=False,      # для удобства (sklearn>=1.2)
    ))
])

preprocess = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(random_state=42)),
])

model.fit(X_train, y_train)
```

---

## 5. Label Encoding: когда можно, а когда нельзя

`LabelEncoder` превращает категории в числа `0, 1, 2, ...`, **без учёта порядка**.

### 5.1. Использование для таргета (классификация)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df["target_str"])  # например, "yes"/"no" → 1/0
```

### 5.2. Почему опасно для признаков

- Линейные модели и KNN будут считать, что класс `2` “больше” класса `1`,
  что **неправда** для номинальных категорий.
- Для таких моделей **лучше OHE**.

### 5.3. Когда можно использовать Label/Ordinal encoding для признаков

- Для деревьев/бустинга (RandomForest, CatBoost, XGBoost) иногда используют Ordinal кодирование,
  если модель умеет работать с ним без предположения о порядке.
- На практике:
  - в sklearn лучше `OneHotEncoder` / TargetEncoder,
  - CatBoost умеет работать с “сырыми” категориальными признаками напрямую.

---

## 6. Target / Mean Encoding (Target Encoding)

Идея: заменить категорию на **среднее значение таргета** по этой категории.

Пример (бинарная классификация, таргет ∈ {0, 1}):

```python
mean_target = df.groupby("city")["target"].mean()
df["city_te"] = df["city"].map(mean_target)
```

Проблемы:
- **жуткая утечка данных**, если делать на всём датасете,
- переобучение на редкие категории (одна‑две строки).

### 6.1. Сглаживание (smoothing) — идея

Типовая формула:

\[
\text{TE}(category) = \frac{n \cdot \text{mean\_cat} + \alpha \cdot \text{global\_mean}}{n + \alpha}
\]

где:
- `n` — количество объектов в категории,
- `mean_cat` — средний таргет в категории,
- `global_mean` — общий средний таргет по всему train,
- `α` — параметр сглаживания (чем больше, тем сильнее тянет к общему среднему).

### 6.2. Использование `category_encoders.TargetEncoder`

```python
# pip install category_encoders
import category_encoders as ce
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

X = df.drop(columns=["target"])
y = df["target"]

cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

te = ce.TargetEncoder(
    cols=cat_cols,
    smoothing=1.0,          # сглаживание (чем больше, тем сильнее)
)

model = LogisticRegression(max_iter=1000)

pipe = Pipeline(steps=[
    ("te", te),
    ("model", model),
])

cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
```

Особенности:
- `TargetEncoder` **сам** внутри CV‑циклов делает fit только на train‑fold → меньше риска утечки.
- Для финального обучения на всём train просто делаем `pipe.fit(X_train, y_train)`.

---

## 7. Binary / Hash / другие энкодеры (category_encoders)

### 7.1. Binary Encoding (BinaryEncoder)

Идея: кодировать категории как бинарные числа → число признаков растёт медленнее, чем при OHE.

```python
import category_encoders as ce

bin_enc = ce.BinaryEncoder(cols=cat_cols)

X_bin = bin_enc.fit_transform(X)
X_bin.head()
```

Когда полезно:
- очень много категорий (id товаров, uid и т.п.),
- OHE сильно раздувает размерность → проклятие размерности.

### 7.2. HashingEncoder (если нужно)

Хеш‑кодирование без хранения словаря категорий. Может использоваться для потоковых данных / большого онлайн‑ML.

---

## 8. Проклятие размерности и Dummy Trap

### 8.1. Проклятие размерности

- При OHE число признаков = сумма по всем категорям (кол-во уникальных − 1 или без −1).
- Если категорий много (например, city с 5000 уникальных), то:
  - размер матрицы признаков огромный,
  - переобучение, рост времени обучения,
  - ухудшение обобщающей способности моделей, особенно линейных/KNN/SVM.

Способы борьбы:
- агрегация/объединение редких категорий,
- Target/Binary Encoding,
- удаление неинформативных признаков,
- PCA (в редких случаях, но для OHE не всегда полезно).

### 8.2. Dummy Trap

Dummy Trap = идеальная линейная зависимость между OHE‑признаками:
- если все категории закодированы, суммы OHE‑колонок = 1,
- линейные модели получают мультиколлинеарность (VIF → ∞).

Решения:
- `drop='first'` в `OneHotEncoder`,
- или удалить одну из колонок вручную,
- в деревьях и бустинге это обычно не страшно, но в линейной регрессии — проблема.

```python
ohe = OneHotEncoder(
    handle_unknown="ignore",
    drop="first",         # избегаем Dummy Trap
    sparse_output=False,
)
```

---

## 9. Работа с редкими категориями

### 9.1. Объединение в “Other”

```python
col = "city"
value_counts = df[col].value_counts()

# Пусть категории с частотой < threshold станут "Other"
threshold = 50
rare_cats = value_counts[value_counts < threshold].index

df[col] = df[col].replace(rare_cats, "Other")
```

Полезно:
- уменьшает количество столбцов при OHE,
- снижает переобучение на редких категориях.

---

## 10. Пайплайн: комбинируем разные энкодеры

Пример: числовые + OHE для малых категорий + TargetEncoding для “тяжёлых” признаков.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier

X = df.drop(columns=["target"])
y = df["target"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

# Допустим, разделим категориальные на:
low_card_cols = [c for c in cat_cols if df[c].nunique() <= 10]
high_card_cols = [c for c in cat_cols if df[c].nunique() > 10]

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

ohe_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# TargetEncoder нельзя просто так запихнуть в ColumnTransformer,
# проще сделать общий Pipeline (см. пример выше в разделе 6.2).

# Вариант: только OHE (если без TargetEncoding)
preprocess = ColumnTransformer(transformers=[
    ("num", num_pipe, num_cols),
    ("cat_ohe", ohe_pipe, cat_cols),
])

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(random_state=42)),
])

clf.fit(X, y)
```

Если нужен TargetEncoder:
- чаще всего его используют **во всём пайплайне сразу**, как первый шаг (см. раздел 6.2).

---

## 11. Минимальный шаблон: OHE + простой пайплайн

```python
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
y_pred = clf.predict(X_test)
```

Этот шаблон — основной “кирпичик” для большинства задач с категориальными признаками.
```