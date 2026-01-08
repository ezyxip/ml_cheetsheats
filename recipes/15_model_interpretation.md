```markdown
# 15. Интерпретация моделей

Цель: понимать, **почему** модель принимает решения:
- какие признаки в целом важны (глобально),
- какие признаки повлияли на конкретное предсказание (локально),
- использовать интерпретацию для проверки здравого смысла и поиска утечек.

---

## 1. Глобальная интерпретация

### 1.1. Feature importance для деревьев/ансамблей

Почти все модели на деревьях (DecisionTree, RandomForest, GradientBoosting, XGBoost/LightGBM/CatBoost) умеют показывать фичерные важности.

#### 1.1.1. sklearn: RandomForest / GradientBoosting

```python
import pandas as pd
import numpy as np

model.fit(X_train, y_train)

importances = model.feature_importances_
feat_names = X_train.columns

fi = pd.Series(importances, index=feat_names).sort_values(ascending=False)
print(fi.head(20))

# простой барплот
fi.head(20).plot(kind="barh", figsize=(6, 8))
plt.gca().invert_yaxis()
plt.title("Feature importance (sklearn)")
plt.show()
```

Особенности:
- это т.н. “Gini importance” / “gain” (зависит от того, как много “улучшений” внес признак),
- может быть смещён в пользу признаков с большим числом уникальных значений,
- не всегда надёжен → лучше смотреть ещё permutation importance / SHAP.

---

### 1.2. Коэффициенты линейных моделей

Для линейной регрессии/логистической регрессии:

\[
\hat{y} = w_0 + w_1 x_1 + \dots + w_p x_p
\]

- знак коэффициента: в какую сторону влияет признак;
- модуль: сила влияния (при условии масштабирования).

#### 1.2.1. Линейная регрессия

```python
from sklearn.linear_model import LinearRegression
import pandas as pd

linreg = LinearRegression()
linreg.fit(X_train, y_train)

coef = pd.Series(linreg.coef_, index=X_train.columns).sort_values()
print(coef)

# Топ по модулю
coef_abs = coef.abs().sort_values(ascending=False)
print(coef_abs.head(20))
```

#### 1.2.2. Логистическая регрессия

При логистической регрессии коэффициент \(w_j\) — это изменение **логарифма шанса** (log-odds).

```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

coef = pd.Series(logreg.coef_[0], index=X_train.columns).sort_values()
print(coef)
```

Интерпретация:
- \(w_j > 0\): рост признака увеличивает шанс класса 1,
- \(w_j < 0\): рост признака уменьшает шанс класса 1,
- чем больше |w_j|, тем существеннее влияние (если признаки отсcaled).

Важно:
- для осмысленной интерпретации:
  - использовать масштабирование (StandardScaler),
  - не иметь сильной мультиколлинеарности.

---

### 1.3. Permutation importance (перестановочная важность)

Идея:
- перемешиваем значения одного признака, измеряем, насколько падает метрика;
- если метрика сильно ухудшилась → признак важен.

```python
from sklearn.inspection import permutation_importance

model.fit(X_train, y_train)

result = permutation_importance(
    model,
    X_valid, y_valid,
    n_repeats=10,
    random_state=42,
    scoring="roc_auc",   # или любая другая метрика
)

perm_imp = pd.Series(result.importances_mean, index=X_valid.columns).sort_values(ascending=False)
print(perm_imp.head(20))

perm_imp.head(20).plot(kind="barh", figsize=(6, 8))
plt.gca().invert_yaxis()
plt.title("Permutation importance")
plt.show()
```

Плюсы:
- менее смещённая оценка, чем `feature_importances_`,
- подходит **для любой** модели (не только деревья).

Минусы:
- медленно (нужно много переоценок модели на permuted‑данных),
- требует отдельного val‑набора.

---

## 2. Локальная интерпретация: SHAP

### 2.1. Общая идея SHAP

SHAP (SHapley Additive exPlanations):

- разлагает предсказание модели для **конкретного объекта** на вклад каждого признака;
- базируется на теории кооперативных игр (значения Шепли);
- свойства:
  - аддитивность: сумма вкладов фич + базовый уровень = предсказание,
  - консистентность: более важные признаки получают большую важность.

Формально (упрощённо):

\[
f(x) \approx \phi_0 + \sum_{j=1}^{p} \phi_j
\]

где:
- \(f(x)\) — предсказание модели для объекта x,
- \(\phi_0\) — базовое значение (обычно среднее предсказание по train),
- \(\phi_j\) — вклад признака j.

---

### 2.2. Установка и базовый импорт

```bash
pip install shap
```

```python
import shap
import matplotlib.pyplot as plt
```

---

## 3. SHAP для разных моделей

### 3.1. RandomForest / другие деревья (sklearn)

```python
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)  # для классификации shap_values[1] — класс 1
```

Для бинарной классификации:

```python
# shap_values — список, 0 и 1 для классов
shap_values_class1 = shap_values[1]
```

---

### 3.2. XGBoost

```python
from xgboost import XGBClassifier
import shap

xgb = XGBClassifier(...)
xgb.fit(X_train, y_train)

explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(X_valid)
```

---

### 3.3. LightGBM

```python
from lightgbm import LGBMClassifier
import shap

lgbm = LGBMClassifier(...)
lgbm.fit(X_train, y_train)

explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_valid)
```

---

### 3.4. CatBoost

CatBoost имеет свои методы интерпретации, но SHAP тоже работает:

```python
from catboost import CatBoostClassifier
import shap

cat = CatBoostClassifier(..., verbose=0)
cat.fit(X_train, y_train, cat_features=cat_features_idx)

explainer = shap.TreeExplainer(cat)
shap_values = explainer.shap_values(X_valid)
```

CatBoost также умеет:

```python
cat.get_feature_importance(type="FeatureImportance")
cat.get_feature_importance(type="ShapValues", data=Pool(X_valid, y_valid, cat_features=cat_features_idx))
```

---

## 4. Виды SHAP‑графиков

Предположим:
- `X_valid` — DataFrame с валидационными признаками,
- `shap_values` — массив (n_samples × n_features) для нужного класса / регрессии.

### 4.1. Summary plot (глобальная картина)

Показывает:
- важность признаков (по вертикали, сверху — самые важные),
- распределение значений SHAP (влияние) по объектам,
- цвет точки — значение признака (красный = большое, синий = маленькое).

```python
shap.summary_plot(shap_values, X_valid, plot_type="dot")
```

Регрессия / бинарная классификация:
- `shap_values` — 2D (n_samples, n_features).

Бинарная классификация (TreeExplainer XGB/LGBM/RF):
- shap_values может быть списком → берём `shap_values[1]` для класса 1.

---

### 4.2. Bar summary plot (упрощённый)

Только глобальная важность признаков (среднее |SHAP|):

```python
shap.summary_plot(shap_values, X_valid, plot_type="bar")
```

---

### 4.3. Dependence plot (фича vs её SHAP + влияние других фичей)

Показывает, как значение признака влияет на вклад в предсказание.

```python
shap.dependence_plot("feature_name", shap_values, X_valid)
```

Можно указать `interaction_index`:

```python
shap.dependence_plot("feature_name", shap_values, X_valid, interaction_index="another_feature")
```

---

### 4.4. Force plot (локальное объяснение одного объекта)

Для одного объекта (например, i‑й валидационный пример):

```python
i = 0

shap.initjs()  # для Jupyter
shap.force_plot(
    explainer.expected_value,        # базовое значение
    shap_values[i, :],               # shap'ы для объекта
    X_valid.iloc[i, :],              # значения признаков
)
```

В Jupyter → интерактивная визуализация: красные/синие стрелки, какие фичи и насколько подтолкнули предсказание вверх/вниз.

Вне Jupyter (например, чтобы сохранить в HTML):

```python
force = shap.force_plot(
    explainer.expected_value,
    shap_values[i, :],
    X_valid.iloc[i, :],
    matplotlib=True
)
plt.show()
```

---

## 5. Как использовать интерпретацию на практике

### 5.1. Проверка здравого смысла

- Посмотреть глобальную важность признаков:
  - summary plot (bar),
  - feature importance (деревья),
  - permutation importance.
- Вопросы:
  - действительно ли самые важные признаки кажутся логичными для задачи?
  - нет ли признаков, которые не должны иметь такого влияния (например, ID)?

Примеры:

- Кредитный скоринг:
  - логично, что доход, долговая нагрузка, история платежей — в топе.
  - странно, если в топе, например, ID менеджера или индекс строки.

---

### 5.2. Поиск утечек (data leakage)

Если какой‑то признак:
- имеет **аномально высокую** важность,
- при этом по смыслу **не должен** сильно влиять на таргет,
- или “подозрительные” признаки (дата оформления договора, ID сгорающих купонов и т.п.)

→ стоит проверить:

- как он формируется (может использовать будущее или сам таргет?),
- нет ли утечки таргета в признаке (например, “флаг просрочки” в момент, когда мы ещё только хотим её предсказать).

Конкретный приём:
- временно удалить подозрительный признак,
- переобучить модель,
- посмотреть, как изменятся метрики.

---

### 5.3. Поиск странных фичей / улучшение фичей

Через SHAP:
- можно увидеть, как признак влияет на предсказания:
  - dependence plot: “когда фича растёт, предсказание растёт/падает?”,
  - есть ли пороги, насыщения, нелинейности.
- на основе этого:
  - скорректировать биннинг/нелинейные трансформации,
  - сделать новые производные признаки.

---

### 5.4. Локальное объяснение конкретных кейсов

С помощью force‑plot / `shap_values[i]`:

- объяснить, почему модель отказала в кредите именно этому клиенту;
- объяснить, почему система детектировала транзакцию как fraud;
- найти “странные” объекты, где:
  - модель сильно ошибается,
  - или опирается на неожиданные признаки.

Пример:

```python
i = 5
print("y_true:", y_valid.iloc[i])
print("y_pred_proba:", model.predict_proba(X_valid.iloc[[i]])[0, 1])

shap.force_plot(
    explainer.expected_value,
    shap_values[i, :],
    X_valid.iloc[i, :],
    matplotlib=True,
)
plt.show()
```

---

## 6. Мини‑чек‑лист по интерпретации

1. **Глобально**:
   - посмотреть feature importance (деревья/бустинг);
   - проверить permutation importance;
   - сделать SHAP summary plot (dot и bar).
2. **Локально**:
   - для интересующих объектов (ошибки, “краевые” кейсы) → force plot;
   - dependence plot для топ‑фич.
3. **Проверка здравого смысла**:
   - топ‑фичи логичны для предметной области?
   - нет ли ID/дат/вспомогательных величин, которые “знают будущее” или таргет?
4. **Если есть подозрения**:
   - отключить подозрительные фичи,
   - сравнить качество модели,
   - убедиться, что не полагаемся на утечку.

---

## 7. Мини‑рецепт SHAP “под ключ” для дерева/бустинга

```python
import shap
import matplotlib.pyplot as plt

# 1. Обучаем модель
model.fit(X_train, y_train)

# 2. Создаём explainer и считаем shap_values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_valid)

# Для регрессии: shap_values — 2D (n_samples, n_features)
# Для бинарной классификации: shp для класса 1
if isinstance(shap_values, list):  # XGBoost/LightGBM/RandomForest классификация
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

# 3. Глобальный summary plot
shap.summary_plot(shap_values_to_use, X_valid, plot_type="bar")   # важность фич

# 4. Детальный summary plot (dot)
shap.summary_plot(shap_values_to_use, X_valid, plot_type="dot")

# 5. Dependence plot для конкретного признака
shap.dependence_plot("feature_name", shap_values_to_use, X_valid)

# 6. Локальное объяснение для конкретного объекта
i = 0
shap.force_plot(
    explainer.expected_value,
    shap_values_to_use[i, :],
    X_valid.iloc[i, :],
    matplotlib=True
)
plt.show()
```

Этого достаточно, чтобы на экзамене и в практике продемонстрировать понимание:
- глобальной и локальной интерпретации,
- базовой работы SHAP,
- использования интерпретации для проверки моделей и поиска утечек.
```