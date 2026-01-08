```markdown
# 13. Регрессия: метрики и диагностика

Цель: уметь считать и интерпретировать метрики регрессии, строить диагностические графики и проверять адекватность модели (остатки, гетероскедастичность).

---

## 1. Обозначения

- \( y_i \) — истинное значение таргета,
- \( \hat{y}_i \) — предсказанное значение,
- \( n \) — количество наблюдений.

Во всех формулах суммирование по i = 1..n.

---

## 2. Основные метрики регрессии

### 2.1. MAE — Mean Absolute Error

\[
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
\]

- Средняя **абсолютная** ошибка.
- Интерпретация: “в среднем модель ошибается на X единиц таргета”.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
```

---

### 2.2. MSE — Mean Squared Error

\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
\]

- Средняя **квадратичная** ошибка.
- Сильно штрафует крупные ошибки.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_true, y_pred)          # по умолчанию squared=True
```

---

### 2.3. RMSE — Root Mean Squared Error

\[
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
\]

- Корень из MSE.
- В **тех же единицах**, что и таргет.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = mean_squared_error(y_true, y_pred, squared=False)
# или
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
```

---

### 2.4. MAPE — Mean Absolute Percentage Error

\[
\text{MAPE} = \frac{100\%}{n} \sum_{i=1}^{n} \left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

- Средняя абсолютная **процентная** ошибка.
- Интерпретация: “в среднем ошибаемся на X%”.

Проблемы:
- нельзя делить на 0 (или очень маленькие \( y_i \)),
- сильно искажает метрику при малых значениях таргета.

```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred) * 100  # в процентах
```

(Если нет этой функции в версии sklearn — можно реализовать вручную.)

---

### 2.5. SMAPE — Symmetric MAPE

Один из вариантов:

\[
\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^{n} 
\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}
\]

- Более “симметричная” версия MAPE (ошибки для больших/малых значений не так дисбалансны).
- Часто используют во временных рядах.

Реализация вручную:

```python
import numpy as np

def smape(y_true, y_pred, eps=1e-9):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    smape_vals = np.abs(y_true - y_pred) / np.maximum(denom, eps)
    return np.mean(smape_vals) * 100
```

---

### 2.6. R² — коэффициент детерминации

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]
где \(\bar{y}\) — среднее по \(y_i\).

- 1 → идеальное предсказание,
- 0 → модель не лучше, чем просто предсказывать среднее \(\bar{y}\),
- может быть **отрицательным** (хуже среднего).

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
```

---

## 3. Диагностические графики

Цель диагностики: увидеть систематические ошибки модели (нелинейность, пропущенные фичи, гетероскедастичность и т.п.).

Обозначим:
- `y_true` — истинные значения,
- `y_pred` — предсказанные,
- `residuals = y_true - y_pred`.

```python
import numpy as np
residuals = y_true - y_pred
```

---

### 3.1. Predicted vs Actual (y_pred vs y_true)

Хорошая регрессионная модель:
- точки лежат **вдоль диагонали** \(y = x\),
- нет серьёзных систематических отклонений.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.scatter(y_true, y_pred, alpha=0.5)
plt.plot([y_true.min(), y_true.max()],
         [y_true.min(), y_true.max()],
         "r--", label="y = x")
plt.xlabel("Actual (y_true)")
plt.ylabel("Predicted (y_pred)")
plt.title("Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.show()
```

Что искать:
- смещение (всё выше/ниже диагонали) → систематическая ошибка,
- кривизна → нелинейность (линейная модель не справляется).

---

### 3.2. Остатки: Residuals vs Predicted

Считаем остатки:

```python
residuals = y_true - y_pred
```

Строим:

```python
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual (y_true - y_pred)")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()
```

Идеально:
- “облачко” точек случайно вокруг 0,
- нет явных паттернов.

Плохие признаки:
- видимый тренд, кривая (например, остатки растут с предсказанием) → модель не ловит нелинейность,
- “веер” (фан‑шэйп) → гетероскедастичность (см. ниже).

---

### 3.3. Распределение остатков

Смотрим на распределение остатков:

```python
import seaborn as sns

plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Распределение остатков")
plt.show()
```

- При классической линейной регрессии хорошо, если остатки примерно нормальны (но это не всегда критично).
- Сильная асимметрия / тяжёлые хвосты → могут быть выбросы или пропущенные важные фичи.

---

### 3.4. Проверка гетероскедастичности

**Гетероскедастичность** — дисперсия остатков меняется с уровнем предсказаний/признаков.

Признаки (по графику Residuals vs Predicted):

- В начале малые разбросы, затем всё шире (веер) → дисперсия растёт.
- Или наоборот.

Визуальный пример см. выше. Можно также:

```python
plt.figure(figsize=(6, 4))
plt.scatter(y_pred, np.abs(residuals), alpha=0.5)
plt.xlabel("Predicted")
plt.ylabel("|Residual|")
plt.title("|Residual| vs Predicted")
plt.grid(True)
plt.show()
```

Что делать при гетероскедастичности:

- применить преобразование таргета (например, логарифм: `log(y)`),
- использовать модели, устойчивые к гетероскедастичности (robust regression),
- использовать методы, учитывающие разную дисперсию ошибок (в продвинутой статистике).

---

## 4. Как интерпретировать метрики и сравнивать модели

### 4.1. Общие принципы

1. **Сравнивать только на одном и том же наборе данных**:
   - одна и та же выборка (или схема CV),
   - одинаковая метрика.

2. **Нужно учитывать масштаб таргета**:
   - MAE=10 плохо или нормально? зависит от того, таргет в районе 20 или 2000.

3. **Сравнивать с baseline**:
   - DummyRegressor (среднее/медиана),
   - простая линейная регрессия.
   - Если модель лишь немного лучше baseline → возможно, в данных мало сигнала.

---

### 4.2. Интерпретация конкретных метрик

**MAE / RMSE**:

- MAE = 5 → “в среднем ошибаемся на 5 единиц”.
- RMSE ≥ MAE (всегда).
- Большой разрыв между RMSE и MAE → есть крупные выбросы/ошибки.

**MAPE / SMAPE**:

- MAPE=10% → ошибки ~10% в среднем.
- Сравнивать проценты проще, чем абсолютные значения (особенно, если диапазон y широкий).
- Осторожно с нулями/маленькими y.

**R²**:

- R² ≈ 0.9 → модель объясняет 90% вариации таргета.
- R² ≈ 0 → модель не лучше, чем просто среднее.
- R² < 0 → модель хуже “просто среднего”.

---

### 4.3. Как выбирать метрику

- Если важны **абсолютные отклонения** → MAE.
- Если хотим сильнее наказывать большие ошибки → MSE / RMSE.
- Если важен **относительный** процент ошибки → MAPE / SMAPE (но следить за нулями).
- Для отчёта:
  - хорошо давать 2–3 метрики: MAE + RMSE + R² (например).

---

### 4.4. Сравнение моделей

Пример:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def print_regression_metrics(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2   = r2_score(y_true, y_pred)
    print(f"{name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

print_regression_metrics("Baseline", y_test, y_pred_baseline)
print_regression_metrics("Model A",  y_test, y_pred_a)
print_regression_metrics("Model B",  y_test, y_pred_b)
```

Смотрим:

- меньше MAE/RMSE → лучше,
- больше R² → лучше.

Важно смотреть и на **диагностические графики**:
- модель с чуть лучшей метрикой, но с явной нелинейной структурой в остатках → возможно, стоит сменить тип модели / добавить фичи.

---

## 5. Мини‑рецепт диагностики “под ключ”

После обучения модели:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Метрики
mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2   = r2_score(y_test, y_pred)

print(f"MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")

# 2. Predicted vs Actual
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Predicted vs Actual")
plt.grid(True)
plt.show()

# 3. Остатки
residuals = y_test - y_pred

plt.figure(figsize=(6, 4))
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted")
plt.grid(True)
plt.show()

# 4. Распределение остатков
plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color="red", linestyle="--")
plt.title("Распределение остатков")
plt.show()
```

Эти шаги обычно достаточны, чтобы:
- оценить качество модели,
- обнаружить основные проблемы (смещение, нелинейность, гетероскедастичность, выбросы).
```