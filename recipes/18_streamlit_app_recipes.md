```markdown
# 18. Быстрые приложения на Streamlit

Цель: быстро собрать веб‑интерфейс для модели:
- онлайн‑предсказание по введённым признакам,
- загрузка CSV и прогон через модель,
- простой EDA и визуализация важности признаков / SHAP.

---

## 0. Установка и запуск

### 0.1. Установка

```bash
pip install streamlit
```

(Плюс обычные зависимости: `pandas`, `scikit-learn`, `joblib`, `shap` и т.д.)

### 0.2. Структура

Простейший вариант:

```text
project/
  app.py
  model.pkl
  utils.py      (опционально)
  data/         (опционально)
```

### 0.3. Запуск

Из папки с `app.py`:

```bash
streamlit run app.py
```

---

## 1. Минимальный шаблон приложения

`app.py`:

```python
import streamlit as st

def main():
    st.title("Моё первое Streamlit-приложение")
    st.write("Привет! Это простой пример.")

    name = st.text_input("Как вас зовут?")
    if st.button("Сказать привет"):
        st.write(f"Привет, {name or 'незнакомец'}!")

if __name__ == "__main__":
    main()
```

---

## 2. Рецепт: онлайн‑предсказание обученной моделью

### 2.1. Подготовка: сохранить модель после обучения

В ноутбуке/скрипте обучения:

```python
import joblib

# model = Pipeline(...) или просто модель
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
```

Можно сохранять и список признаков, если нужно.

---

### 2.2. Структура `app.py` для онлайн‑предсказаний

```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    # Загружаем модель (может быть sklearn Pipeline)
    model = joblib.load("model.pkl")
    return model

def main():
    st.title("Онлайн-предсказание модели")

    model = load_model()

    st.sidebar.header("Настройки")

    st.write("Введите значения признаков:")

    # ======== Пример для небольшой табличной задачи ========
    # Здесь надо руками описать нужные поля под вашу модель

    age = st.number_input("Возраст", min_value=0, max_value=120, value=30)
    income = st.number_input("Доход", min_value=0.0, value=50000.0, step=1000.0)
    gender = st.selectbox("Пол", ["male", "female"])
    city = st.selectbox("Город", ["Moscow", "SPb", "Other"])

    # Собираем в DataFrame с теми же именами столбцов, что были при обучении
    input_dict = {
        "age": [age],
        "income": [income],
        "gender": [gender],
        "city": [city],
    }
    X_input = pd.DataFrame(input_dict)

    if st.button("Сделать предсказание"):
        # Если модель — классификатор с predict_proba
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_input)[0, 1]
            y_pred = int(y_proba >= 0.5)

            st.write(f"Предсказанный класс: **{y_pred}**")
            st.write(f"Вероятность класса 1: **{y_proba:.3f}**")
        else:
            # Регрессия
            y_pred = model.predict(X_input)[0]
            st.write(f"Предсказание: **{y_pred:.3f}**")

if __name__ == "__main__":
    main()
```

Комментарии:

- Важно, чтобы **имена столбцов** в `X_input` совпадали с теми, что были на тренировке (особенно, если используется Pipeline с ColumnTransformer).

---

## 3. Рецепт: анализ загруженного файла (EDA + прогон через модель)

### 3.1. Загрузка модели (как выше)

```python
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")
```

### 3.2. `st.file_uploader` и чтение CSV

```python
import streamlit as st
import pandas as pd

def main():
    st.title("Анализ загруженного CSV + предсказания модели")

    model = load_model()

    uploaded_file = st.file_uploader("Загрузите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Первые строки файла")
        st.dataframe(df.head())

        st.subheader("Размер и общая информация")
        st.write(f"Форма: {df.shape[0]} строк, {df.shape[1]} столбцов")
        if st.checkbox("Показать .info()"):
            buffer = []
            df.info(buf=buffer)
            s = "\n".join(buffer)
            st.text(s)

        # Простые графики
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        if numeric_cols:
            col_to_plot = st.selectbox("Числовой признак для гистограммы", numeric_cols)
            st.bar_chart(df[col_to_plot].value_counts().sort_index())  # быстрый барчарт

        # Предсказания по всему датасету (если в файле нет таргета)
        if st.button("Посчитать предсказания для всех строк"):
            # Если в df есть лишние колонки (например, таргет), то нужно их отбросить
            # Здесь предположим, что модель ожидает такой же набор столбцов, как при обучении
            y_pred = model.predict(df)
            st.subheader("Примеры предсказаний")
            st.write(y_pred[:20])
            df_with_pred = df.copy()
            df_with_pred["prediction"] = y_pred
            st.download_button(
                label="Скачать CSV с предсказаниями",
                data=df_with_pred.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

if __name__ == "__main__":
    main()
```

Расширения:

- если в загруженном файле есть таргет → можно посчитать метрики (accuracy/ROC‑AUC/MAE и т.п.);
- добавить больше графиков (гистограммы, boxplot, корреляции) — из файла `visualizations_recipes`.

---

## 4. Визуализация важности признаков и SHAP в Streamlit

### 4.1. Показываем feature importance (деревья / бустинг)

Если модель имеет `feature_importances_`:

```python
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    importances = model.feature_importances_
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    fi.head(20).plot(kind="barh", ax=ax)
    ax.invert_yaxis()
    ax.set_title("Feature importance (top-20)")
    plt.tight_layout()
    return fig
```

В `app.py`:

```python
st.subheader("Важность признаков")

# Если model — Pipeline: достаём внутреннюю модель
base_model = model
if hasattr(model, "named_steps"):
    base_model = model.named_steps.get("model", model)

if hasattr(base_model, "feature_importances_"):
    # Если вы знаете итоговые feature_names после препроцессинга — используйте их
    # Для упрощения ниже предполагаем, что обучали без ColumnTransformer
    fig = plot_feature_importance(base_model, X_train.columns)
    st.pyplot(fig)
else:
    st.write("У модели нет атрибута feature_importances_.")
```

(В реальном проекте с ColumnTransformer надо отдельно собрать имена признаков — это уже более сложный трюк.)

---

### 4.2. SHAP‑графики в Streamlit (простой вариант)

#### 4.2.1. Расчёт SHAP (пример для дерева/бустинга)

```python
import shap

@st.cache_resource
def load_shap_explainer(model):
    explainer = shap.TreeExplainer(model)
    return explainer

def compute_shap_values(explainer, X_sample):
    # Лучше брать небольшой сэмпл (например, 200–500 объектов), чтобы быстро рисовалось
    shap_values = explainer.shap_values(X_sample)
    return shap_values
```

#### 4.2.2. Отображение summary_plot в Streamlit

SHAP summary_plot возвращает matplot‑figure, если использовать `matplotlib=True` (для force_plot) или можно просто позволить shap самому рисовать и “перехватить” текущую фигуру.

Простой способ — использовать `st.pyplot`:

```python
import matplotlib.pyplot as plt

st.subheader("SHAP summary plot")

# Допустим, у нас есть небольшой X_sample (DataFrame)
X_sample = X_train.sample(n=min(200, len(X_train)), random_state=42)

explainer = load_shap_explainer(base_model)
shap_values = explainer.shap_values(X_sample)

# Для бинарной классификации shap_values может быть списком
if isinstance(shap_values, list):
    shap_values_to_use = shap_values[1]
else:
    shap_values_to_use = shap_values

fig, ax = plt.subplots(figsize=(8, 4))
shap.summary_plot(shap_values_to_use, X_sample, show=False)  # show=False, чтобы не выводить вне Streamlit
st.pyplot(fig)
```

#### 4.2.3. Локальное объяснение одного объекта

```python
st.subheader("SHAP force plot для одного объекта")

idx = st.number_input("Индекс объекта из сэмпла", min_value=0, max_value=len(X_sample)-1, value=0, step=1)

# В Jupyter shap.force_plot интерактивный, но в Streamlit проще использовать matplotlib=True
fig = shap.force_plot(
    explainer.expected_value,
    shap_values_to_use[int(idx), :],
    X_sample.iloc[int(idx), :],
    matplotlib=True,
    show=False,
)
st.pyplot(fig)
```

(В реальном проде часто используют `st.components.v1.html` для встраивания интерактивного HTML из SHAP, но для шпоры достаточно `matplotlib=True`.)

---

## 5. Мини‑шаблон комбинированного приложения

Пример структуры `app.py` с двумя вкладками:
- “Онлайн‑предсказание”
- “CSV + EDA + предсказания”

```python
import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

def page_online_prediction(model):
    st.header("Онлайн-предсказание")

    age = st.number_input("Возраст", 0, 120, 30)
    income = st.number_input("Доход", 0.0, 1e9, 50000.0, step=1000.0)
    gender = st.selectbox("Пол", ["male", "female"])
    city = st.selectbox("Город", ["Moscow", "SPb", "Other"])

    X_input = pd.DataFrame({
        "age": [age],
        "income": [income],
        "gender": [gender],
        "city": [city],
    })

    if st.button("Предсказать"):
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)[0, 1]
            pred = int(proba >= 0.5)
            st.write(f"Класс: **{pred}**, P(1) = **{proba:.3f}**")
        else:
            pred = model.predict(X_input)[0]
            st.write(f"Предсказание: **{pred:.3f}**")

def page_file_analysis(model):
    st.header("Анализ CSV и пакетные предсказания")

    uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Первые строки:", df.head())

        if st.button("Сделать предсказания для всех строк"):
            y_pred = model.predict(df)
            df_out = df.copy()
            df_out["prediction"] = y_pred
            st.write("Примеры предсказаний:")
            st.write(df_out.head())

            st.download_button(
                label="Скачать CSV с предсказаниями",
                data=df_out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
            )

def main():
    st.title("ML-приложение на Streamlit")

    model = load_model()

    page = st.sidebar.selectbox(
        "Страница",
        ("Онлайн-предсказание", "CSV-анализ и пакетные предсказания"),
    )

    if page == "Онлайн-предсказание":
        page_online_prediction(model)
    else:
        page_file_analysis(model)

if __name__ == "__main__":
    main()
```

Этого файла достаточно, чтобы:
- быстро поднять веб‑интерфейс для вашей sklearn‑модели,
- дать возможность пользователю ввести признаки/загрузить CSV,
- показать базовые визуализации и интерпретацию (feature importance / SHAP).
```