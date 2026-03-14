
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# --- Модели для примера (можно заменить на другие) ---
# Регрессоры
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# Классификаторы
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Для деплоя (пример, может потребоваться установка)
# import joblib
# from flask import Flask, request, jsonify

print("Начало выполнения шаблона ML проекта")

# ==============================================================================
# 0. Загрузка данных (Начальный этап, не входит в 8, но необходим)
# ==============================================================================
print("\n--- 0. Загрузка данных ---")
# Замените 'your_dataset.csv' на путь к вашему файлу данных
# Убедитесь, что файл находится в той же директории или укажите полный путь
try:
    # Пример загрузки CSV. Для других форматов используйте pd.read_excel, pd.read_sql и т.д.
    df = pd.read_csv('your_dataset.csv') 
    print("Данные успешно загружены. Первые 5 строк:")
    print(df.head())
    print("\nИнформация о данных:")
    df.info()
except FileNotFoundError:
    print("Ошибка: Файл 'your_dataset.csv' не найден. Пожалуйста, укажите корректный путь к вашему датасету.")
    print("Для демонстрации будет создан фиктивный датасет.")
    # Создаем фиктивный датасет для демонстрации, если файл не найден
    data = {
        'Feature1': np.random.rand(100) * 100,
        'Feature2': np.random.randint(0, 2, 100),
        'Feature3': np.random.randn(100),
        'CategoricalFeature': np.random.choice(['A', 'B', 'C'], 100),
        'Target': np.random.rand(100) * 50 + np.random.randint(0, 2, 100) * 20 # Пример для регрессии
    }
    df = pd.DataFrame(data)
    # Добавим немного пропущенных значений для демонстрации предобработки
    df.loc[df.sample(frac=0.1).index, 'Feature1'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'CategoricalFeature'] = np.nan
    # Для демонстрации классификации, если Target - бинарный
    # df['Target_Class'] = np.random.randint(0, 2, 100)
    # df['Target'] = df['Target_Class'] # Если целевая переменная для классификации
    print("Фиктивный датасет создан. Первые 5 строк:")
    print(df.head())
    print("\nИнформация о фиктивном датасете:")
    df.info()

# Определите целевую переменную и признаки
TARGET_COLUMN = 'Target' # Замените на имя вашей целевой переменной
# Если задача классификации, убедитесь, что TARGET_COLUMN содержит дискретные значения
# Если задача регрессии, убедитесь, что TARGET_COLUMN содержит непрерывные значения

if TARGET_COLUMN not in df.columns:
    print(f"Ошибка: Целевая переменная '{TARGET_COLUMN}' не найдена в датасете.")
    print("Пожалуйста, проверьте имя целевой переменной или создайте ее.")
    # Пример создания целевой переменной для демонстрации, если ее нет
    df[TARGET_COLUMN] = np.random.rand(len(df)) * 100
    print(f"Создана фиктивная целевая переменная '{TARGET_COLUMN}'.")

# Определяем тип задачи (регрессия или классификация) на основе целевой переменной
is_classification = df[TARGET_COLUMN].nunique() < 20 and df[TARGET_COLUMN].dtype in ['int64', 'object', 'bool']
print(f"\nОпределен тип задачи: {'Классификация' if is_classification else 'Регрессия'}")

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if is_classification else None)
print(f"\nРазмер тренировочной выборки: {X_train.shape[0]} строк")
print(f"Размер тестовой выборки: {X_test.shape[0]} строк")

# ==============================================================================
# 1. Предобработка данных
# ==============================================================================
print("\n--- 1. Предобработка данных ---")

# Выделение числовых и категориальных признаков
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include='object').columns.tolist()

print(f"Числовые признаки: {numerical_features}")
print(f"Категориальные признаки: {categorical_features}")

# Создание пайплайнов для предобработки
# Числовые признаки: заполнение пропусков медианой, затем стандартизация
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Категориальные признаки: заполнение пропусков наиболее частым значением, затем One-Hot кодирование
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Объединение пайплайнов с помощью ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Применение предобработки к данным
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Предобработка данных завершена.")
print(f"Размерность данных после предобработки (тренировочная): {X_train_processed.shape}")
print(f"Размерность данных после предобработки (тестовая): {X_test_processed.shape}")

# ==============================================================================
# 2. Исследовательский анализ данных (EDA)
# ==============================================================================
print("\n--- 2. Исследовательский анализ данных (EDA) ---")

# Статистическое описание данных
print("\nСтатистическое описание числовых признаков:")
print(df[numerical_features].describe())

print("\nРаспределение категориальных признаков:")
for col in categorical_features:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Визуализация распределения целевой переменной
plt.figure(figsize=(8, 6))
sns.histplot(y, kde=True)
plt.title(f'Распределение целевой переменной: {TARGET_COLUMN}')
plt.xlabel(TARGET_COLUMN)
plt.ylabel('Частота')
plt.show()

# Матрица корреляций для числовых признаков
if numerical_features:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[numerical_features + [TARGET_COLUMN]].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Матрица корреляций числовых признаков и целевой переменной')
    plt.show()

# Анализ взаимосвязи категориальных признаков с целевой переменной (для классификации/регрессии)
for col in categorical_features:
    plt.figure(figsize=(10, 6))
    if is_classification:
        sns.countplot(data=df, x=col, hue=TARGET_COLUMN)
        plt.title(f'Распределение {col} по классам {TARGET_COLUMN}')
    else:
        sns.boxplot(data=df, x=col, y=TARGET_COLUMN)
        plt.title(f'Взаимосвязь {col} и {TARGET_COLUMN}')
    plt.xlabel(col)
    plt.ylabel(TARGET_COLUMN)
    plt.show()

print("EDA завершен. Проверьте графики и статистику для получения инсайтов.")

# ==============================================================================
# 3. Конструирование и выбор признаков
# ==============================================================================
print("\n--- 3. Конструирование и выбор признаков ---")

# Этот этап сильно зависит от конкретного датасета и задачи.
# Здесь приведены общие примеры, которые нужно адаптировать.

# Пример конструирования признаков (создание нового признака)
# df['New_Feature'] = df['Feature1'] * df['Feature3'] # Пример умножения двух числовых признаков
# print("Создан новый признак 'New_Feature'.")

# Пример выбора признаков (можно использовать методы отбора признаков)
# from sklearn.feature_selection import SelectKBest, f_classif, f_regression
# if is_classification:
#     selector = SelectKBest(f_classif, k='all')
# else:
# #     selector = SelectKBest(f_regression, k='all')
# selector.fit(X_train_processed, y_train)
# selected_features_indices = selector.get_support(indices=True)
# print(f"Выбрано {len(selected_features_indices)} признаков после отбора.")
# X_train_selected = X_train_processed[:, selected_features_indices]
# X_test_selected = X_test_processed[:, selected_features_indices]
# print("Внимание: Для использования SelectKBest нужно будет преобразовать X_train_processed обратно в DataFrame с именами колонок или использовать его после ColumnTransformer.")

print("Этап конструирования и выбора признаков требует ручной работы и анализа EDA.")
print("Для простоты в шаблоне используются все предобработанные признаки.")

# Используем предобработанные данные для дальнейших шагов
X_train_final = X_train_processed
X_test_final = X_test_processed

# ==============================================================================
# 4. Выбор и обоснование трех регрессоров или классификаторов
# ==============================================================================
print("\n--- 4. Выбор и обоснование трех моделей ML ---")

models = {}

if is_classification:
    print("Задача: Классификация. Выбираем классификаторы.")
    models['Logistic Regression'] = LogisticRegression(random_state=42, solver='liblinear')
    models['Decision Tree Classifier'] = DecisionTreeClassifier(random_state=42)
    models['Random Forest Classifier'] = RandomForestClassifier(random_state=42)
    # models['SVC'] = SVC(random_state=42, probability=True) # SVC может быть медленным на больших данных
    # models['KNeighborsClassifier'] = KNeighborsClassifier()
else:
    print("Задача: Регрессия. Выбираем регрессоры.")
    models['Linear Regression'] = LinearRegression()
    models['Decision Tree Regressor'] = DecisionTreeRegressor(random_state=42)
    models['Random Forest Regressor'] = RandomForestRegressor(random_state=42)
    # models['Gradient Boosting Regressor'] = GradientBoostingRegressor(random_state=42)

print("Выбраны следующие модели:")
for name in models:
    print(f"- {name}")

# Обоснование выбора:
print("\nОбоснование выбора моделей:")
print("1. **Линейная/Логистическая регрессия:** Простые, интерпретируемые модели, служат хорошим бейзлайном. Помогают понять линейные зависимости.")
print("2. **Дерево решений:** Нелинейная модель, способная улавливать сложные взаимодействия. Легко интерпретируется (особенно небольшие деревья).")
print("3. **Случайный лес:** Ансамблевый метод, основанный на деревьях решений. Обладает высокой точностью, устойчив к переобучению и хорошо работает с различными типами данных. Отличный выбор для большинства задач.")

# ==============================================================================
# 5. Гиперпараметрическая настройка модели ML
# ==============================================================================
print("\n--- 5. Гиперпараметрическая настройка модели ML ---")

# Определяем метрику для оптимизации
scoring_metric = 'roc_auc' if is_classification else 'neg_mean_squared_error'
print(f"Метрика для оптимизации гиперпараметров: {scoring_metric}")

best_models = {}

for name, model in models.items():
    print(f"\nНастройка гиперпараметров для {name}...")
    param_grid = {}
    if name == 'Logistic Regression':
        param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
    elif name == 'Decision Tree Classifier' or name == 'Decision Tree Regressor':
        param_grid = {'max_depth': [None, 5, 10, 20], 'min_samples_leaf': [1, 5, 10]}
    elif name == 'Random Forest Classifier' or name == 'Random Forest Regressor':
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    # Добавьте param_grid для других моделей, если они используются

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring=scoring_metric, n_jobs=-1, verbose=1)
        grid_search.fit(X_train_final, y_train)
        best_models[name] = grid_search.best_estimator_
        print(f"Лучшие гиперпараметры для {name}: {grid_search.best_params_}")
        print(f"Лучшая оценка (CV) для {name}: {grid_search.best_score_:.4f}")
    else:
        print(f"Для {name} не задана сетка гиперпараметров. Используется модель по умолчанию.")
        best_models[name] = model.fit(X_train_final, y_train)

# ==============================================================================
# 6. Сравнение метрик нескольких моделей ML
# ==============================================================================
print("\n--- 6. Сравнение метрик нескольких моделей ML ---")

results = []

for name, model in best_models.items():
    y_pred = model.predict(X_test_final)
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        try:
            y_proba = model.predict_proba(X_test_final)
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr') # 'ovr' для мультиклассовой, 'raise' для бинарной
        except AttributeError: # Если модель не поддерживает predict_proba (например, некоторые SVM без probability=True)
            roc_auc = np.nan
            print(f"Предупреждение: Модель {name} не поддерживает predict_proba для ROC AUC.")
        results.append({'Model': name, 'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'ROC AUC': roc_auc})
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        results.append({'Model': name, 'MSE': mse, 'RMSE': rmse, 'R2-Score': r2})

results_df = pd.DataFrame(results)
print("\nРезультаты сравнения моделей на тестовой выборке:")
print(results_df.round(4))

# Определение лучшей модели
if is_classification:
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax()]['Model'] # Можно выбрать другую метрику
else:
    best_model_name = results_df.loc[results_df['RMSE'].idxmin()]['Model'] # Можно выбрать другую метрику

best_model = best_models[best_model_name]
print(f"\nЛучшая модель по выбранной метрике: {best_model_name}")

# ==============================================================================
# 7. Интерпретация результатов работы модели ML и оценка её обобщающей способности
# ==============================================================================
print("\n--- 7. Интерпретация результатов и оценка обобщающей способности ---")

print(f"\nИнтерпретация лучшей модели: {best_model_name}")

if is_classification:
    print("\nОтчет по классификации для лучшей модели:")
    y_pred_best = best_model.predict(X_test_final)
    print(classification_report(y_test, y_pred_best, zero_division=0))

    print("\nМатрица ошибок (Confusion Matrix) для лучшей модели:")
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Матрица ошибок для {best_model_name}')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.show()

    # Важность признаков для древовидных моделей
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_, index=preprocessor.get_feature_names_out())
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title(f'Топ-10 важных признаков для {best_model_name}')
        plt.show()
    elif hasattr(best_model, 'coef_'): # Для линейных моделей
        if isinstance(best_model.coef_, np.ndarray) and best_model.coef_.ndim > 1:
            # Для мультиклассовой логистической регрессии, где coef_ имеет форму (n_classes, n_features)
            print("Коэффициенты для мультиклассовой логистической регрессии не так просто интерпретировать напрямую.")
        else:
            coefficients = pd.Series(best_model.coef_, index=preprocessor.get_feature_names_out())
            coefficients.nlargest(10).plot(kind='barh')
            plt.title(f'Топ-10 коэффициентов для {best_model_name}')
            plt.show()

else:
    print("\nПредсказания лучшей модели на тестовой выборке (первые 10):")
    y_pred_best = best_model.predict(X_test_final)
    print(pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_best}).head(10))

    # Важность признаков для древовидных моделей
    if hasattr(best_model, 'feature_importances_'):
        feature_importances = pd.Series(best_model.feature_importances_, index=preprocessor.get_feature_names_out())
        feature_importances.nlargest(10).plot(kind='barh')
        plt.title(f'Топ-10 важных признаков для {best_model_name}')
        plt.show()
    elif hasattr(best_model, 'coef_'): # Для линейных моделей
        coefficients = pd.Series(best_model.coef_, index=preprocessor.get_feature_names_out())
        coefficients.nlargest(10).plot(kind='barh')
        plt.title(f'Топ-10 коэффициентов для {best_model_name}')
        plt.show()

print("\nОценка обобщающей способности:")
print("Обобщающая способность модели оценивается по ее производительности на тестовой выборке (метрики из п.6).")
print("Если метрики на тестовой выборке значительно хуже, чем на тренировочной (или при кросс-валидации), это может указывать на переобучение.")
print("Визуализация ошибок (например, остатков для регрессии или матрицы ошибок для классификации) помогает понять, где модель ошибается.")

# ==============================================================================
# 8. Деплой лучшей модели ML
# ==============================================================================
print("\n--- 8. Деплой лучшей модели ML ---")

# Сохранение лучшей модели и препроцессора
# try:
#     joblib.dump(best_model, 'best_model.pkl')
#     joblib.dump(preprocessor, 'preprocessor.pkl')
#     print("Лучшая модель и препроцессор сохранены как 'best_model.pkl' и 'preprocessor.pkl'.")
# except NameError:
#     print("Модуль 'joblib' не установлен. Установите его: pip install joblib")

print("\nПример кода для деплоя (Flask API):")
print("""
# app.py
# from flask import Flask, request, jsonify
# import joblib
# import pandas as pd

# app = Flask(__name__)

# # Загрузка модели и препроцессора
# try:
#     model = joblib.load('best_model.pkl')
#     preprocessor = joblib.load('preprocessor.pkl')
#     print("Модель и препроцессор успешно загружены для деплоя.")
# except FileNotFoundError:
#     print("Ошибка: Файлы модели или препроцессора не найдены. Сначала обучите и сохраните их.")
#     model = None
#     preprocessor = None

# @app.route('/predict', methods=['POST'])
# def predict():
#     if model is None or preprocessor is None:
#         return jsonify({'error': 'Модель или препроцессор не загружены.'}), 500

#     try:
#         json_ = request.json
#         # Преобразование входных данных в DataFrame
#         query_df = pd.DataFrame(json_)

#         # Применение предобработки
#         query_processed = preprocessor.transform(query_df)

#         # Предсказание
#         prediction = model.predict(query_processed)

#         # Если классификация, можно вернуть вероятности
#         # if hasattr(model, 'predict_proba'):
#         #     probabilities = model.predict_proba(query_processed).tolist()
#         #     return jsonify({'prediction': prediction.tolist(), 'probabilities': probabilities})

#         return jsonify({'prediction': prediction.tolist()})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     # Для запуска локально: python app.py
#     # app.run(debug=True, host='0.0.0.0', port=5000)
#     print("Для запуска Flask-приложения используйте 'python app.py' в терминале.")
#     print("Убедитесь, что 'best_model.pkl' и 'preprocessor.pkl' существуют.")
""")

print("\nШаблон кода для решения задач ML на экзамене готов.")
print("Не забудьте заменить 'your_dataset.csv' и 'Target' на ваши реальные данные.")
print("Также адаптируйте параметры моделей и сетки для GridSearchCV под вашу задачу.")
