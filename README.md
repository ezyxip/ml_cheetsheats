# 📚 Шпаргалка к экзамену по Machine Learning

> Репозиторий для быстрой подготовки к экзамену. Все важное собрано в одном месте.

## 🚨 Экзаменационный срочный поиск

**«Надо быстро найти ответ на вопрос про...»**

| Что спрашивают | Где искать |
|----------------|------------|
| **Теория, определения, формулы** | `recipes/19_formulas_and_definitions.md` |
| **Как обрабатывать пропуски?** | `recipes/03_missing_values_and_imputation.md` |
| **Как кодировать категории?** | `recipes/04_categorical_encoding.md` |
| **Как искать выбросы/аномалии?** | `recipes/05_outliers_and_anomalies.md` |
| **Как делать Feature Engineering?** | `recipes/06_feature_engineering_general.md` |
| **Как масштабировать признаки?** | `recipes/08_scaling_multicollinearity_pca.md` |
| **Как делить данные? Утечки?** | `recipes/09_splits_validation_and_leakage.md` |
| **Дисбаланс классов — что делать?** | `recipes/10_imbalanced_classes.md` |
| **Какую модель выбрать?** | `recipes/11_model_selection_and_baselines.md` |
| **Как тюнить гиперпараметры?** | `recipes/12_training_and_hyperparameter_tuning.md` |
| **Метрики для регрессии** | `recipes/13_metrics_regression_and_diagnostics.md` |
| **Метрики для классификации** | `recipes/14_metrics_classification_and_diagnostics.md` |
| **Как интерпретировать модель?** | `recipes/15_model_interpretation.md` |
| **Готовый пайплайн sklearn** | `recipes/16_pipelines_and_columntransformer.md` |
| **Код для визуализаций** | `recipes/17_visualizations_recipes.md` |


## 📚 Соответствие лекций и шпаргалок

| Лекция | Основные темы | Шпаргалки |
|--------|---------------|-----------|
| **Лекция 1** | Принципы ML, Bias-Variance, Data Leakage | 00, 09, 19 |
| **Лекция 2** | Препроцессинг, валидация, дисбаланс | 03, 04, 08, 09, 10 |
| **Лекция 3** | Пропуски, категории, KL-дивергенция | 03, 04, 19 |
| **Лекция 4** | Выбросы, аномалии, IQR, Isolation Forest | 05 |
| **Лекция 5** | Feature Engineering, scaling, мультиколлинеарность | 06, 08 |
| **Лекция 6** | Выбор модели, метрики, интерпретация | 11-15 |



---

> *P.S. Все шпаргалки следуют формату: «Когда применять → Алгоритм → Код → Подводные камни»*


Ссылочки для пересдачи:
https://colab.research.google.com/drive/1U3v2jjapGHk0cVIyI1x4UbWYdUs3JCp7?usp=sharing - Регрессия
https://colab.research.google.com/drive/1KhspWsn-lTCmbbZuQ9BneC4kFzev3_La?usp=sharing - Классификация



```
1. https://scikit-learn.org/stable/modules/model_evaluation.html     — ВСЕ метрики + scoring строки
2. https://scikit-learn.org/stable/modules/classes.html              — быстрый поиск любого класса
3. https://scikit-learn.org/stable/modules/ensemble.html             — Random Forest / GBM параметры
4. https://scikit-learn.org/stable/modules/cross_validation.html     — все виды валидации
5. https://scikit-learn.org/stable/modules/grid_search.html          — GridSearch / RandomSearch
```



| Что                                   | Ссылка                                                                                                    |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Карта выбора модели**               | [scikit-learn.org/stable/machine_learning_map](https://scikit-learn.org/stable/machine_learning_map.html) |
| **API Reference (всё в одном месте)** | [scikit-learn.org/stable/modules/classes.html](https://scikit-learn.org/stable/modules/classes.html)      |
| **User Guide (теория + код)**         | [scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)                |
| **Глоссарий параметров**              | [scikit-learn.org/stable/glossary.html](https://scikit-learn.org/stable/glossary.html)                    |


| Тема                                 | Ссылка                                                                                                                                                                                     |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Preprocessing (все трансформеры)** | [scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html)                                                                           |
| **StandardScaler**                   | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)         |
| **MinMaxScaler**                     | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)             |
| **RobustScaler**                     | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html)             |
| **OneHotEncoder**                    | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)           |
| **LabelEncoder**                     | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)             |
| **OrdinalEncoder**                   | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html)         |
| **PolynomialFeatures**               | [scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) |



| Тема                   | Ссылка                                                                                                                                                                   |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Imputation (обзор)** | [scikit-learn.org/stable/modules/impute.html](https://scikit-learn.org/stable/modules/impute.html)                                                                       |
| **SimpleImputer**      | [scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)       |
| **KNNImputer**         | [scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)             |
| **IterativeImputer**   | [scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html) |




| Тема                  | Ссылка                                                                                                                                                                       |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Pipeline (обзор)**  | [scikit-learn.org/stable/modules/compose.html](https://scikit-learn.org/stable/modules/compose.html)                                                                         |
| **Pipeline**          | [scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)                 |
| **ColumnTransformer** | [scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) |
| **make_pipeline**     | [scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html)       |



| Тема                         | Ссылка                                                                                                                                                                                       |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Cross-validation (обзор)** | [scikit-learn.org/stable/modules/cross_validation.html](https://scikit-learn.org/stable/modules/cross_validation.html)                                                                       |
| **train_test_split**         | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)   |
| **KFold**                    | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)                         |
| **StratifiedKFold**          | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)     |
| **cross_val_score**          | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)     |
| **cross_val_predict**        | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) |
| **LeaveOneOut**              | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html)             |



| Тема                      | Ссылка                                                                                                                                                                                         |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tuning (обзор)**        | [scikit-learn.org/stable/modules/grid_search.html](https://scikit-learn.org/stable/modules/grid_search.html)                                                                                   |
| **GridSearchCV**          | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)             |
| **RandomizedSearchCV**    | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) |
| **Все доступные scoring** | [scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)                                     |



| Тема                               | Ссылка                                                                                                                 |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Model evaluation (ВСЕ метрики)** | [scikit-learn.org/stable/modules/model_evaluation.html](https://scikit-learn.org/stable/modules/model_evaluation.html) |



| Метрика                     | Ссылка                                                                                                                                                                                   |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **accuracy_score**          | [scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html)                   |
| **precision_score**         | [scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html)                 |
| **recall_score**            | [scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)                       |
| **f1_score**                | [scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html)                               |
| **roc_auc_score**           | [scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)                     |
| **roc_curve**               | [scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)                             |
| **confusion_matrix**        | [scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)               |
| **classification_report**   | [scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)     |
| **precision_recall_curve**  | [scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html)   |
| **log_loss**                | [scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html)                               |
| **matthews_corrcoef**       | [scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)             |
| **balanced_accuracy_score** | [scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html) |
| **cohen_kappa_score**       | [scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html)             |



| Метрика                            | Ссылка                                                                                                                                                                                                 |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **r2_score**                       | [scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)                                             |
| **mean_squared_error**             | [scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html)                         |
| **mean_absolute_error**            | [scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html)                       |
| **mean_absolute_percentage_error** | [scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html) |



| Модель                         | Ссылка                                                                                                                                                                                           |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LogisticRegression**         | [scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)         |
| **KNeighborsClassifier**       | [scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)           |
| **SVC**                        | [scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)                                                         |
| **DecisionTreeClassifier**     | [scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)                 |
| **RandomForestClassifier**     | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)         |
| **GradientBoostingClassifier** | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) |
| **AdaBoostClassifier**         | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)                 |
| **GaussianNB**                 | [scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html)                           |
| **VotingClassifier**           | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)                     |
| **StackingClassifier**         | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html)                 |




| Модель                        | Ссылка                                                                                                                                                                                         |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LinearRegression**          | [scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)           |
| **Ridge**                     | [scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)                                 |
| **Lasso**                     | [scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)                                 |
| **ElasticNet**                | [scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)                       |
| **SVR**                       | [scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)                                                       |
| **KNeighborsRegressor**       | [scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)           |
| **DecisionTreeRegressor**     | [scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)                 |
| **RandomForestRegressor**     | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)         |
| **GradientBoostingRegressor** | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html) |
| **StackingRegressor**         | [scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)                 |




| Тема                          | Ссылка                                                                                                                                                                                                     |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Feature selection (обзор)** | [scikit-learn.org/stable/modules/feature_selection.html](https://scikit-learn.org/stable/modules/feature_selection.html)                                                                                   |
| **SelectKBest**               | [scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)                       |
| **RFE**                       | [scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)                                       |
| **mutual_info_classif**       | [scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html)       |
| **mutual_info_regression**    | [scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html) |
| **Permutation Importance**    | [scikit-learn.org/stable/modules/permutation_importance.html](https://scikit-learn.org/stable/modules/permutation_importance.html)                                                                         |





| Тема                        | Ссылка                                                                                                                                                                                         |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Visualizations (обзор)**  | [scikit-learn.org/stable/visualizations.html](https://scikit-learn.org/stable/visualizations.html)                                                                                             |
| **ConfusionMatrixDisplay**  | [scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html)         |
| **RocCurveDisplay**         | [scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.RocCurveDisplay.html)                       |
| **PrecisionRecallDisplay**  | [scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PrecisionRecallDisplay.html)         |
| **DecisionBoundaryDisplay** | [scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html) |
| **learning_curve**          | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html)         |
| **validation_curve**        | [scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.validation_curve.html)     |
| **Calibration curve**       | [scikit-learn.org/stable/modules/calibration.html](https://scikit-learn.org/stable/modules/calibration.html)                                                                                   |




| Тема                          | Ссылка                                                                                                           |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Линейные модели**           | [scikit-learn.org/stable/modules/linear_model.html](https://scikit-learn.org/stable/modules/linear_model.html)   |
| **SVM**                       | [scikit-learn.org/stable/modules/svm.html](https://scikit-learn.org/stable/modules/svm.html)                     |
| **Деревья**                   | [scikit-learn.org/stable/modules/tree.html](https://scikit-learn.org/stable/modules/tree.html)                   |
| **Ансамбли**                  | [scikit-learn.org/stable/modules/ensemble.html](https://scikit-learn.org/stable/modules/ensemble.html)           |
| **KNN**                       | [scikit-learn.org/stable/modules/neighbors.html](https://scikit-learn.org/stable/modules/neighbors.html)         |
| **Naive Bayes**               | [scikit-learn.org/stable/modules/naive_bayes.html](https://scikit-learn.org/stable/modules/naive_bayes.html)     |
| **Кластеризация**             | [scikit-learn.org/stable/modules/clustering.html](https://scikit-learn.org/stable/modules/clustering.html)       |
| **Decomposition (PCA и др.)** | [scikit-learn.org/stable/modules/decomposition.html](https://scikit-learn.org/stable/modules/decomposition.html) |


