"""Практика 8. Подготовка, сравнение и сохранение модели Iris.

    X: четыре измерения цветка в сантиметрах.
    y: species, название вида. Для обучения переводится в код 0, 1 или 2.
    Тест 20% проверяется после выбора модели по пяти блокам CV на train.
"""
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from baseline import PetalLengthRule

BASE = Path(__file__).resolve().parent
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
RANDOM_STATE = 42


def save_plot(name):
    plt.tight_layout()
    plt.savefig(BASE / 'figures' / name, dpi=160, bbox_inches='tight')
    plt.close()


def main():
    for name in ['figures', 'reports', 'models']:
        (BASE / name).mkdir(exist_ok=True)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    data = pd.read_csv(BASE / 'data' / 'iris.csv')
    # Проверяю структуру и пропуски вместо молчаливого удаления данных.
    if data[FEATURES + ['species']].isna().any().any():
        raise ValueError('В Iris не ожидаются пропуски. Проверьте файл данных.')
    X = data[FEATURES].astype(float)
    mapping = {name: number for number, name in enumerate(SPECIES)}
    y = data['species'].map(mapping)
    if y.isna().any():
        raise ValueError('В файле встретился неизвестный вид ириса.')
    y = y.astype(int)
    print('Размер:', data.shape, 'Пропуски:', int(data.isna().sum().sum()))
    print('Классы:', data['species'].value_counts().to_dict())
    # 120 примеров оставляю для обучения и CV, 30 — для итогового теста.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        'Правило по длине лепестка': PetalLengthRule(),
        'Логистическая регрессия': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])}
    comparison = []
    for name, estimator in candidates.items():
        # Pipeline обучает StandardScaler заново внутри каждого тренировочного блока.
        scores = cross_validate(estimator, X_train, y_train, cv=cv,
                                scoring={'f1': 'f1_macro', 'accuracy': 'accuracy'})
        comparison.append({'model': name, 'cv_macro_f1': float(scores['test_f1'].mean()),
                           'cv_f1_std': float(scores['test_f1'].std()),
                           'cv_accuracy': float(scores['test_accuracy'].mean())})
    comparison = pd.DataFrame(comparison).sort_values(
        ['cv_macro_f1', 'cv_f1_std'], ascending=[False, True])
    comparison.to_csv(BASE / 'reports' / 'cv_results.csv', index=False)
    print(comparison.to_string(index=False))
    best_name = comparison.iloc[0]['model']
    best = candidates[best_name]
    # Диагностирую ошибки на train с предсказаниями вне обучавшего их блока.
    oof = cross_val_predict(best, X_train, y_train, cv=cv)
    mistakes = X_train.copy()
    mistakes['actual'] = [SPECIES[i] for i in y_train]
    mistakes['predicted'] = [SPECIES[i] for i in oof]
    mistakes = mistakes[y_train.to_numpy() != oof]
    mistakes.to_csv(BASE / 'reports' / 'cv_errors.csv', index=False)
    ConfusionMatrixDisplay.from_predictions(y_train, oof, display_labels=SPECIES, values_format='d')
    plt.title('Ошибки кросс-валидации на train')
    save_plot('cv_confusion.png')
    # Теперь окончательно обучаю победителя и один раз оцениваю его на test.
    best.fit(X_train, y_train)
    predictions = best.predict(X_test)
    report = classification_report(y_test, predictions, target_names=SPECIES, output_dict=True)
    metrics = {'accuracy': float(accuracy_score(y_test, predictions)),
               'macro_f1': float(f1_score(y_test, predictions, average='macro')),
               'test_size': int(len(y_test)), 'test_errors': int(sum(y_test.to_numpy() != predictions))}
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, display_labels=SPECIES, values_format='d')
    plt.title('Итоговый тест после выбора модели')
    save_plot('test_confusion.png')
    plt.bar(comparison['model'], comparison['cv_macro_f1'], yerr=comparison['cv_f1_std'], capsize=5)
    plt.ylim(0, 1.05)
    plt.ylabel('Macro-F1 на CV')
    save_plot('model_comparison.png')
    for code, name in enumerate(SPECIES):
        part = data[data['species'] == name]
        plt.scatter(part['petal_length'], part['petal_width'], label=name)
    plt.xlabel('Длина лепестка, см'); plt.ylabel('Ширина лепестка, см'); plt.legend()
    save_plot('petal_distribution.png')
    payload = {'model': best, 'model_name': best_name, 'features': FEATURES, 'species': SPECIES,
               'train_min': X_train.min().to_dict(), 'train_max': X_train.max().to_dict(), 'metrics': metrics}
    with open(BASE / 'models' / 'iris_model.pkl', 'wb') as stream:
        pickle.dump(payload, stream)
    # Сохраняю примеры, чтобы выступление опиралось на реальные результаты.
    success = int(np.flatnonzero((predictions == y_test.to_numpy()) & (y_test.to_numpy() == 0))[0])
    demo = {'successful': {'values': X_test.iloc[success].tolist(), 'actual': SPECIES[int(y_test.iloc[success])]}}
    if hasattr(best, 'predict_proba'):
        probabilities = best.predict_proba(X_test)
        order = np.sort(probabilities, axis=1)
        border = int(np.argmin(order[:, -1] - order[:, -2]))
    else:
        border = int(np.argmin(abs(X_test['petal_length'].to_numpy() - X_train['petal_length'].median())))
    demo['borderline'] = {'values': X_test.iloc[border].tolist(),
                          'actual': SPECIES[int(y_test.iloc[border])],
                          'predicted': SPECIES[int(predictions[border])]}
    pattern = mistakes.groupby(['actual', 'predicted']).size().sort_values(ascending=False)
    common_error = 'На CV ошибок нет.' if pattern.empty else f'На CV чаще всего истинный вид {pattern.index[0][0]} определяется как {pattern.index[0][1]}.'
    summary = {'model': best_name, 'train_size': len(y_train), 'cv_results': comparison.to_dict('records'),
               'metrics': metrics, 'classification_report': report, 'cv_errors': len(mistakes),
               'error_pattern': common_error, 'demo': demo}
    (BASE / 'reports' / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Лучшая модель:', best_name)
    print('Итоговый тест:', metrics)
    print(classification_report(y_test, predictions, target_names=SPECIES))
    print(common_error)
    print('Модель сохранена. Результат на 30 цветках не гарантирует качество на любых новых растениях.')


if __name__ == '__main__':
    main()
