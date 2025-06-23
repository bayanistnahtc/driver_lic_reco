from typing import Dict, List


def compute_accuracy(df_real, preds) -> Dict[str, List[float]]:
    """
    Входные значения:
        df_real: Датафрейм, который содержит реальные значения
        preds: предсказания модели
    Выходные значения:
        accuracy: Словарь, который показывает точность для всех колонок
    """
    assert df_real["c_guid"].tolist() == list(preds["c_guid"]), "Не совпадают айди"
    assert set(df_real.columns.tolist()) == set(
        list(preds.keys())
    ), "Все колонки должны совпадать!"

    cols = df_real.columns.tolist()
    accuracy = {}
    for col in cols:
        for inx in range(len(df_real)):
            if accuracy.get(col):
                accuracy[col] += 1 if df_real.loc[inx, col] == preds[col][inx] else 0
            else:
                accuracy[col] = 1 if df_real.loc[inx, col] == preds[col][inx] else 0
        accuracy[col] /= len(df_real)
    return accuracy
