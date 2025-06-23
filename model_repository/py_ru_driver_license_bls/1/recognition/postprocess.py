import numpy as np


class ResultRecognition:
    def __init__(self, predict_word, symbol_scores, word_score, is_correct):
        """Результат распознавания

        Parameters
        ----------
        predict_word : str
            Распознанный текст
        symbol_scores : list(float)
            Скоры распознавания каждой буквы
        word_score : float
            Минимальный скор буквы
        is_correct : bool
            Флаг корректности распознавания
        """

        self.predict_word = predict_word
        self.symbol_scores = symbol_scores
        self.word_score = word_score  # min of symbol_scores
        self.is_correct = is_correct
    
    def to_dict(self):
        return {
            "predict_word": self.predict_word,
            "symbol_scores": self.symbol_scores,
            "word_score": self.word_score,
            "is_correct": self.is_correct 
        }


def postprocess(prediction, vocabulary, threshold, validation):
    """Постобработка результатов модели.

    Parameters
    ----------
    prediction : numpy.array
        Результат модели
    model : dict
        Конфиг модели из конфига сервиса
    validation : callable()
        Функция валидации

    Returns
    -------
    ResultRecognition
        Результат распознования поля паспорта
    """

    predict_word, symbol_scores, word_score = decode_predict_with_scores(prediction, vocabulary)
    is_correct = check_recognition_correct(predict_word, word_score, threshold, validation)
    result = ResultRecognition(predict_word=predict_word, symbol_scores=symbol_scores, 
                               word_score=word_score, is_correct=is_correct)
    return result


def decode_predict_with_scores(prediction, vocabulary):
    """Преобразование результатов детекции в текст

    Parameters
    ----------
    prediction : numpy.array
        Результаты модели распознования
    vocabulary : list(str)
        Словарь символов

    Returns
    -------
    tuple
        Текст, Скоры символов, Минимальный скор символов
    """
    prediction = prediction[0]
    # index and max score for each predicted character
    predicted_symbols = np.argmax(prediction, axis=1)
    predicted_scores = np.max(prediction, axis=1)

    word_symbols = []
    word_symbol_scores = []

    symbol_start_index = 0
    for index in range(len(predicted_symbols)):
        # is end of prediction or is end of repeating symbol
        if index == len(prediction) - 1 or predicted_symbols[index] != predicted_symbols[index + 1]:
            # is not blank
            if predicted_symbols[index] != len(vocabulary):
                symbol_scores = predicted_scores[symbol_start_index: index + 1]
                symbol_mean_score = np.mean(symbol_scores).item()
                word_symbol_scores.append(symbol_mean_score)
                word_symbols.append(predicted_symbols[index])
            symbol_start_index = index + 1

    word = ''
    for item in word_symbols:
        word += vocabulary[item]

    word_score = np.min(word_symbol_scores).item()

    return word, word_symbol_scores, word_score



def check_recognition_correct(predict_word, word_score, threshold, validation):
    """Проверка корректности распознавания с помощью порога

    Parameters
    ----------
    predict_word : str
        Текст распознавания
    word_score : float
        Скор распознавания слова
    threshold : float
        Порог скора распознавания
    validation : callable()
        Фнкция валидации распознавания

    Returns
    -------
    bool
        Флаг корректности распознавания
    """

    is_correct = word_score > threshold
    is_valid = validation(predict_word)

    return is_correct and is_valid
