from typing import Dict

fio_characters = [
    " ",
    "-",
    "А",
    "Б",
    "В",
    "Г",
    "Д",
    "Е",
    "Ж",
    "З",
    "И",
    "Й",
    "К",
    "Л",
    "М",
    "Н",
    "О",
    "П",
    "Р",
    "С",
    "Т",
    "У",
    "Ф",
    "Х",
    "Ц",
    "Ч",
    "Ш",
    "Щ",
    "Ъ",
    "Ы",
    "Ь",
    "Э",
    "Ю",
    "Я",
    "",
]
fio_characters = sorted(fio_characters)
fio_char_to_num = {char: idx for idx, char in enumerate(fio_characters)}
fio_num_to_char = {idx: char for char, idx in fio_char_to_num.items()}

date_characters = [" ", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
date_characters = sorted(date_characters)
date_char_to_num = {char: idx for idx, char in enumerate(date_characters)}
date_num_to_char = {idx: char for char, idx in date_char_to_num.items()}

serial_characters = [
    "",
    " ",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "K",
    "O",
    "P",
    "Y",
]
serial_characters = sorted(serial_characters)
serial_char_to_num = {char: idx for idx, char in enumerate(serial_characters)}
serial_num_to_char = {idx: char for char, idx in serial_char_to_num.items()}


def fio_indices_to_text(indices):
    return "".join([fio_num_to_char.get(i, "") for i in indices if i != -1])


def date_indices_to_text(indices):
    return "".join([date_num_to_char.get(i, "") for i in indices if i != -1])


def serial_indices_to_text(indices):
    return "".join([serial_num_to_char[letter] for letter in indices])


def ctc_decode(pred_indices, blank=0):
    decoded_batch = []
    for indices in pred_indices:
        decoded = []
        prev_idx = blank
        for idx in indices:
            if idx != blank and idx != prev_idx:
                decoded.append(idx.item())
            prev_idx = idx
        decoded_batch.append(decoded)
    return decoded_batch


def decode(output, nums_to_word):
    word = ctc_decode(output.argmax(dim=-1))
    word = nums_to_word(word[0])
    return word


def rename_cols(metrics: Dict[str, float], rename_cols: Dict[str, str]):
    for new_col, old_col in rename_cols.items():
        metrics[new_col] = metrics.pop(old_col)
