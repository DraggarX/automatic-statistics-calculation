import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import Levenshtein
from multiprocessing import Pool, cpu_count
from functools import partial
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("logfile.log"),
        logging.StreamHandler()
    ]
)

# Функция для вычисления схожести строк с использованием Levenshtein
def similar(a, b):
    return Levenshtein.ratio(a.lower(), b.lower())

# Функция для вычисления семантической схожести с использованием TF-IDF и косинусной меры
def semantic_similarity(query, choices, vectorizer, vectors):
    query_vector = vectorizer.transform([query]).toarray()
    cosine_similarities = cosine_similarity(query_vector, vectors)[0]
    return max((similarity, choice) for similarity, choice in zip(cosine_similarities, choices))

# Загрузка списка специализаций из Excel-файла
def load_specializations(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Специализация' not in df.columns:
        logging.error("Файл должен содержать столбец 'Специализация'.")
        raise ValueError("Файл должен содержать столбец 'Специализация'.")
    logging.info(f"Загружено {len(df['Специализация'].dropna())} специализаций.")
    return df['Специализация'].dropna().tolist()

# Загрузка специализаций пользователей из Excel-файла
def load_user_specializations(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Специальность' not in df.columns or 'Город' not in df.columns:
        logging.error("Файл должен содержать столбцы 'Специальность' и 'Город'.")
        raise ValueError("Файл должен содержать столбцы 'Специальность' и 'Город'.")
    logging.info(f"Загружено {len(df)} записей пользователей.")
    return df[['Специальность', 'Город']].dropna()

# Загрузка соответствия городов и регионов
def load_city_to_region(file_path):
    if not os.path.exists(file_path):
        logging.error(f"Файл {file_path} не найден.")
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Город' not in df.columns or 'Регион' not in df.columns:
        logging.error("Файл должен содержать столбцы 'Город' и 'Регион'.")
        raise ValueError("Файл должен содержать столбцы 'Город' и 'Регион'.")
    logging.info(f"Загружено {len(df)} соответствий городов и регионов.")
    return df.set_index('Город')['Регион'].to_dict()

# Функция для обработки одной специализации пользователя
def process_specialization(line, specializations, threshold, vectorizer, vectors):
    line = line.strip()
    if not line:
        return []

    # Разделяем строку на отдельные специальности
    specialties = [spec.strip() for spec in line.split(',')]
    logging.debug(f"Обработка строки: {line}, специализации: {specialties}")

    matched_specializations = []

    for spec in specialties:
        max_similarity = 0
        best_match = None

        # Используем схожесть строк
        for s in specializations:
            similarity = similar(spec, s)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = s
        logging.debug(f"Лучшая схожесть для '{spec}': {max_similarity} с '{best_match}'")

        if max_similarity > threshold:
            matched_specializations.append(best_match)
        else:
            # Используем семантическую схожесть
            semantic_sim, semantic_match = semantic_similarity(spec, specializations, vectorizer, vectors)
            logging.debug(f"Семантическая схожесть для '{spec}': {semantic_sim} с '{semantic_match}'")
            if semantic_sim > threshold:
                matched_specializations.append(semantic_match)
    logging.debug(f"Найденные специализации для строки '{line}': {matched_specializations}")
    return matched_specializations

# Подсчет упоминаний специализаций с использованием параллельных вычислений
def count_specializations(user_specializations, specializations, city_to_region, threshold=0.6):
    counts = Counter()
    region_counts = Counter()
    vectorizer = TfidfVectorizer().fit(specializations)
    vectors = vectorizer.transform(specializations).toarray()
    logging.info("Начата обработка пользовательских специализаций.")

    worker_partial = partial(process_specialization, specializations=specializations, threshold=threshold, vectorizer=vectorizer, vectors=vectors)

    with Pool(cpu_count()) as pool:
        matches = list(tqdm(pool.imap(worker_partial, user_specializations['Специальность']), total=len(user_specializations), desc="Processing", unit="user_specialization"))
        logging.info("Обработка пользовательских специализаций завершена.")

    for idx, user_matches in enumerate(matches):
        if user_matches:
            row = user_specializations.iloc[idx]
            city = row['Город'].strip()
            region = city_to_region.get(city, 'Другие')
            for match in user_matches:
                counts[match] += 1
                region_counts[(match, region)] += 1
                logging.debug(f"Увеличение счетчика для '{match}' в регионе '{region}'.")

    return counts, region_counts

# Основная программа
if __name__ == "__main__":
    # Укажите пути к Excel-файлам
    specializations_file_path = "specializations.xlsx"
    user_specializations_file_path = "user_specializations.xlsx"
    city_to_region_file_path = "city_to_region.xlsx"

    try:
        # Загрузка данных
        specializations = load_specializations(specializations_file_path)
        user_specializations = load_user_specializations(user_specializations_file_path)
        city_to_region = load_city_to_region(city_to_region_file_path)
        
        # Подсчет упоминаний
        counts, region_counts = count_specializations(user_specializations, specializations, city_to_region)

        # Вывод результатов
        for spec in counts:
            print(f"{spec}: {counts[spec]}")
            # Получаем все регионы для текущей специализации
            regions = [reg for (s, reg) in region_counts if s == spec]
            region_counts_spec = {reg: region_counts[(spec, reg)] for reg in regions}
            for region in sorted(region_counts_spec, key=region_counts_spec.get, reverse=True):
                print(f"    {region} - {region_counts_spec[region]}")
            print()

    except (FileNotFoundError, ValueError) as e:
        logging.error(e)
