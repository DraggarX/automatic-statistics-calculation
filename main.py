import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import Levenshtein
from multiprocessing import Pool, cpu_count
from functools import partial

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
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Специализация' not in df.columns:
        raise ValueError("Файл должен содержать столбец 'Специализация'.")
    return df['Специализация'].dropna().tolist()

# Загрузка специализаций пользователей из Excel-файла
def load_user_specializations(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Специальность' not in df.columns or 'Город' not in df.columns:
        raise ValueError("Файл должен содержать столбцы 'Специальность' и 'Город'.")
    return df[['Специальность', 'Город']].dropna()

# Загрузка соответствия городов и регионов
def load_city_to_region(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден.")
    df = pd.read_excel(file_path)
    if 'Город' not in df.columns or 'Регион' not in df.columns:
        raise ValueError("Файл должен содержать столбцы 'Город' и 'Регион'.")
    return df.set_index('Город')['Регион'].to_dict()

# Функция для обработки одной специализации пользователя
def process_specialization(line, specializations, threshold, vectorizer, vectors):
    line = line.strip()
    if not line:
        return []

    # Разделяем строку на отдельные специальности
    specialties = [spec.strip() for spec in line.split(',')]

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

        if max_similarity > threshold:
            matched_specializations.append(best_match)
        else:
            # Используем семантическую схожесть
            semantic_sim, semantic_match = semantic_similarity(spec, specializations, vectorizer, vectors)
            if semantic_sim > threshold:
                matched_specializations.append(semantic_match)

    return matched_specializations

# Подсчет упоминаний специализаций с использованием параллельных вычислений
def count_specializations(user_specializations, specializations, city_to_region, threshold=0.6):
    counts = Counter()
    region_counts = Counter()
    vectorizer = TfidfVectorizer().fit(specializations)
    vectors = vectorizer.transform(specializations).toarray()

    worker_partial = partial(process_specialization, specializations=specializations, threshold=threshold, vectorizer=vectorizer, vectors=vectors)

    with Pool(cpu_count()) as pool:
        matches = list(tqdm(pool.imap(worker_partial, user_specializations['Специальность']), total=len(user_specializations), desc="Processing", unit="user_specialization"))

    for idx, user_matches in enumerate(matches):
        if user_matches:
            city = user_specializations.iloc[idx]['Город'].strip()
            region = city_to_region.get(city, 'Другие')
            for match in user_matches:
                counts[match] += 1
                region_counts[(match, region)] += 1

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
        print(e)