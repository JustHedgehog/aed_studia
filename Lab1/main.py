import pandas as pd
import seaborn as sns
import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from datetime import datetime

# Zadanie 1

df_netflix = pd.read_csv('netflix_titles.csv')

# 1. Ilość wczytanych wierszy danych
num_rows = len(df_netflix)
print(f"1. Ilość wczytanych wierszy danych: {num_rows}")

# 2. Wymiar wczytanych danych (kolumny x indeksy)
shape = df_netflix.shape
print(f"2. Wymiar wczytanych danych: {shape[1]} kolumny x {shape[0]} indeksy")

# 3. Zlicz ilość wartości pustych w każdej z kolumn
null_counts = df_netflix.isnull().sum()
print("3. Ilość wartości pustych w każdej kolumnie:")
print(null_counts)

# Zadanie 2
df_titanic = sns.load_dataset('titanic')

# 1. Zlicz ile jest wartości pustych w zbiorze NaN
total_nan = df_titanic.isnull().sum().sum()
print(f"1. Całkowita liczba wartości pustych (NaN): {total_nan}")

# 2. Zlicz ilość wartości pustych (null) w każdej kolumnie, a wynik zapisz w postaci sumy skumulowanej
null_per_column = df_titanic.isnull().sum()
print("2. Ilość wartości pustych w każdej kolumnie")
print(null_per_column)
cumsum_null = null_per_column.sum()
print("3. Suma skumulowana wartości pustych w kolumnach:")
print(cumsum_null)

# 3. Usuń te kolumny, jeśli takie istnieją dla których liczba wartości pustych jest większa niż 30% wielkości pobranego zbioru danych
threshold = 0.3 * len(df_titanic)
cols_to_drop = null_per_column[null_per_column > threshold].index
df_titanic_cleaned = df_titanic.drop(columns=cols_to_drop)
print(f"Usunięte kolumny: {list(cols_to_drop)}")
print("Wymiar po usunięciu:", df_titanic_cleaned.shape)

# 4. Zamień dane kategoryczne w kolumnie sex {female,male} na wartości liczbowe {0,1}
df_titanic_cleaned['sex'] = df_titanic_cleaned['sex'].map({'female': 0, 'male': 1})
print("Kolumna 'sex' po zamianie:")
print(df_titanic_cleaned['sex'].head())

# Zadanie 3
username = 'MikiKru'
base_url = f'https://api.github.com/users/{username}'
api_token = ""
headers = {'Authorization': 'token %s' % api_token}

# 1. Odszukaj profil o nazwie użytkownika: MikiKru
response = requests.get(base_url, headers=headers)
if response.status_code == 200:
    repos_number = response.json()['public_repos']
    print(f"Znaleziono {repos_number} repozytoriów dla użytkownika {username}")
else:
    print(f"Błąd: {response.status_code}")
    repos_number = None

# 2. Oblicz statystykę języków programowania
languages = {}
all_repos = []
page = 1
while True:
    response_repos_url = requests.get(response.json()['repos_url'], headers=headers, params={'per_page': "100", 'page': page})

    if response_repos_url.status_code != 200:
        raise Exception("Nie udało pobrać się repo")

    page_data = response_repos_url.json()
    if not page_data:
        break
    all_repos.extend(page_data)
    page += 1

for repo in all_repos:
    lang_url = repo['languages_url']
    lang_response = requests.get(lang_url, headers=headers)
    if lang_response.status_code == 200:
        repo_langs = lang_response.json()
        for lang, bytes_count in repo_langs.items():
            if lang in languages:
                languages[lang] += bytes_count
            else:
                languages[lang] = bytes_count

total_bytes = sum(languages.values())
language_percentages = {lang: (count / total_bytes) * 100 for lang, count in languages.items()}

print("Procentowy udział języków programowania:")
for lang, perc in language_percentages.items():
    print(f"{lang}: {perc:.2f}%")

# 3. Wynik zaprezentuj za pomocą wykresu kołowego
if languages:
    plt.pie(language_percentages.values(), labels=language_percentages.keys(), autopct='%1.1f%%')
    plt.title(f"Udział języków programowania w repozytoriach {username}")
    plt.show()
else:
    print("Brak danych do wykresu.")

#2 Trochę inne obliczenie ilości języków
languages = {}
all_repos = []
page = 1
while True:
    response_repos_url = requests.get(response.json()['repos_url'], headers=headers, params={'per_page': "100", 'page': page})

    if response_repos_url.status_code != 200:
        raise Exception("Nie udało pobrać się repo")

    page_data = response_repos_url.json()
    if not page_data:
        break
    all_repos.extend(page_data)
    page += 1

for repo in all_repos:
    lang_url = repo['languages_url']
    lang_response = requests.get(lang_url, headers=headers)
    if lang_response.status_code == 200:
        repo_langs = lang_response.json()
        for lang, bytes_count in repo_langs.items():
            if lang in languages:
                languages[lang] += 1
            else:
                languages[lang] = 1

total_bytes = sum(languages.values())
language_percentages = {lang: (count / total_bytes) * 100 for lang, count in languages.items()}

print("Procentowy udział języków programowania:")
for lang, perc in language_percentages.items():
    print(f"{lang}: {perc:.2f}%")

# 3. Wynik zaprezentuj za pomocą wykresu kołowego
if languages:
    plt.pie(language_percentages.values(), labels=language_percentages.keys(), autopct='%1.1f%%')
    plt.title(f"Udział języków programowania w repozytoriach {username}")
    plt.show()
else:
    print("Brak danych do wykresu.")

# Zadanie 4

url = "https://www.timeanddate.com/weather/poland/poznan"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

# Pobierz bieżącą temperaturę
current_temp_today = soup.find("div", class_="h2").getText(strip=True)



# Pobierz temperaturę na jutro

soup_today_name = soup.find("table", id="wt-48").find("thead").find_all("tr")[0].find_all("th")[1]

if soup_today_name.has_attr('colspan') and '4' in soup_today_name['colspan']:
    day_temp_tommorow_index = 4
else:
    day_temp_tommorow_index = 0

print(soup_today_name)

day_temp_tommorow_headers = soup.find("table", id="wt-48").find("thead").find_all("tr")[1].find_all("th")
for i, tag in enumerate(day_temp_tommorow_headers):
    if tag.name == 'th' and tag.has_attr('class') and 'sep-l' in tag['class']:
        span = tag.find('span')
        if span and span.text.strip() == 'Night':
            day_temp_tommorow_index += i
            break

print(day_temp_tommorow_index)
day_temp_tommorow = soup.find("table", id="wt-48").find("tbody").find_all("tr")[1].find_all("td")[day_temp_tommorow_index].getText(strip=True)
night_temp_tommorow = soup.find("table", id="wt-48").find("tbody").find_all("tr")[1].find_all("td")[day_temp_tommorow_index-1].getText(strip=True)

# Bieżąca data
now = datetime.now()
date_str = now.strftime('%d.%m.%Y %H:%M')

print(f"Data: {date_str}")
print(f"Temperatura: {current_temp_today}")
print(f"Jutro: {day_temp_tommorow} / {night_temp_tommorow}")