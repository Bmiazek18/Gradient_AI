# 🩺 AI do Przewidywania Cukrzycy (Drzewo Decyzyjne)

System sztucznej inteligencji służący do wczesnego wykrywania ryzyka cukrzycy na podstawie danych klinicznych. Projekt koncentruje się na minimalizacji błędów krytycznych w medycynie (przeoczenie choroby) poprzez zastosowanie zoptymalizowanych progów decyzyjnych oraz technik uczenia maszynowego priorytetyzujących bezpieczeństwo pacjenta.

Głównym celem projektu było stworzenie modelu **Explainable AI (XAI)** – czyli "wyjaśnialnej sztucznej inteligencji", gdzie każda diagnoza może być poparta konkretną ścieżką logiczną zrozumiałą dla lekarza.

---

## 📊 Wyniki i Raport Klasyfikacji

Obecny model z obniżonym progiem decyzyjnym (0.4) osiąga **72% ogólnej dokładności**, ale co najważniejsze w kontekście medycznym – wykazuje bardzo wysoką czułość dla pacjentów chorych.

Oto szczegółowy raport wygenerowany przez model na zbiorze testowym:

```text
              precision    recall  f1-score   support

           0       0.86      0.69      0.77       100
           1       0.57      0.79      0.66        52

    accuracy                           0.72       152
   macro avg       0.72      0.74      0.71       152
weighted avg       0.76      0.72      0.73       152
```

### 🧠 Logika Podejmowania Decyzji
Poniższe drzewo pokazuje, jakie warunki muszą zostać spełnione, aby model sklasyfikował pacjenta jako zagrożonego cukrzycą. Model został ograniczony do głębokości 4, aby uniknąć przeuczenia i zachować czytelność dla człowieka.

![Logika Drzewa Decyzyjnego](tree_logic.png)

### 📊 Skuteczność (Macierz Pomyłek)
Macierz pokazuje bilans trafnych diagnoz oraz błędów w przejrzystej formie wizualnej, ilustrując kompromis między precyzją a czułością, który osiągnęliśmy dzięki zmianie progu decyzyjnego.

![Macierz Pomyłek](confusion_matrix.png)

### 🔑 Najważniejsze Parametry
Wykres przedstawia, które dane medyczne miały największy wpływ na werdykt modelu. Pozwala to na optymalizację badań – wiemy, na które parametry (np. Glukoza, BMI) lekarz powinien zwrócić szczególną uwagę.

![Istotność Cech](features_importance.png)

---

## 🛠️ Architektura Modelu i Przetwarzanie Danych

Aby osiągnąć stabilne wyniki, zastosowano następujące techniki inżynierii danych:

* **Klasyfikator:** `DecisionTreeClassifier` (Scikit-Learn).
* **Maksymalna Głębokość (max_depth):** 4.
* **Próg decyzyjny (Custom Threshold):** 0.4 (Priorytet bezpieczeństwa pacjenta).
* **Czyszczenie Danych (Data Imputation):** Zastąpienie zerowych, fizjologicznie niemożliwych wartości (np. ciśnienie równe 0) medianą, co zapobiegło zafałszowaniu wyników.
* **Usuwanie Outlierów:** Usunięcie ekstremalnych wartości odstających dla BMI i Glukozy za pomocą metody IQR (Interquartile Range).
* **Cost-Sensitive Learning:** Użycie wag `class_weight='balanced'` do wyrównania szans uczenia się na mniejszościowej grupie osób chorych (kara za pominięcie chorego jest większa niż za pomyłkę przy zdrowym).
* **Stratified Sampling:** Zachowanie równych proporcji chorych i zdrowych podczas podziału na zbiór treningowy i testowy.

---

## 🧪 Parametry Wejściowe
Model ocenia następujące cechy (Pima Indians Diabetes Dataset):
1. **Ciąże** – liczba przebytych ciąż.
2. **Glukoza** – stężenie glukozy w osoczu.
3. **Ciśnienie krwi** – rozkurczowe ciśnienie krwi (mm Hg).
4. **Grubość skóry** – grubość fałdu skórnego nad tricepsem (mm).
5. **Insulina** – 2-godzinne stężenie insuliny w surowicy.
6. **BMI** – wskaźnik masy ciała.
7. **Rodowód cukrzycy** – wskaźnik obciążeń genetycznych.
8. **Wiek** – wiek pacjenta (lata).

---
*Zastrzeżenie prawne: Prezentowany model ma charakter badawczy i edukacyjny. Wszelkie diagnozy i decyzje medyczne muszą być weryfikowane przez wykwalifikowany personel.*