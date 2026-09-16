# Notatki autora

## Rzeczywisty cel badawczy

Projekt bada selekcję częstotliwości cieniowania po rasteryzacji w deferred rendererze. Decyzja jest podejmowana wyłącznie z danych istniejącego G-buffera i lokalnej czułości kontrolowanego modelu GGX. Nie jest to praca o nowości VRS ani o samym multirate shading.

## Hipoteza robocza

Przy podobnym koszcie selektor wykorzystujący estymowaną zmianę radiancji powinien mieć mniejszy odsetek pikseli przekraczających zadany budżet błędu niż selektor oparty jedynie na progach zmian normalnej, pozycji, albedo i roughness.

## Czego nie wolno deklarować bez pomiarów

- formalnej górnej granicy błędu finalnej radiancji,
- równoważności z hardware VRS,
- perceptual optimality,
- ogólnej przewagi dla wszystkich rendererów i BRDF,
- sprzętowo niezależnego przyspieszenia.

## Najważniejsze eksperymenty do wykonania

1. Zweryfikować literaturę i zaimplementować najbliższy porównywalny baseline.
2. Wykonać matched-cost i matched-quality comparison z heurystyką ekranową.
3. Dodać stress test z wysokoczęstotliwościową normal mapą i cienką geometrią.
4. Wykonać ablacją składników predyktora: normal, position, roughness, albedo i light-count scaling.
5. Dodać sekwencje ruchu kamery i pomiar temporal rate instability.
6. Powtórzyć timing na drugim GPU przed szeroką tezą wydajnościową.

## Oczekiwany typ wyniku

Najważniejsza zależność to kalibracja: czy zmiana żądanego budżetu powoduje przewidywalną zmianę rzeczywistego błędu oraz udziału 1x1/2x2/4x4. Przyspieszenie ma być oceniane dopiero razem z tą zależnością.
