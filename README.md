# 🧠 Neuronales Netz zur Ziffernerkennung

Dieses Projekt implementiert ein künstliches neuronales Netz (KNN), das handschriftlich geschriebene Ziffern erkennt. Es verwendet den MNIST-Datensatz und wurde vollständig mit **NumPy, PIL und scikit-learn** umgesetzt – ohne High-Level-Frameworks wie TensorFlow oder PyTorch.  
Ziel ist es, ein fundiertes Verständnis von Forward- und Backpropagation, Gewichtsanpassung und Visualisierung neuronaler Netzprozesse zu erlangen.

---

## 📦 Features

- Eigenständiges neuronales Netz (Feedforward-Netz)
- Training über mehrere Epochen mit Sigmoid-Aktivierung und MSE-Verlust
- Manuelles Backpropagation-Verfahren
- Visualisierung von:
  - Fehler- und Genauigkeitsverlauf
  - Gewichtsmatrizen
  - Confusion-Matrix (als Tabelle & Heatmap)
  - Heatmap der Fehlklassifikationen
  - Falsch klassifizierten Beispielen
- Vollständige Speicherung aller Ergebnisse als `.png` und `.csv`
- Generierung eines strukturierten Projektberichts mit Trainingsstatistiken

---

## 🧠 Modellarchitektur

| Schicht           | Knotenanzahl     | Beschreibung                    |
|------------------|------------------|---------------------------------|
| Eingabeschicht    | 784              | 28 × 28 Pixel des MNIST-Bildes |
| Versteckte Schicht| 300              | Vollständig verbunden           |
| Ausgabeschicht    | 10               | Ziffern 0–9                     |

- Aktivierungsfunktion: **Sigmoid**
- Verlustfunktion: **Mean Squared Error (MSE)**
- Optimierung: **Stochastischer Gradientenabstieg (SGD)**

---

## 🔧 Voraussetzungen

```bash
pip install numpy pillow torchvision scikit-learn

| Visualisierung       | Beschreibung                                        |
| -------------------- | --------------------------------------------------- |
| 🟩 Fehlerkurve       | MSE-Verlauf pro Epoche                              |
| 🔵 Genauigkeitskurve | Erkennungsgenauigkeit auf Testdaten                 |
| 🟥 Gewichtsbilder    | Visualisierung gelernter Merkmalsdetektoren         |
| 🔶 Confusion-Matrix  | Tabellen- und Heatmapdarstellung der Klassifikation |
| 🧊 Heatmap Fehler    | Durchschnittsbild falsch klassifizierter Stellen    |
| 📷 Beispielbilder    | Trainings- und Testbilder mit eingebettetem Label   |
| 📁 CSV-Dateien       | Heatmap-Daten, Confusion-Matrix als Text            |
| 📄 Projektbericht    | Automatisch erzeugte Projektdokumentation (TXT)     |

---

## 👩‍💻 Autorin & Projektstand

**Autorin:** Heike Fasold  
**Projekt:** Entwicklung eines neuronalen Netzes zur Ziffernerkennung  
**Stand:** 26. Juli 2025  
**Status:** ✅ abgeschlossen (funktionierendes Modell, Visualisierung, Dokumentation)

 Ziel dieses Projekt war es, die mathematischen Grundlagen neuronaler Netze in 
 Python zu verstehen und praktisch umzusetzen – ohne Einsatz vorgefertigter Deep-Learning-Frameworks.
