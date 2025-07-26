# Import der benötigten Module
import numpy as np
import os
import time
from sklearn.model_selection import train_test_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw, ImageFont
# Festlegung der Konstanten
# Anzahl der Durchläufe
EPOCHS = 5
# Lernrate
LEARNING_RATE = 0.5
# Definition der einzelnen Knoten
# Eingabeknoten
INPUT_NODES = 784
# versteckte Knoten
HIDDEN_NODES = 300
# Ausgabeknoten
OUTPUT_NODES = 10
# Berechnung der Anfangsgewichte
WEIGHT_INPUT_HIDDEN = np.random.uniform(-0.5, 0.5, (INPUT_NODES, HIDDEN_NODES))
WEIGHT_HIDDEN_OUTPUT = np.random.uniform(-0.5, 0.5,(HIDDEN_NODES, OUTPUT_NODES))
# Bias-Vektoren für die Eingabe- und Ausgabeschichten
BIAS_HIDDEN = np.zeros((HIDDEN_NODES,))
BIAS_OUTPUT = np.zeros((OUTPUT_NODES,))
# Konstanten für die Visualisierung
NUMBER = 10
# Batchgröße für das Training
BATCH_SIZE = 32
# Festlegung des Zufallsseed für Reproduzierbarkeit
np.random.seed(42)
# Erzeute Ausgabeordner für alle Ergebnisse
OUTPUT_DIR = "Projektarbeit_Output_Bilder"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Definition der einzelnen Klassen
class MNISTDownLoad:
    ''' Klasse zum Herunterladen des vollständigen MNIST-Datensatzes über torch-
        vision. 
        
        Sie kombiniert Trainings- und Testdaten, ermöglicht eine flexible 
        Aufteilung in Trainings- und Testdaten, und speichert die Daten lokal 
        im .npz-Format für die Offline-Nutzung.
    '''
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42,
                    data_way: str = "mnist_data"):
        '''
        Initialisiert die Klasse zum Laden und Aufteilen des MNIST-Datensatzes.
        
        Parameter:
        test_size: float
            Anteil der Testdaten (zwischen 0.0 und 1.0).
        random_state:int
            Startwert für den Zufallsgenerator (Reproduzierbarkeit).
        data_way: str
            Verzeichnis, in dem die Daten gespeichert werden..
        ''' 
        self.test_size = test_size
        self.random_state = random_state
        self.data_way = data_way
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def load_and_split_mnist_data(self):
        ''' 
        Lädt den MNIST-Datensatz herunter, kombiniert Trainings- und Testdaten 
        und teilt sie gemäß der angegebenen Testgröße in Trainings- und Testda-
        ten auf.
        '''
        print("Lade den vollständigen MNIST-Datensatz ...... ")
        train_set = datasets.MNIST(root=self.data_way, train=True, 
                                    download=True, transform=ToTensor())
        test_set = datasets.MNIST(root=self.data_way, train=False, 
                                    download=True, transform=ToTensor())
        # Umwandlung der Tensor-Daten in NumPy-Arrays
        X_train_raw = train_set.data.numpy()
        y_train_raw = train_set.targets.numpy()
        X_test_raw = test_set.data.numpy()
        y_test_raw = test_set.targets.numpy()
        # Zusammenführen beider Datensätze
        X_all = np.concatenate([X_train_raw, X_test_raw], axis=0)
        y_all = np.concatenate([y_train_raw, y_test_raw], axis=0)
        # Aufteilen in Trainings- und Testdaten nach test_size
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, 
            random_state=self.random_state, stratify=y_all,)
        print("Aufteilung in Trainings- und Testdaten abgeschlossen")
    
    def save_mnist_data_new(self, way: str = "mnist_daten_split.npz"):
        ''' 
        Speichert die gesplitteten Trainings- und Testdaten als komprimierte 
        .npz-Datei.

        Parameter:
        way : str
            Pfad zur Ausgabedatei (.npz).
        '''
        try:
            if self.X_train is None or self.X_test is None:
                raise ValueError('''Daten wurden nicht geladen. Bitte erneut 
                            Funktion: load_and_split_mnist_data ausführen''')
            way = os.path.abspath(way)
            np.savez_compressed(way, X_train = self.X_train, 
                                y_train = self.y_train, X_test = self.X_test,
                                y_test = self.y_test)
            print(f"Die Daten wurden gespeichert unter {way}")
        except Exception as e:
            print(f"Fehler beim Speichern der Daten: {e}")
    
    def load_mnist_data_npz(self, way: str):
        '''
        Lädt Trainings- und Testdaten aus einer vorhandenen .npz-Datei.

        Parameter:
        way : str
            Pfad zur gespeicherten .npz-Datei.
        '''
        try:
            way = os.path.abspath(way)
            data = np.load(way)
            self.X_train = data["X_train"]
            self.y_train = data["y_train"]
            self.X_test = data["X_test"]
            self.y_test = data["y_test"]
            print(f"Daten erfolgreich geladen aus: {way}")
        except Exception as e:
            print(f"Datei wurde nicht gefunden: {e}")

class NeuralNetTraining:
    '''
    Klasse zur Definition und zum Training eines einfachen künstlichen neurona-
    len Netzwerks mit einer versteckten Schicht. Sie enthält Methoden zur Vor-
    wärts- und Rückwärtspropagation, Fehlerberechnung und Trainingsauswertung.
    '''
    def __init__(self):
        '''
        Konstruktor der Klasse zur Initialisierung eines neuronalen Netzes.
        Alle Parameter stammen aus den oben definierten globalen Konstanten.
        '''
        self.epochs = EPOCHS
        self.learning_rate = LEARNING_RATE
        self.input_nodes = INPUT_NODES
        self.hidden_nodes = HIDDEN_NODES
        self.output_nodes = OUTPUT_NODES
        # Kopie der globalen Konstanten zur Veränderung und Nutzung
        self.weight_input_hidden = WEIGHT_INPUT_HIDDEN.copy()
        self.weight_hidden_output = WEIGHT_HIDDEN_OUTPUT.copy()
        # Bias-Vektoren für die Eingabe- und Ausgabeschichten
        # Bias für die versteckte Schicht, als Kopie der globalen Konstante
        self.bias_hidden = BIAS_HIDDEN.copy()   
        # Bias für die Ausgabeschicht, als Kopie der globalen Konstante
        self.bias_output = BIAS_OUTPUT.copy()
    
    def sig(self, x: float | np.ndarray) -> float | np.ndarray:
        '''
        Berechnet die Sigmoid-Aktivierungsfunktion.

        Parameter:
        x : float oder np.ndarray
            Eingabewert(e)
        Rückgabe:
        float oder np.ndarray
            Aktivierungswert(e)     
        '''
        return 1 / (1 + np.exp(-x))
    
    def sig_der(self, y) -> float | np.ndarray:
        '''
        Berechnet die Ableitung der Sigmoid-Funktion für die Rückpropagation.

        Parameter:
        y : float oder np.ndarray
            Ausgabewert(e) der Sigmoid-Funktion
        Rückgabe:
        float oder np.ndarray
            Ableitungswert(e)
        '''
        return y * (1 - y)
    
    def training_batch(self, X_batch, y_batch):
        '''
        Führt einen Trainingsdurchgang (Vorwärts- und Rückwärtspropagation) für 
        ein Batch aus.

        Parameter:
            X_batch : np.ndarray
            Eingabedaten (Batch)
            y_batch : np.ndarray
            Zielausgaben (One-hot-codiert)
        Rückgabe:
        float
            Mittlerer quadratischer Fehler des Batches
        '''
        # Vorwärtspropagation
        hidden_input = np.dot(X_batch, self.weight_input_hidden) + self.bias_hidden
        hidden_output = self.sig(hidden_input)
        output_input = np.dot(hidden_output, self.weight_hidden_output) + self.bias_output
        output_data = self.sig(output_input)
        # Fehlerberechnung
        error_output = y_batch - output_data
        error_hidden = np.dot(error_output * self.sig_der(output_data),
                    self.weight_hidden_output.T) * self.sig_der(hidden_output)
        # Aktualisierung der Gewichte (Rückwärtspropagation)
        self.weight_hidden_output += self.learning_rate * np.dot(hidden_output.T,
                    error_output * self.sig_der(output_data))/ X_batch.shape[0]
        self.weight_input_hidden += self.learning_rate * np.dot(X_batch.T, 
                                            error_hidden) / X_batch.shape[0] 
        # Bias-Vektoren aktualisieren
        self.bias_output += self.learning_rate * np.mean(error_output * self.sig_der(output_data), axis=0)  
        self.bias_hidden += self.learning_rate * np.mean(error_hidden, axis=0)  
        return np.mean(np.square(error_output))                   
    
    def predict(self, X,):      
        '''
        Führt eine Vorwärtspropagation mit beliebiger Eingabe durch.

        Parameter:
        X : np.ndarray
            Eingabedaten (einzelnes Bild oder Array von Bildern)
        Rückgabe:
        np.ndarray
            Netzwerkausgaben für jede Eingabe
        '''
        # Falls ein einzelnes Bild als 1D-Vektor kommt, auf (1, 784) bringen
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # Bias beim Vorwärtsdurchlauf hinzufügen
        H_predict = self.sig(np.dot(X, self.weight_input_hidden) + self.bias_hidden)
        O_predict = self.sig(np.dot(H_predict, self.weight_hidden_output) + self.bias_output)
        return O_predict
    
    def training_epochs(self, X_train, y_train, X_val=None, y_val=None):
        '''
        Trainiert das neuronale Netz über mehrere Epochen und gibt Fehler- und
        Genauigkeitsverlauf zurück.

        Parameter:
        X_train : np.ndarray
            Trainingsbilder
        y_train : np.ndarray
            Zielwerte für das Training
        X_val : np.ndarray, optional
            Validierungsbilder
        y_val : np.ndarray, optional
            Zielwerte für die Validierung
        Rückgabe:
        Tuple[List[float], List[float or None]]
            Fehler pro Epoche, Genauigkeit pro Epoche (falls Val-Daten vorhanden)
        '''
        n_samples = X_train.shape[0]
        y_train_oh = np.eye(self.output_nodes)[y_train]
        error_list = []
        acc_list = []
        for ep in range(self.epochs):
            idx = np.random.permutation(n_samples)
            X_shuf = X_train[idx]
            y_shuf = y_train_oh[idx]
            batch_losses = []
            for start in range(0, n_samples, BATCH_SIZE):
                end = start + BATCH_SIZE
                X_batch = X_shuf[start:end].reshape(-1, self.input_nodes) / 255.0
                Y_batch = y_shuf[start:end].reshape(-1, self.output_nodes)
                batch_loss = self.training_batch(X_batch, Y_batch)
                batch_losses.append(batch_loss)
            ep_loss = np.mean(batch_losses)
            error_list.append(ep_loss)
            # Optional: Accuracy auf Val-Set pro Epoche
            if X_val is not None and y_val is not None:
                y_pred = np.argmax(self.predict(X_val.reshape(-1, 
                                    self.input_nodes)/255.0), axis=1)
                acc = np.mean(y_pred == y_val)
                acc_list.append(acc)
            else:
                acc_list.append(None)
            if (ep + 1) % 10 == 0 or ep == self.epochs - 1:
                print(f"Epoche {ep+1}/{self.epochs} abgeschlossen, "
                        f"Verlust: {ep_loss:.4f}")
        return error_list, acc_list
    
    def accuracy(self, X, y):
        '''
        Berechnet die Klassifikationsgenauigkeit für gegebene Eingabedaten.

        Parameter:
        X : np.ndarray
            Eingabedaten
        y : np.ndarray
            Wahre Zielwerte
        Rückgabe:
        float
            Genauigkeit (Anteil korrekt klassifizierter Beispiele)
        '''
        preds = self.predict(X.reshape(-1, self.input_nodes)/255.0)
        y_pred = np.argmax(preds, axis=1)
        return np.mean(y_pred == y)                       
        
class Visualizer:
    '''
    Klasse zur Visualisierung der Trainings- und Testergebnisse eines neuronalen
    Netzes. Enthält Methoden zum Speichern von Beispieldaten, Fehlerkurven,
    Genauigkeitskurven, Gewichtsmatrizen und falsch klassifizierten Bildern.
    '''
    
    def __init__(self, X_train, y_train, X_test, y_test):
        '''
        Initialisiert die Visualisierungsinstanz mit Trainings- und Testdaten.

        Parameter:
        X_train, y_train, X_test, y_test : np.ndarray
            Datensätze zur Verwendung für Visualisierungen.
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test
    
    def save_sample_picture_train(self, data_way="mnist_bsp_train"):
        '''
        Speichert Beispielbilder mit Labels aus dem Trainings-/Testset als PNG-
        Dateien.

        Parameter:
        data_way : str
            Verzeichnispräfix für die Ausgabedateien.
        '''
        # Ordner sicherstellen
        os.makedirs(data_way, exist_ok=True)

        for i in range(NUMBER):
            # Bild aus Trainigsdaten als PIL-Bild erzeugen
            picture = Image.fromarray(self.X_train[i].astype(np.uint8))
            # Bild vergrößern
            picture_train_bigger = picture.resize((115, 115), 
                                                Image.Resampling.NEAREST)
            # Bild speichern
            # Pfad für das Bild
            filename = f"trainbild_{i}_label_{self.y_train[i]}.png"
            way = os.path.join(data_way, filename)
            picture_train_bigger.save(way)
        print(f"{NUMBER} Beispielbilder für das Training gespeichert")
    
    def save_sample_picture_test(self, data_way="mnist_bsp_test"):  
        ''''
        Speichert Beispielbilder mit Labels aus dem Trainings-/Testset als PNG-
        Dateien.

        Parameter:
        data_way : str
            Verzeichnispräfix für die Ausgabedateien.
        '''
        # Ordner sicherstellen
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_test[i].astype(np.uint8))
            picture_test_bigger = picture.resize((115, 115), 
                                                Image.Resampling.NEAREST)
            # Pfad zusammensetzen
            filename = f"testbild_{i}_label_{self.y_test[i]}.png"
            way = os.path.join(data_way, filename)

            picture_test_bigger.save(way)
        print(f"{NUMBER} Beispielbilder für Testbilder gespeichert")
    
    def save_error_curve(self, error_list, data_way="fehlerkurve.png"):
        '''
        Zeichnet die Fehlerkurve (Loss) über die Trainingsverläufe und speichert 
        sie als Bild.

        Parameter:
        error_list : List[float]
            Fehlerwerte pro Epoche.
        data_way : str
            Speicherpfad für die Ausgabegrafik.
        '''
        wide, hight, = 480, 320
        picture_error_curve = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture_error_curve)
        # Achsen, x, y
        draw.line((40, hight-40, wide-20, hight-40), fill = "black")
        draw.line((40, hight-40, 40, 20), fill = "black")
        # Werte festlegen
        if len(error_list) > 1:
            max_error = max(error_list)
            min_error = min(error_list)
            n = len(error_list)
            scale_x = (wide-60) / (n-1)
            scale_y = (hight-60) / (max_error - min_error + 1e-8)
            # Fehlerkurve einzeichnen
            for i in range(n-1):
                x1 = 40 + i * scale_x
                y1 = hight - 40 - (error_list[i] - min_error) * scale_y
                x2 = 40 + (i + 1) * scale_x
                y2 = hight - 40 - (error_list[i + 1] - min_error) * scale_y
                draw.line((x1, y1, x2, y2), fill = "red", width = 2)
        picture_error_curve.save(data_way)
        print(f"Fehlerkurve wurde als {data_way} gespeichert")
    
    def save_accuracy(self, acc_list, data_way="genauigkeitskurve.png"):
        '''
        Zeichnet die Genauigkeitskurve über die Trainings-/Validierungsepochen.

        Parameter:
        acc_list : List[float or None]
            Genauigkeitswerte pro Epoche.
        data_way : str
            Speicherpfad für die Ausgabegrafik.
        '''
        wide, hight = 480, 320
        picture_accuracy = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture_accuracy)
        draw.line((40, hight-40, wide-20, hight-40), fill = "black")
        draw.line((40, hight-40, 40, 20), fill = "black")
        if len(acc_list) > 1:
            valid = [v for v in acc_list if v is not None]
            if valid:
                max_acc = max(acc_list)
                min_acc = min(acc_list)
            else:
                print("Keine gültigen Genauigkeitswerte vorhanden.")
                return
            n = len(acc_list)
            scale_x = (wide-60) / (n-1)
            scale_y = (hight-60) / (max_acc - min_acc + 1e-8)
            for i in range(n-1):
                x1 = 40 + i * scale_x
                y1 = hight - 40 - (acc_list[i] - min_acc) * scale_y
                x2 = 40 + (i + 1) * scale_x
                y2 = hight - 40 - (acc_list[i + 1] - min_acc) * scale_y
                draw.line((x1, y1, x2, y2), fill = "red", width = 2)
        picture_accuracy.save(data_way)
        print(f"Genauigkeitskurve wurde als {data_way} gespeichert")
    
    def save_train_vs_test_curve(self, error_list, acc_list, 
                                    data_way="train_vs_test.png"):
        '''
        Zeichnet Fehler- und Genauigkeitskurve für Training und Test in einem 
        Bild. Es werden die Verläufe übereinandergelegt (y-Achse links = Verlust, 
        rechts = Genauigkeit).
        
        Parameter:
        error_list : List[float]
            Fehlerwerte pro Epoche für das Training.
        acc_list : List[float or None]
            Genauigkeitswerte pro Epoche (Training vs. Test).
        data_way : str
            Speicherpfad für die Ausgabegrafik.
        '''
        wide, hight = 640, 400
        picture = Image.new("RGB", (wide, hight), "white")
        draw = ImageDraw.Draw(picture)
        # Achsen
        draw.line((60, hight-60, wide-30, hight-60), fill="black")
        draw.line((60, hight-60, 60, 30), fill="black")
        # Fehlerkurve links (rot), Genauigkeitskurve rechts (blau)
        if len(error_list) > 1:
            n = len(error_list)
            max_error = max(error_list)
            min_error = min(error_list)
            max_acc = max([v for v in acc_list if v is not None], default=1)
            min_acc = min([v for v in acc_list if v is not None], default=0)
            scale_x = (wide-90) / (n-1)
            scale_y_err = (hight-90) / (max_error - min_error + 1e-8)
            scale_y_acc = (hight-90) / (max_acc - min_acc + 1e-8)
            # Fehlerkurve (rot, links)
            for i in range(n-1):
                x1 = 60 + i * scale_x
                y1 = hight-60 - (error_list[i] - min_error) * scale_y_err
                x2 = 60 + (i+1) * scale_x
                y2 = hight-60 - (error_list[i+1] - min_error) * scale_y_err
                draw.line((x1, y1, x2, y2), fill="red", width=2)
            # Genauigkeit (blau, rechts)
            for i in range(n-1):
                if acc_list[i] is not None and acc_list[i+1] is not None:
                    x1 = 60 + i * scale_x
                    y1 = hight-60 - (acc_list[i] - min_acc) * scale_y_acc
                    x2 = 60 + (i+1) * scale_x
                    y2 = hight-60 - (acc_list[i+1] - min_acc) * scale_y_acc
                    draw.line((x1, y1, x2, y2), fill="blue", width=2)
            # Legende
            draw.text((wide-200, 40), "Fehler (rot, links)", fill="red")
            draw.text((wide-200, 60), "Genauigkeit (blau, rechts)", fill="blue")
        picture.save(data_way)
        print(f"Trainings-/Testkurve gespeichert als {data_way}")

    def save_weights(self, weight_input_hidden, NUMBER, 
                            data_way="gewichte"):
        '''
        Visualisiert die ersten NUMBER Gewichte der Eingabeschicht als 
        Bilder.

        Parameter:
        weight_input_hidden : np.ndarray
            Gewichtsmatrix (784 x HIDDEN_NODES).
        NUMBER : int
            Anzahl der Gewichtsbilder, die gespeichert werden sollen.
        data_way : str
            Zielpfad für das resultierende Bild.
        '''
        # Ordner sicherstellen
        os.makedirs(data_way, exist_ok=True)
        w_picture = []
        for i in range(NUMBER):
            picture = weight_input_hidden[:, i].reshape(28, 28)
            # Bild einheitlich machen
            picture = 255 * (picture - picture.min()) / (np.ptp(picture) + 1e-8)
            picture_unit = Image.fromarray(picture.astype(np.uint8)).resize((56, 56),
                                            Image.Resampling.NEAREST)
            w_picture.append(picture_unit)
        # Als Bildserie
        picture_all = Image.new("L", (56 * NUMBER, 56))
        for i, picture_unit in enumerate(w_picture):
            picture_all.paste(picture_unit, (i * 56, 0))
        # Speicherpfad festlegen
        save_way = os.path.join(data_way, "gewichte.png")
        picture_all.convert("RGB").save(save_way)
        print(f"{NUMBER} Gewichte gespeichert als {save_way}")
    
    def save_misclassified(self, net, X, y, max_picture,
                            data_way="falsch_klassifiziert"):
        '''
        Speichert falsch klassifizierte Testbilder als PNG-Dateien.

        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder
        y : np.ndarray
            Wahre Labels
        max_picture : int
            Maximale Anzahl falsch klassifizierter Bilder, die gespeichert 
            werden sollen.
        data_way : str
            Speicherpfad (Dateinamenpräfix)
        '''
        wrong_classified = 0
        # Ordner sicherstellen
        os.makedirs(data_way, exist_ok=True)
        for i in range(len(X)):
            petition = X[i].flatten() / 255.0
            output_data = net.predict(petition)
            available = np.argmax(output_data)
            if available != y[i]:
                arr = X[i].astype(np.uint8)
                picture = Image.fromarray(arr)
                picture = picture.resize((115, 115), Image.Resampling.NEAREST)
                filename = f"{wrong_classified}_richtig_{y[i]}_vorhergesagt_{available}.png"
                way = os.path.join(data_way, filename)
                try:
                    picture.save(way)
                except Exception as e:
                    print(f"Fehler beim Speichern des Bildes: {e}")
                wrong_classified += 1
                if wrong_classified >= max_picture:
                    break
        if wrong_classified == 0:
            print("Keine falsch klassifizierten Bilder gefunden.")
        else:
            print(f'''{wrong_classified} falsch klassifizierte Bilder 
                        gespeichert.''')

    def save_misclassified_table(self, net, X, y, max_picture, 
                            data_way="falsch_klassifiziert_tabelle"):
        '''
        Erstellt eine PNG-Tabelle mit echten und vorhergesagten Labels der falsch 
        klassifizierten Testbilder.

        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder
        y : np.ndarray
            Wahre Labels
        max_picture : int
            Maximale Anzahl falsch klassifizierter Bilder, die aufgeführt werden 
            sollen.
        data_way : str
            Dateiname für die Ausgabegrafik.
        '''
        # Sammle falsch klassifizierte Beispiele
        wrongs = []
        for i in range(len(X)):
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                wrongs.append((i, int(y[i]), int(pred)))
                # Bild speichern
                filename = f"{i}_richtig_{int(y[i])}_vorhergesagt_{int(pred)}.png"
                way = os.path.join(data_way, filename)
                try:
                    os.makedirs(data_way, exist_ok=True)
                    arr_unit8 = (picture.reshape(28, 28) * 255).astype(np.uint8)
                    img_to_save = Image.fromarray(arr_unit8, mode="L")
                    img_to_save = img_to_save.resize((115, 115), Image.Resampling.NEAREST)
                    img_to_save.save(way)
                except Exception as e:
                    print(f"Fehler beim Speichern des Bildes {filename}: {e}")
                if len(wrongs) >= max_picture:
                    break
        # Tabellengröße
        rows = len(wrongs) + 1
        cols = 3
        cell_w = 150
        cell_h = 40
        width = cols * cell_w
        height = rows * cell_h
        # Tabelle als Bild erzeugen
        picture = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(picture)
        # Optional: Schriftart (funktioniert nur, wenn eine TTF-Datei vorhanden ist)
        try:
            font = ImageFont.truetype("arial.ttf", 22)
        except:
            font = None  # Verwende Standardschrift
        # Kopfzeile
        header = ["Bild-Index", "Wahr", "Vorhersage"]
        for c, text in enumerate(header):
            draw.rectangle([c*cell_w, 0, (c+1)*cell_w, cell_h], outline="black", 
                            width=2)
            draw.text((c*cell_w+10, 5), text, fill="black", font=font)
        # Zeilen
        for r, (idx, label_true, label_pred) in enumerate(wrongs, start=1):
            for c, value in enumerate([idx, label_true, label_pred]):
                draw.rectangle([c*cell_w, r*cell_h, (c+1)*cell_w, (r+1)*cell_h], 
                                outline="black", width=1)
                draw.text((c*cell_w+10, r*cell_h+5), str(value), fill="black", 
                            font=font)
        # Speichern der Tabelle
        table_way = os.path.join(data_way, "falsch_klassifiziert_tabelle.png")
        picture.save(table_way)
        print(f"Tabelle der falsch klassifizierten Bilder gespeichert als"
                f"{table_way}")

    def save_error_heatmap(self, net, X, y, max_picture, 
                            data_way="heatmap_fehler"):
        '''
        Erstellt eine Heatmap (PNG), die zeigt, in welchen Bildbereichen das 
        Netzwerk besonders häufig Fehlklassifikationen hat.

        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder
        y : np.ndarray
            Wahre Labels
        max_picture : int
            Maximale Anzahl falsch klassifizierter Bilder, die einfließen sollen.
        data_way : str
            Dateiname für die Ausgabegrafik.
        '''
        # Heatmap-Array für Fehler
        error_map = np.zeros((28, 28), dtype=np.float32)
        count = 0
        for i in range(len(X)):
            x_img = X[i].reshape(28, 28)
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                # Zähle Pixelintensitäten falsch klassifizierter Bilder
                error_map += x_img  
                count += 1
                if count >= max_picture:
                    break
        if count == 0:
            print("Keine Fehlklassifikationen für Heatmap gefunden.")
            return
        # Mittelwert bilden und normalisieren
        heatmap = error_map / count
        heatmap = 255 * (heatmap - heatmap.min()) / (np.ptp(heatmap) + 1e-8)
        heatmap_img = Image.fromarray(heatmap.astype(np.uint8)).convert("L").resize((140, 140), Image.Resampling.NEAREST)
        # Farbige Heatmap erzeugen (optional)
        heatmap_img = heatmap_img.convert("RGB")
        pixels = heatmap_img.load()
        for i in range(140):
            for j in range(140):
                # je heller, desto röter (z. B. für Fehler)
                value = heatmap_img.getpixel((i, j))[0]
                pixels[i, j] = (value, 0, 255 - value)  # rot-blau Verlauf
        os.makedirs((data_way), exist_ok=True)
        save_way = os.path.join(data_way, "heatmap_fehler.png")
        heatmap_img.save(save_way)
        print(f"Heatmap der Fehlklassifizierungen gespeichert als {save_way}")

    def save_error_heatmap_csv(self, net, X, y, max_picture, 
                                data_way="heatmap_fehler.csv"):
        '''
        Speichert die Fehlpixel-Heatmap als CSV-Datei (nur Werte, ohne Kopfzeile,
        Semikolon-getrennt).
        
        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder (N, 28, 28 oder N, 784).
        y : np.ndarray
            Wahre Labels (N).
        max_picture : int
            Max. Anzahl einfließender Fehlklassifikationen.
        data_way : str
            Pfad zur Ausgabedatei (.csv).
        '''
        error_map = np.zeros((28, 28), dtype=np.float32)
        count = 0
        for i in range(len(X)):
            x_img = X[i].reshape(28, 28)
            picture = X[i].flatten() / 255.0
            output_data = net.predict(picture)
            pred = np.argmax(output_data)
            if pred != y[i]:
                error_map += x_img
                count += 1
                if count >= max_picture:
                    break
        if count == 0:
            print("Keine Fehlklassifikationen für Heatmap gefunden.")
            return
        heatmap = error_map / count
        # Ordner für die CSV-Datei erstellen, falls nicht vorhanden
        dir_name = os.path.dirname(data_way)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # Schreiben als CSV-Textdatei
        with open(data_way, "w", encoding="utf-8") as f:
            for row in heatmap:
                f.write(";".join(f"{v:.2f}" for v in row) + "\n")
        print(f"Fehler-Heatmap als Text gespeichert: {data_way}")

    def save_confusion_heatmap(self, net, X, y, data_way="confusion_heatmap"):
        '''
        Erstellt und speichert eine Heatmap der Confusion-Matrix als PNG-Bild.

        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder
        y : np.ndarray
            Wahre Labels
        data_way : str
            Dateiname für die Ausgabegrafik.
        '''
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes)/255.0), 
                            axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        max_val = np.max(matrix)
        cell = 42
        picture = Image.new("RGB", (cell * num_classes, cell * num_classes), "white")
        draw = ImageDraw.Draw(picture)
        for r in range(num_classes):
            for c in range(num_classes):
                val = matrix[r, c]
                # Einfache Heatmap von weiß (0) bis rot (max)
                rot = int(255 * val / max_val) if max_val > 0 else 0
                col = (255, 255 - rot, 255 - rot)
                draw.rectangle([c*cell, r*cell, (c+1)*cell, (r+1)*cell], fill=col)
                draw.text((c*cell+cell//4, r*cell+cell//4), str(val), fill="black")
        # Ordner sicherstellen
        os.makedirs(data_way, exist_ok=True)
        # Pfad mit Dateiname
        save_way = os.path.join(data_way, "confusion_heatmap.png")
        picture.save(save_way)
        print(f"Heatmap der Confusion-Matrix gespeichert: {save_way}")

    def save_confusion_matrix(self, net, X, y, data_way="confusion_matrix.png"):
        '''
        Erstellt und speichert eine Verwechslungs-Matrix (Confusion Matrix) als 
        PNG-Bild.

        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder
        y : np.ndarray
            Wahre Labels
        data_way : str
            Dateiname für die Ausgabegrafik.
        '''
        # Vorhersagen berechnen
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes)/255.0), axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        # Bildgröße
        cell_w = 55
        cell_h = 40
        width = (num_classes + 1) * cell_w
        height = (num_classes + 1) * cell_h
        picture = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(picture)
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = None
        # Header
        for i in range(num_classes):
            draw.rectangle([cell_w + i*cell_w, 0, cell_w + (i+1)*cell_w, cell_h], 
                            outline="black", width=2)
            draw.text((cell_w + i*cell_w + 10, 8), str(i), fill="black", font=font)
            draw.rectangle([0, cell_h + i*cell_h, cell_w, cell_h + (i+1)*cell_h], 
                            outline="black", width=2)
            draw.text((10, cell_h + i*cell_h + 8), str(i), fill="black", font=font)
        # Werte eintragen
        for r in range(num_classes):
            for c in range(num_classes):
                val = matrix[r, c]
                col = "red" if (r == c) else "black"
                draw.rectangle([cell_w + c*cell_w, cell_h + r*cell_h, 
                                cell_w + (c+1)*cell_w, cell_h + (r+1)*cell_h], 
                                outline="black", width=1)
                draw.text((cell_w + c*cell_w + 8, cell_h + r*cell_h + 10), 
                            str(val), fill=col, font=font)
        # Beschriftungen
        draw.text((cell_w//3, 2), "wahr", fill="black", font=font)
        draw.text((cell_w*2, height - cell_h + 2), "vorhergesagt →", fill="black", 
                    font=font)
        picture.save(data_way)
        print(f"Confusion-Matrix als Tabelle gespeichert: {data_way}")

    def save_confusion_matrix_csv(self, net, X, y, 
                                    data_way="confusion_matrix.csv"):
        '''
        Speichert die Confusion-Matrix (Verwechslungs-Matrix) als CSV-Datei 
        (nur Werte, ohne Kopfzeile, Semikolon-getrennt).
        
        Parameter:
        net : NeuralNetTraining
            Das trainierte Netzwerk.
        X : np.ndarray
            Testbilder.
        y : np.ndarray
            Wahre Labels.
        data_way : str
            Pfad zur Ausgabedatei (.txt).
        '''
        # Confurion-Matrix berechnen
        preds = np.argmax(net.predict(X.reshape(-1, net.input_nodes)/255.0), axis=1)
        num_classes = np.max(y) + 1
        matrix = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y, preds):
            matrix[int(true), int(pred)] += 1
        # Ordner für die CSV-Datei erstellen, falls nicht vorhanden
        dir_name = os.path.dirname(data_way)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        # Schreiben als CSV-Textdatei
        with open(data_way, "w", encoding="utf-8") as f:
            # Kopfzeile z.B. für Export in Excel
            f.write("wahr/vorhergesagt;" + ";".join(str(i) for i in range(num_classes)) + "\n")
            for r in range(num_classes):
                row = ";".join(str(matrix[r, c]) for c in range(num_classes))
                f.write(f"{r};{row}\n")
        print(f"Confusion-Matrix als Text gespeichert: {data_way}")

    def save_sample_picture_train_labeled(self, data_way="mnist_bsp_train_labeled"):
        '''
        Speichert Trainingsbeispiele mit eingebettetem Label oben links im Bild.
        
        Parameter:
        data_way : str
            Verzeichnispräfix für die Ausgabedateien.
        '''
        # Ordner anlegen
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_train[i].astype(np.uint8))
            picture_bigger = picture.resize((115, 115), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(picture_bigger)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = None
            # Label oben links einblenden
            label = str(self.y_train[i])
            draw.rectangle([0, 0, 35, 35], fill="yellow")
            draw.text((8, 2), label, fill="black", font=font)
            # Ordner zum speichern
            filename = f"trainbild_{i}_label_{self.y_train[i]}.png"
            way = os.path.join(data_way, filename)
            picture_bigger.save(way)
        print(f"{NUMBER} Trainingsbeispiele mit Label-Overlay gespeichert.")

    def save_sample_picture_test_labeled(self, data_way="mnist_bsp_test_labeled"):
        '''
        Speichert Testbeispiele mit eingebettetem Label oben links im Bild.
        
        Parameter:
        data_way : str
            Verzeichnispräfix für die Ausgabedateien.
        '''
        # Ordner anlegen
        os.makedirs(data_way, exist_ok=True)
        for i in range(NUMBER):
            picture = Image.fromarray(self.X_test[i].astype(np.uint8))
            picture_bigger = picture.resize((115, 115), Image.Resampling.NEAREST)
            draw = ImageDraw.Draw(picture_bigger)
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = None
            label = str(self.y_test[i])
            draw.rectangle([0, 0, 35, 35], fill="yellow")
            draw.text((8, 2), label, fill="black", font=font)
            # Ordner zum speichern
            filename = f"testbild_{i}_label_{self.y_test[i]}.png"
            way = os.path.join(data_way, filename)
            picture_bigger.save(way)
        print(f"{NUMBER} Testbeispiele mit Label-Overlay gespeichert.")

# Hauptprogramm
if __name__ == "__main__":
    # Zeitstempel für Dateinamen
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Daten laden und splitten
    loader = MNISTDownLoad(test_size = 0.2)
    loader.load_and_split_mnist_data()
    loader.save_mnist_data_new(os.path.join(OUTPUT_DIR, 
                                    f"mnist_daten_split_{timestamp}.npz"))
    # Netz initialisieren
    net = NeuralNetTraining()
    # Startzeit für die Messung der Trainingszeit
    start_time = time.time()
    # Training mit Fehler - und Genauigkeitslisten
    error_list, acc_list = net.training_epochs(loader.X_train, loader.y_train, 
                                                loader.X_test, loader.y_test)
    # Endzeit für die Messung der Trainingszeit
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training abgeschlossen in {training_time/60:.2f} Minuten")
    # Vorhersage
    petition = loader.X_test[0].flatten() / 255.0
    output_data = net.predict(petition.reshape(1, -1))
    digit = np.argmax(output_data)
    print("Netzwerkausgabe:", output_data.flatten())
    print("Vermutete Ziffer:", digit) 
    # Visualisierung
    vis = Visualizer(loader.X_train, loader.y_train, loader.X_test, 
                        loader.y_test)
    vis.save_sample_picture_train(data_way=os.path.join(OUTPUT_DIR,
                                                f"mnist_bsp_train_{timestamp}"))
    vis.save_sample_picture_test(data_way=os.path.join(OUTPUT_DIR,
                                                f"mnist_bsp_test_{timestamp}"))           
    vis.save_error_curve(error_list, data_way=os.path.join(OUTPUT_DIR,
                                                f"fehlerkurve_{timestamp}.png"))
    vis.save_accuracy(acc_list, data_way=os.path.join(OUTPUT_DIR,
                                        f"genauigkeitskurve_{timestamp}.png"))
    vis.save_train_vs_test_curve(error_list, acc_list, 
        data_way=os.path.join(OUTPUT_DIR, f"train_vs_test_{timestamp}.png"))
    vis.save_weights(net.weight_input_hidden, NUMBER, 
            data_way=os.path.join(OUTPUT_DIR, f"gewichte_{timestamp}.png"))
    vis.save_misclassified(net, loader.X_test, loader.y_test, max_picture=10, 
        data_way=os.path.join(OUTPUT_DIR, f"falsch_klassifiziert_{timestamp}"))    
    vis.save_misclassified_table(net, loader.X_test, loader.y_test, max_picture=10,
            data_way=os.path.join(OUTPUT_DIR, f"falsch_klassifiziert_tabelle_"
            f"{timestamp}.png"))
    vis.save_error_heatmap(net, loader.X_test, loader.y_test, max_picture=100,
        data_way=os.path.join(OUTPUT_DIR, f"heatmap_fehler_{timestamp}.png"))
    vis.save_error_heatmap_csv(net, loader.X_test, loader.y_test, max_picture=100,
        data_way=os.path.join(OUTPUT_DIR, f"heatmap_fehler_{timestamp}.csv"))
    vis.save_confusion_matrix(net, loader.X_test, loader.y_test, 
        data_way=os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.png"))
    vis.save_confusion_heatmap(net, loader.X_test, loader.y_test, 
        data_way=os.path.join(OUTPUT_DIR, f"confusion_heatmap_{timestamp}.png"))
    vis.save_confusion_matrix_csv(net, loader.X_test, loader.y_test,
        data_way=os.path.join(OUTPUT_DIR, f"confusion_matrix_{timestamp}.csv"))
    vis.save_sample_picture_train_labeled(data_way=os.path.join(OUTPUT_DIR, 
                                        f"mnist_bsp_train_labeled_{timestamp}"))
    vis.save_sample_picture_test_labeled(data_way=os.path.join(OUTPUT_DIR, 
                                        f"mnist_bsp_test_labeled_{timestamp}"))
    
    # Projektergebnis-Block 
    with open(os.path.join(OUTPUT_DIR, f"projekt_reflexion_{timestamp}.txt"), 
                "w", encoding="utf-8") as f:
        f.write("Projektabschluss – Reflexion und Auswertung\n")
        f.write("===========================================\n\n")
        f.write("╔══════════════════════════════════════════════════════════════════╗\n")
        f.write("║                 NEURONALES NETZ: PARAMETER-TABELLE               ║\n")
        f.write("╠════════════════════════════════════════╦═════════════════════════╣\n")
        f.write(f"║ Eingabeknoten                          ║ {INPUT_NODES:<20}    ║\n")
        f.write(f"║ Versteckte Knoten                      ║ {HIDDEN_NODES:<20}    ║\n")
        f.write(f"║ Ausgabeknoten                          ║ {OUTPUT_NODES:<20}    ║\n")
        f.write(f"║ Lernrate                               ║ {LEARNING_RATE:<20}    ║\n")
        f.write(f"║ Epochen                                ║ {EPOCHS:<20}    ║\n")
        f.write(f"║ Batch-Größe                            ║ {BATCH_SIZE:<20}    ║\n")
        f.write(f"║ Trainingsbeispiele                     ║ {len(loader.X_train):<20}    ║\n")
        f.write(f"║ Testbeispiele                          ║ {len(loader.X_test):<20}    ║\n")
        f.write(f"║ Verlust (letzte Epoche)                ║ {error_list[-1]:<20.6f}    ║\n")
        if acc_list[-1] is not None:
            f.write(f"║ Genauigkeit (letzte Epoche)            ║ {acc_list[-1]*100:<17.2f} %     ║\n")
        else:
            f.write(f"║ Genauigkeit (letzte Epoche)            ║ Nicht berechnet       ║\n")
        # → Trainingszeit ergänzen:
        f.write(f"║ Trainingszeit                          ║ {training_time/60:<16.2f} min    ║\n")
        f.write(f"║ Gewichtsinitialisierung                ║ uniform(-0.5, 0.5)      ║\n")
        f.write(f"║ Bias verwendet                         ║ Ja                      ║\n")
        f.write(f"║ Aktivierungsfunktion                   ║ Sigmoid                 ║\n")
        f.write(f"║ Verlustfunktion                        ║ MSE (Mean Squared Error)║\n")
        f.write(f"║ Optimierer                             ║ SGD (Gradientenabstieg) ║\n")
        f.write(f"║ Zufallsseed                            ║ 42                      ║\n")
        f.write(f"║ Trainingsstart (Timestamp)             ║ {timestamp:<20}    ║\n")
        f.write("╚════════════════════════════════════════╩═════════════════════════╝\n")
        f.write("\n")
    # Abschlussmeldung
    print(f"Alle Ergebnisse, Bilder und Dokumentationen wurden im Ordner "
            f"{OUTPUT_DIR} mit Zeitstempel {timestamp} gespeichert.")

