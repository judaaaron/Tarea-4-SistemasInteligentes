import sys
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, recall_score, f1_score, accuracy_score, classification_report, average_precision_score
import pickle
import matplotlib.pyplot as plt

import time
import csv
import os

modelFile = sys.argv[1]
validationFile = sys.argv[2]

validation_data = pd.read_csv(validationFile)
clases = validation_data.pop('class')
clasesNoRepetidas = clases.drop_duplicates()
validation_data = validation_data.replace({"Si": 1, "No": 0})
cols = validation_data.columns.tolist()

x = validation_data[cols]
y = clases

f = open(modelFile, 'rb')
f.seek(0)
pickle_model = pickle.load(f)

modelo = pickle_model["modelo"]
start = time.time()
y_pred = modelo.predict(x)
end = time.time()
print("Tiempo total de predicción: ", end-start)


report = classification_report(
    y, y_pred, labels=clasesNoRepetidas, digits=4, output_dict=True, zero_division=0)
print(report)
trainAcc = '{:.4f}'.format(pickle_model["trainingAcc"])
valAcc = '{:.4f}'.format(report["accuracy"])
avgF1 = '{:.4f}'.format(report["macro avg"]["f1-score"])
avgPrecision = '{:.4f}'.format(report["macro avg"]["precision"])
avgRecall = '{:.4f}'.format(report["macro avg"]["recall"])

isFile1 = os.path.exists("./Resultados/resultadosKNN.csv")
headerKNN = ["Train Dataset", "k", "Train Acc.", "Val Acc.",
    "Val. Avg Rec", "Val Avg. Prec", "Val Avg. F1", "Time Train", "Time Val."]

if (isFile1):
    with open('./Resultados/resultadosKNN.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        if (pickle_model["size"] == "very"):
            pickle_model["size"] = "Very-Large"
        row = [pickle_model["size"], pickle_model["k"], trainAcc, valAcc,
            avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]), '{:.6f}'.format(end-start)]
        writer.writerow(row)
        print(
            "Datos escritos en ./Resultados/resultadosKNN.csv satisfactoriamente")
else:
    with open('./Resultados/resultadosKNN.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headerKNN)
        if (pickle_model["size"] == "very"):
            pickle_model["size"] = "Very-Large"
        row = [pickle_model["size"], pickle_model["k"], trainAcc, valAcc, avgRecall, avgPrecision, avgF1, '{:.6f}'.format(pickle_model["trainingTime"]) ,'{:.6f}'.format(end-start)]
        writer.writerow(row)
        print("Archivo creado y datos escritos en ./Resultados/resultadosKNN.csv satisfactoriamente")
