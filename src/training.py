from src.early_stopping import EarlyStopping
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import os
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings("ignore")
# Definimos el device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fit(
    model: torch.nn.Sequential, 
    train_loader: DataLoader, 
    eval_loader: DataLoader, 
    epochs: int = 5,
    weight_decay: bool = False,
    model_name: str = 'best_model.pth',
    results_file: str = 'train_data.json'
    ) -> tuple:
    """
    Entrena el modelo y evalúa su rendimiento en el conjunto de validación.

    Args:
        model (torch.nn.Sequential): Modelo PyTorch
        train_loader (DataLoader): Dataloader train
        eval_loader (DataLoader): Dataloader de validación
        epochs (int, optional): Epocas a entrenar. Defaults to 5.
    """
    # enviamos el modelo al device
    model.to(device)
    # definimos optimizer y la función de pérdida
    if not weight_decay:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    # preparamos listas para guardar las loss y la acc a lo largo de la epocas
    epoch_t_loss = []
    epoch_v_loss = []
    epoch_t_acc = []
    epoch_v_acc = []
    epoch_v_f1 = []
    epoch_t_f1 = []
    
    # inicializamos early stopping
    early_stopping = EarlyStopping(patience=12, min_delta=0.001)
    
    # iteramos en las epocas
    for epoch in range(1, epochs+1):
        # ponemos el modelo en train
        model.train()
        # listas de loss y acc de train para esta epoca
        # así despues calculamos la media
        # por que el dataset lo pasamos de a batches
        train_loss, train_acc = [], []
        train_preds = []
        train_targets = []
        
        bar = tqdm(train_loader)
        for batch in bar:
            X, y = batch  # sacamos X e y del batch
            X, y = X.to(device), y.to(device) # lo enviamos al device
            optimizer.zero_grad() # llevamos optimizer a zero
            y_hat = model(X)  # corremos el modelo y vemos su predicción, esto es la pasada forward (ejecuta forward method del modelo)
                        
            if isinstance(y_hat, tuple):
                # main_logits, aux_logits para modelos con salida intermedia como Inception
                main_logits, aux_logits = y_hat
                loss1 = criterion(main_logits, y)
                loss2 = criterion(aux_logits, y)
                loss = loss1 + 0.4 * loss2  # peso sugerido en el paper
                y_hat = y_hat[0]
            else:
                loss = criterion(y_hat, y)
            
            # loss = criterion(y_hat, y)  # calculamos la pérdida
            loss.backward() # back-propagations
            optimizer.step()  # step del optimizer
            train_loss.append(loss.item()) # vamos guardando la pérdida de este batch, en la perdida de la epoca
            
            preds = torch.argmax(y_hat, axis=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y.cpu().numpy())
            
            # calculo de la acc
            acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
            train_acc.append(acc) # vamos guardando la acc de este batch, en la acc de la epoca
                        
            # seteamos descriptores en la barra
            bar.set_description(f"loss {np.mean(train_loss):.5f} acc {np.mean(train_acc):.5f}")

        # luego de pasar todo el batch, guardamos la perdida y acc media del train
        epoch_t_loss.append(np.mean(train_loss))
        epoch_t_acc.append(np.mean(train_acc))
        # calculo f1 score
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        epoch_t_f1.append(train_f1)

        # ahora viene es test
        bar = tqdm(eval_loader)
        # listas de loss y acc de test para esta epoca
        # así despues calculamos la media
        # por que el dataset lo pasamos de a batches
        # agregamos all_preds y all_targets para guardar las predicciones y targets
        # así después podemos calcular el f1 score
        # y la matriz de confusión
        val_loss, val_acc = [], []
        all_preds = []
        all_targets = []
        
        # ponemos en eval el modelo
        model.eval()
        with torch.no_grad():
            for batch in bar:
                X, y = batch
                X, y = X.to(device), y.to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_loss.append(loss.item())
                acc = (y == torch.argmax(y_hat, axis=1)).sum().item() / len(y)
                val_acc.append(acc)
                
                preds = torch.argmax(y_hat, axis=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
                bar.set_description(f"val_loss {np.mean(val_loss):.5f} val_acc {np.mean(val_acc):.5f}")
            # Calcular F1-score
            val_f1 = f1_score(all_targets, all_preds, average='weighted')
            print(f"Epoch {epoch}/{epochs} "
                f"loss {np.mean(train_loss):.5f} val_loss {np.mean(val_loss):.5f} "
                f"acc {np.mean(train_acc):.5f} val_acc {np.mean(val_acc):.5f} "
                f"train_f1 {train_f1:.5f} val_f1 {val_f1:.5f}")
    
        epoch_v_loss.append(np.mean(val_loss))
        epoch_v_acc.append(np.mean(val_acc))
        epoch_v_f1.append(val_f1)
        
        # Guardamos el mejor modelo
        if epoch == 1 or np.mean(val_loss) < min(epoch_v_loss[:-1]): # -1 para no contar el que acabo de agregar
            print(f"Mejor modelo guardado en la época {epoch} con val_loss {np.mean(val_loss):.5f}")
            torch.save(model.state_dict(), model_name)
            print("Mejor modelo guardado.")
        
        # Mecanismo Early stopping
        early_stopping(np.mean(val_loss))
        if early_stopping.early_stop:
            print("Patience counter alcanzado. Deteniendo entrenamiento por Early Stopping.")
            break

    # Guardamos los datos de entrenamiento en JSON
    with open(results_file, 'w') as f:
        json.dump({
            'epoch_t_loss': epoch_t_loss,
            'epoch_v_loss': epoch_v_loss,
            'epoch_t_acc': epoch_t_acc,
            'epoch_v_acc': epoch_v_acc,
            'epoch_t_f1': epoch_t_f1,
            'epoch_v_f1': epoch_v_f1
        }, f, indent=4)
        print("Datos de entrenamiento guardados en train_data.json")
    

    return epoch_t_loss, epoch_v_loss, epoch_t_acc, epoch_v_acc, epoch_t_f1, epoch_v_f1

def plot_train_results_from_json(json_file: str = 'train_data.json') -> None:
    """
    Plotea los resultados del entrenamiento desde un archivo JSON.
    """
    # Cargar los datos del archivo JSON
    with open(json_file, 'r') as f:
        train_data = json.load(f)

    # Extraer métricas
    epoch_t_loss = train_data['epoch_t_loss']
    epoch_v_loss = train_data['epoch_v_loss']
    epoch_t_acc = train_data['epoch_t_acc']
    epoch_v_acc = train_data['epoch_v_acc']
    epoch_t_f1 = train_data['epoch_t_f1']
    epoch_v_f1 = train_data['epoch_v_f1']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    epochs = list(range(1, len(epoch_t_loss) + 1))

    # === LOSS ===
    axs[0].plot(epochs, epoch_t_loss, label="Entrenamiento", color='royalblue')
    axs[0].plot(epochs, epoch_v_loss, label="Validación", color='darkorange')
    axs[0].set_title("Pérdida (Loss)", fontsize=14)
    axs[0].set_xlabel("Época", fontsize=12)
    axs[0].set_ylabel("Valor", fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    # === ACCURACY ===
    axs[1].plot(epochs, epoch_t_acc, label="Entrenamiento", color='royalblue')
    axs[1].plot(epochs, epoch_v_acc, label="Validación", color='darkorange')
    axs[1].set_title("Accuracy", fontsize=14)
    axs[1].set_xlabel("Época", fontsize=12)
    axs[1].set_ylabel("Valor", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    # === F1 SCORE ===
    axs[2].plot(epochs, epoch_t_f1, label="Entrenamiento", color='royalblue')
    axs[2].plot(epochs, epoch_v_f1, label="Validación", color='darkorange')
    axs[2].set_title("F1 Score", fontsize=14)
    axs[2].set_xlabel("Época", fontsize=12)
    axs[2].set_ylabel("Valor", fontsize=12)
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
