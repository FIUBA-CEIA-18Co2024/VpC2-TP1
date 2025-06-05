import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

def evaluate_model(model: torch.nn.Module, data_loader: DataLoader, class_names: list, device: str, output_file: str = 'evaluation_results.json') -> None:
    """
    Evalúa un modelo en un dataset de test y genera métricas de evaluación.

    Args:
        model (torch.nn.Module): Modelo a evaluar.
        data_loader (DataLoader): DataLoader del dataset de test.
        class_names (list): Lista de nombres de las clases.
        device (str): Dispositivo para la evaluación ('cpu' o 'cuda').
        output_file (str): Nombre del archivo JSON para guardar los resultados.

    Returns:
        None
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, axis=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Generar matriz de confusión
    cm = confusion_matrix(all_targets, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Reporte de clasificación
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)

    # Guardar resultados en un archivo JSON
    results = {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Resultados guardados en {output_file}")

    # Visualizar ambas matrices de confusión juntas en la misma fila
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Matriz de Confusión Absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Matriz de Confusión (Absoluta)')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Etiqueta Verdadera')

    # Matriz de Confusión Normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Matriz de Confusión (Normalizada)')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Etiqueta Verdadera')

    plt.tight_layout()
    plt.show()
    

def generate_classification_report(model: torch.nn.Module, data_loader: DataLoader, class_names: list, device: str) -> dict:
    """
    Genera un informe de clasificación para el conjunto de test.

    Args:
        model (torch.nn.Module): Modelo a evaluar.
        data_loader (DataLoader): DataLoader del conjunto de test.
        class_names (list): Lista de nombres de las clases.
        device (str): Dispositivo para la evaluación ('cpu' o 'cuda').

    Returns:
        dict: Diccionario con métricas de evaluación.
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels, reduction='sum')  # Calcular pérdida
            total_loss += loss.item()
            preds = torch.argmax(outputs, axis=1)
            all_preds.extend(preds.cpu().numpy())      # Se mandan a cpu para poder ser computados las métricas con scikit-learn
            all_targets.extend(labels.cpu().numpy())   # Se mandan a cpu para poder ser computados las métricas con scikit-learn

    # Calcular métricas
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    report = classification_report(all_targets, all_preds, target_names=class_names)

    # Imprimir resultados
    print("Classification Report:")
    print(report)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"Loss: {avg_loss:.4f}")

    return {
        "accuracy": accuracy,
        "f1_score": f1,
        "loss": avg_loss,
        "classification_report": report
    }


def plot_cm_from_results(output_file, labels=['adenocarcinoma', 'large_cell_carcinoma', 'normal', 'squamous_cell_carcinoma']):
    # Leer archivo JSON con las matrices
    with open(output_file, 'r') as f:
        data = json.load(f)

    # Convertir listas a numpy arrays
    cm = np.array(data['confusion_matrix'])
    cm_normalized = np.array(data['confusion_matrix_normalized'])

    # Crear figura con dos subplots lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Matriz de Confusión Absoluta
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title('Matriz de Confusión (Absoluta)')
    axes[0].set_xlabel('Predicción')
    axes[0].set_ylabel('Etiqueta Verdadera')

    # Matriz de Confusión Normalizada
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title('Matriz de Confusión (Normalizada)')
    axes[1].set_xlabel('Predicción')
    axes[1].set_ylabel('Etiqueta Verdadera')

    plt.tight_layout()
    plt.show()

def plot_report_from_results(output_file):
    with open(output_file, 'r') as f:
        data = json.load(f)
    # Separar accuracy
    report = data["classification_report"]
    accuracy = report.pop("accuracy")

    # Convertir a DataFrame
    df_report = pd.DataFrame(report).T  # Transponer para que las métricas estén en columnas
    print(df_report)

    # Mostrar accuracy por separado
    print(f"\nAccuracy: {accuracy:.4f}")