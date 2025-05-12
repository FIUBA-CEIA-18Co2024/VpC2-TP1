import os 
import shutil
import pandas as pd
from PIL import Image
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def explore_dataset(dataset_path: str = "dataset/") -> None:
    """
    Explora el contenido del dataset y muestra estadísticas como:
    - Cantidad de elementos por partición (train, test, valid).
    - Cantidad de elementos por clase en cada partición.
    - Cantidad de resoluciones distintas en las imágenes de cada partición.

    Args:
        dataset_path (str): Ruta al dataset.

    Returns:
        None
    """
    partitions = ['train', 'test', 'valid']
    for partition in partitions:
        partition_path = os.path.join(dataset_path, partition)
        if not os.path.exists(partition_path):
            print(f"La partición '{partition}' no existe en el dataset.")
            continue

        print(f"\nPartición: {partition}")
        class_counts = {}
        total_count = 0
        resolutions = set()

        for class_name in os.listdir(partition_path):
            class_path = os.path.join(partition_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                class_counts[class_name] = count
                total_count += count

                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        with Image.open(img_path) as img:
                            resolutions.add(img.size)
                    except Exception as e:
                        print(f"Error al procesar la imagen {img_path}: {e}")

        print(f"Total de elementos: {total_count}")
        for class_name, count in class_counts.items():
            print(f"  Clase '{class_name}': {count} elementos")
            
        print(f"Resoluciones distintas en la partición '{partition}': {len(resolutions)}")
        print("Primeras 5 resoluciones:")
        for resolution in list(resolutions)[:5]:
            print(f"  Resolución: {resolution}")
        if resolutions:
            max_resolution = max(resolutions, key=lambda x: x[0] * x[1])
            min_resolution = min(resolutions, key=lambda x: x[0] * x[1])
            print(f"Resolución máxima: {max_resolution}")
            print(f"Resolución mínima: {min_resolution}")

def map_counts_to_classnames(class_counts: Dict, class_names: Dict) -> Dict:
    """
    Convierte un dict {clase_id: cantidad} a {nombre_clase: cantidad}
    """
    return {class_names[cls]: count for cls, count in class_counts.items()}

def plot_dataset_distribution(dataset_path: str = "dataset/") -> None:
    """
    Genera un gráfico de barras que muestra la distribución de clases por partición
    (train/test/valid), con colores personalizados y leyenda prolija.
    """

    partitions = ['train', 'test', 'valid']
    data = []
    partition_totals = {}

    for partition in partitions:
        partition_path = os.path.join(dataset_path, partition)
        if not os.path.exists(partition_path):
            continue

        total_count = 0
        for class_name in sorted(os.listdir(partition_path)):
            class_path = os.path.join(partition_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                label = f"{partition.capitalize()} (Total: {{}})"  # Se completa después
                data.append({'Clase': class_name, 'Partición': partition, 'Cantidad': count})
                total_count += count

        partition_totals[partition] = total_count

    if data:
        df = pd.DataFrame(data)

        # Actualizar nombres de partición con los totales
        df['Partición'] = df['Partición'].apply(
            lambda p: f"{p.capitalize()} (Total: {partition_totals.get(p, 0)})"
        )

        # Paleta fija (puede personalizarse si hay más de 2)
        color_palette = {
            f"Train (Total: {partition_totals.get('train', 0)})": '#1f77b4',
            f"Test (Total: {partition_totals.get('test', 0)})": '#ff7f0e',
            f"Valid (Total: {partition_totals.get('valid', 0)})": '#2ca02c',
        }

        plt.figure(figsize=(14, 7))
        ax = sns.barplot(
            data=df,
            x='Clase', y='Cantidad',
            hue='Partición',
            palette=color_palette
        )

        # Etiquetas encima de cada barra
        for container in ax.containers:
            ax.bar_label(container, padding=3, fontsize=9)

        plt.title("Distribución de clases en train y test", fontsize=14)
        plt.xlabel("Clases de emociones")
        plt.ylabel("# de imágenes")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

def plot_class_distribution(partition_to_class_counts):
    """
    Recibe un dict {partición: {clase: cantidad}} y genera un gráfico de barras.
    """
    df = pd.DataFrame(partition_to_class_counts).fillna(0).astype(int)
    df = df.T  # ahora cada fila es una partición
    df = df.reset_index().melt(id_vars='index', var_name='Clase', value_name='Cantidad')
    df = df.rename(columns={'index': 'Partición'})

    # Totales por partición
    totales = df.groupby('Partición')['Cantidad'].sum().to_dict()

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(data=df, x='Clase', y='Cantidad', hue='Partición', palette='muted')

    for container in ax.containers:
        ax.bar_label(container, padding=3, fontsize=9)

    plt.title("Distribución de clases en train, valid y test", fontsize=14)
    plt.xlabel("Clases")
    plt.ylabel("# de imágenes")
    plt.xticks(rotation=45, ha='right')

    # Leyenda con totales
    handles, labels = ax.get_legend_handles_labels()
    labels = [f"{label} (Total: {totales[label]})" for label in labels]
    ax.legend(handles, labels, title="Partición")

    plt.tight_layout()
    plt.show()
        
def reorganize_dataset(source_path, target_path, class_map, partitions=("train", "test", "valid")):
    """
    Reorganiza un dataset de imágenes agrupando clases en carpetas estandarizadas.

    Args:
        source_path (str or Path): Ruta al dataset original (donde están /train, /test, /valid).
        target_path (str or Path): Ruta donde guardar el nuevo dataset limpio.
        class_map (dict): Diccionario de mapeo {nombre_incompleto: clase_normalizada}.
        partitions (tuple): Lista de particiones a procesar (default: ("train", "test", "valid")).
    """
    source_path = Path(source_path)
    target_path = Path(target_path)

    print(f"Reorganizando dataset desde: {source_path}")
    print(f"Guardando en: {target_path}\n")

    for partition in partitions:
        partition_path = source_path / partition
        if not partition_path.exists():
            print(f"Partición no encontrada: {partition}")
            continue

        for class_dir in partition_path.iterdir():
            if class_dir.is_dir():
                class_base = class_dir.name.split("_")[0].replace("-", ".").lower()

                # Buscar clase en el mapeo
                matched = False
                for key in class_map:
                    if key in class_base:
                        class_target = class_map[key]
                        matched = True
                        break

                if not matched:
                    print(f"Clase no mapeada: {class_dir.name}")
                    continue

                dest_dir = target_path / partition / class_target
                dest_dir.mkdir(parents=True, exist_ok=True)

                count = 0
                for img in class_dir.glob("*.*"):
                    shutil.copy(img, dest_dir / img.name)
                    count += 1

                print(f"{partition}/{class_dir.name} → {class_target} ({count} imágenes)")

    print("\nDataset reorganizado correctamente.")
   
def show_first_image_per_class_no_transform(dataset):
    """
    Muestra la primera imagen original (sin transformaciones) de cada clase.

    Args:
        dataset: instancia de torchvision.datasets.ImageFolder
    """
    seen = set()
    class_to_path = {}

    for path, label in dataset.samples:
        if label not in seen:
            class_to_path[label] = path
            seen.add(label)
        if len(seen) == len(dataset.classes):
            break

    num_classes = len(class_to_path)
    plt.figure(figsize=(3 * num_classes, 3))

    for i, (label, img_path) in enumerate(sorted(class_to_path.items())):
        img = Image.open(img_path).convert('L')  # 'L' = grayscale

        plt.subplot(1, num_classes, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(dataset.classes[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()
     
def show_first_image_per_class(dataset, class_names):
    """
    Muestra la primera imagen de cada clase del dataset.
    
    Args:
        dataset: objeto tipo torchvision.datasets.ImageFolder
        class_names: lista de nombres de clases (dataset.classes)
    """
    seen_classes = set()
    class_to_img = {}

    for img, label in dataset:
        if label not in seen_classes:
            class_to_img[label] = img
            seen_classes.add(label)
        if len(seen_classes) == len(class_names):
            break

    num_classes = len(class_to_img)
    plt.figure(figsize=(3 * num_classes, 3))

    for i, (label, img) in enumerate(sorted(class_to_img.items())):
        plt.subplot(1, num_classes, i + 1)
        plt.imshow(F.to_pil_image(img), cmap="gray")
        plt.title(class_names[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()