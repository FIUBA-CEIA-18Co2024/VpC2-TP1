import os 
import shutil
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt


def explore_dataset(dataset_path):
    """
    Explora el contenido del dataset y muestra estadísticas como:
    - Cantidad de elementos por partición (train, test, valid).
    - Cantidad de elementos por clase en cada partición.

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

        for class_name in os.listdir(partition_path):
            class_path = os.path.join(partition_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                class_counts[class_name] = count
                total_count += count

        print(f"Total de elementos: {total_count}")
        for class_name, count in class_counts.items():
            print(f"  Clase '{class_name}': {count} elementos")

def plot_dataset_distribution(dataset_path):
    """
    Genera un gráfico de barras horizontales que muestra la distribución de elementos por clase,
    con un hue para cada partición del dataset. Incluye un legend box con el total de muestras por
    partición y el total sumando todos los datos del dataset.

    Args:
        dataset_path (str): Ruta al dataset.

    Returns:
        None
    """

    partitions = ['train', 'test', 'valid']
    data = []
    partition_totals = {}

    for partition in partitions:
        partition_path = os.path.join(dataset_path, partition)
        if not os.path.exists(partition_path):
            print(f"La partición '{partition}' no existe en el dataset.")
            continue

        total_count = 0
        for class_name in os.listdir(partition_path):
            class_path = os.path.join(partition_path, class_name)
            if os.path.isdir(class_path):
                count = len(os.listdir(class_path))
                data.append({'Clase': class_name, 'Partición': partition, 'Cantidad': count})
                total_count += count

        partition_totals[partition] = total_count

    if data:
        total_dataset_count = sum(partition_totals.values())
        legend_text = "\n".join([f"{partition}: {count}" for partition, count in partition_totals.items()])
        legend_text += f"\nTotal: {total_dataset_count}"

        df = pd.DataFrame(data)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=df, y='Clase', x='Cantidad', hue='Partición', orient='h', palette='viridis')
        plt.title("Distribución de elementos por clase y partición")
        plt.xlabel("Cantidad de elementos")
        plt.ylabel("Clases")
        plt.legend(title="Particiones", loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.gcf().text(0.9, 0.5, legend_text, fontsize=10, verticalalignment='center', transform=plt.gcf().transFigure)
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