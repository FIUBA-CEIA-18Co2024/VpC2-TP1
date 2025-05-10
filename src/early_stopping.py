class EarlyStopping:
    """
    Implementaremos **Early Stopping** para optimizar el tiempo de entrenamiento y evitar sobreajuste. Este mecanismo detiene el entrenamiento cuando no se observa una mejora significativa en la función de pérdida del conjunto de validación durante un número consecutivo de épocas, definido por el parámetro *patience*.

    **Funcionamiento:**
    1. **Patience**: Es el número de épocas consecutivas sin mejora aceptable en la pérdida antes de detener el entrenamiento.
    2. **min_delta**: Define la mejora mínima requerida en la pérdida para considerar que el modelo está progresando. Si la pérdida de validación no mejora al menos en `min_delta` respecto a la mejor pérdida registrada, se incrementa el contador de paciencia.
    3. **Detención**: Si el contador de paciencia alcanza el valor de *patience*, el entrenamiento se detiene automáticamente.

    De esta manera nos aseguramos que el modelo no siga entrenándose innecesariamente cuando ya no está mejorando.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        """
        Args:
            patience (int): Número de épocas sin mejora antes de detener.
            min_delta (float): Mínima mejora para resetear el contador.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True