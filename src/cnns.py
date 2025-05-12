import torch
import torch.nn as nn
from torch.nn import functional as F



def conv_block(
    c_in: int, 
    c_out: int, 
    kernel_size: int = 3, 
    padding: str = 'same', 
    stride: int = 1, 
    pooling_kernel_size:int = 2
    ) -> torch.nn.Sequential:
    """
    Bloque de convolución seguido de activación y max pooling.

    Args:
        c_in (int): Parametros de entrada.
        c_out (int): Parametros de salida. (features o cantidad de kernels)
        kernel_size (int, optional): Tamaño del kernel cuadrado. Defaults to 3.
        padding (str, optional): Padding aplicado en la convolución. Puede ser un entero o 'same'. Defaults to 'same'.
        stride (int, optional): Stride (paso) de la convolución. Defaults to 1.
        pooling_kernel_size (int, optional): Tamaño del kernel de pooling. Defaults to 2.

    Returns:
        _type_: Módulo secuencial con conv → activación → max pooling.
    """
    return torch.nn.Sequential( # el módulo Sequential se engarga de hacer el forward de todo lo que tiene dentro.
        torch.nn.Conv2d(c_in, c_out, kernel_size, padding=padding, stride=stride),
        torch.nn.BatchNorm2d(c_out),
        torch.nn.ReLU(inplace=True),
        torch.nn.MaxPool2d(kernel_size=pooling_kernel_size)
    )
    
class CNN(torch.nn.Module):
    """
    Modelo CNN para clasificación de imágenes.
    """
    def __init__(self, n_channels: int = 1, n_outputs: int = 7) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self._build_model()

    def _build_model(self) -> None:
        """
        Construye el modelo CNN.

        Returns:
            torch.nn.Sequential: Modelo CNN.
        """
        # Parte convolucional
        self.conv1 = conv_block(self.n_channels, 32)     # imagen en grises: canal de entrada = 1
        self.conv2 = conv_block(32, 64)
        self.conv3 = conv_block(64, 128)
        self.conv4 = conv_block(128, 256)
        self.conv5 = conv_block(256, 512)
        self.dropout = torch.nn.Dropout2d(p=0.15) # Apagar píxeles aislados rompe coherencia espacial con dropout comun. Se usa dropout2d
        # Parte fully connected
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(512 * 7 * 7, 512), # se va a 7x7 por el maxpooling aplicado en 5 capas imagen de entrada 224x224
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(512, 256), # se aumenta la cantidad de neuronas para que el dropout no afecte tanto
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.25),            
            torch.nn.Linear(256, self.n_outputs)  # 7 clases de emociones
        )
        
        print('Red creada')
        print('arquitectura:')
        print(self)

        # Me fijo en el número de capas
        i=0
        for layer in self.children():
            i=i+1
        print('Número total de capas de CNN (conv+act+polling) + finales : ', i)

        # Me fijo en el número de parámetros entrenables
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Número total de parámetros a entrenar: ', pytorch_total_params)
        
    def forward(self, x) -> torch.Tensor:
        """
        Pasada forward del modelo.

        Args:
            x (torch.Tensor): Tensor de entrada.

        Returns:
            torch.Tensor: Tensor de salida.
        """
        #print('input shape: ', x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout(self.conv3(x))
        x = self.dropout(self.conv4(x))
        x = self.dropout(self.conv5(x))
        y = self.fc(x)
        return y
    
if __name__ == "__main__":
    # Ejemplo de uso
    # Definimos el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    Este modelo consta de `cinco` capas convolucionales, con 32, 64, 128, 256 y 512 kernels respectivamente. Se agrega mayor complejidad respecto a los modelos anteriores para mejorar la capacidad de la red.

    1. **Capas Convolucionales**:
        - Idem CNN Simple y CNN

    2. **Dropout**:
        - Se introduce regularización mediante Dropout en las capas convolucionales (`Dropout2d`) y fully connected para reducir el riesgo de sobreajuste.

    3. **Capa Fully Connected**:
        - Después de las capas convolucionales, se utiliza una capa Flatten para convertir el tensor multidimensional en un vector plano:
        ```
        (batch_size, 512 * 3 * 3) → (batch_size, 4608)
        ```
        - **512**: Número de kernels en la última capa convolucional.
        - **3**: Dimensión espacial resultante tras aplicar MaxPooling cinco veces (100 / 2^5 ≈ 3).
        - Este vector pasa por dos capas lineales con 512 y 256 neuronas, respectivamente, con funciones de activación ReLU y Dropout.
        - Finalmente, se aplica otra capa lineal que proyecta al espacio de clasificación de 7 emociones.

    4. **Weight Decay**:
        - Se utiliza `weight_decay` en el optimizador para agregar regularización adicional y evitar el sobreajuste.

    Este modelo es el más complejo de los desarrollados, integrando distintas regularizaciones y mayor profundidad para mejorar el rendimiento en la clasificación.
    """
    
    # Creamos el modelo          
    model_final = CNN()