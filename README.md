# sia_tp4
TP4 - Sistemas de Inteligencia Artificial ITBA 2020
- Ejecución:

Colocarse en el directorio donde se encuentre el archivo requirements.txt dentro de una terminal e ingresar el comando 'pip install -r requirement.txt' para instalar las dependencias necesarias. Luego para ejecutar un script ingresar el comando 'py archivo.py', siendo "archivo" el nombre del script de Python Siendo nuestros archivos ejecutables: HopfieldNetworkTest.py, OjasRuleTest.py y SOM.py

Dentro de la carpeta data se encuentra el archívo config.json donde se ingresan ingresan los parametros de ejecución:
- ej1b: 
  - learn_factor: el factor de aprendizaje en la regla de Oja
  - max_epochs: Máximo de épocas

- ej2:
  - random_prob: Probabilidad de randomizar un elemento de un patron de prueba en la red de Hopfield
  - epsilon: Si el h del algoritmo de la red de Hopfield es menor que epsilon entonces se considera h = 0
  - max_epochs: Máximo de épocas
  - pattern_file: nombre del archivo de texto que contiene los patrones de entrenamiento/prueba.
  
- Ejemplo de configuración de ejecución:

{
  "ej1b": [
    {
      "learn_factor": "0.001",
      "max_epochs": "1000"

    }
  ],
  "ej2": [
    {
      "random_prob": "0.1",
      "epsilon": "0.00001",
      "max_epochs": "50",
      "patterns_file": "patterns1.txt"
    }
  ]

}
