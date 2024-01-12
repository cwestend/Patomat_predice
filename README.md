# Patomat_predice
Análisis del modelo de Yu et al. "Accurate recognition of colorectal cancer with semi-supervised deep learning on pathological images"
(https://www.nature.com/articles/s41467-021-26643-8) usado en predicciones de Patomat.

El modelo es el provisto por los autores (https://zenodo.org/records/5524324#.YU09Ny-KFLY), entrenado con los datos que ellos proveen en 
https://figshare.com/articles/dataset/Colorectal_cancer_datasets_of_semi-supervised_deep_learning/15072546/1 (CRC.zip).

Se pretende demostrar la predicción sobre datos independientes del Dataset-PAT (NCT-CRC-HE-100K-NONORM) del "NCT biobank and the UMM pathology archive: NCT-UMM, National Center for Tumor diseases, University Medical Center Mannheim, Heidelberg University, Germany" https://zenodo.org/records/1214456#.XV2cJeg3lhF.

## Entorno
Para instalar el entorno con las versiones que usamos en Patomat, lo más sencillo es proceder de la forma:
```
% conda create -n patomat_212_310 python=3.10.12
% conda install -c conda-forge numpy scikit-learn opencv tifffile imagecodecs imutils imageio pandas ipython
% conda install -c conda-forge cudatoolkit=11.8 cudnn
% pip install tensorflow==2.12.*
% pip install jupyter
```
Esto provee las versiones:
```
numpy-1.22.3
scikit-learn-1.0.2
opencv-4.6.0
tifffile-2023.2.28
imagecodecs-2023.1.23
imutils-0.5.4
imageio-2.33.1
pandas-2.1.4
ipython-8.20.0
cudatoolkit-11.8.0, cudnn-8.9.2.26
tensorflow-2.12.1
keras-2.12.0 
jupyterlab-4.0.10
notebook-7.0.6 
```

## Modelo

El modelo de Yu et al. junto con el inception de Google están en el directorio modelos/Yu_meancher_model/

### Pesos

Los pesos son demasiado grandes para github. Se deben descargar de https://cloud.iac.es/index.php/s/feaDGetcSaBwTb3 y poner el en directorio arriba
indicado (modelos/Yu_meancher_model/)

## Datos de prueba

Se han descargado algunas imágenes de patches tumorales (TUM) y no-tumorales o normales (NORM) del archivo NCT-CRC-HE-100K-NONORM (SIN normalizar).
Hay 100.000 imágenes de 224x224 pixeles que deben ser reescaladas y normalizadas para que el modelo pueda procesarlas y predecir si son positivas o no.

Las predicciones son un vector de 4 valores (2 para el teacher y 2 para el student). Tomamos el primero y el criterio elegido es si el segundo valor es cercano a 1 será 
una detección positiva (cáncer o tumoral) y si es cero será negativa (no-cáncer o normal). El criterio elegido es si ese segundo valor es mayor que 0.5 o inferior a 0.5.

NOTA: Al ser datos con diferentes tintes y de zonas no conocidas (no se tiene el porta entero) algunas detecciones pueden fallar. En general si aciertan con el etiquetado, pero no es el 100%.

## Notebook para predecir

En el sguiente notebook está el código para demostrar las predicciones sobre los datos de prueba:
https://github.com/cwestend/Patomat_predice/blob/main/predict.ipynb
 

