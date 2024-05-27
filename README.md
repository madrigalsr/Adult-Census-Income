# Paper repositorio Análisis sobre los ingresos de la población adulta

<font size=4>
    
Paper Análisis sobre los ingresos de la población adulta

Costa Rica, mayo 30 , 2024.

Randall Madrigal Salas
</font>

https://github.com/samuelsaldanav/paper1


## Sobre Análisis sobre los ingresos de la población adulta

Este análisis responde al proyecto final sobre Machine Learning para la especialidad en Ciencias de datos. 

<br>

- Nota: se asume el uso de Python 3.

<br>

<p align="justify">
Este repositorio está basado en _Notebooks Jupyter_ (_Jupyterlab_), así que puede seguirse a través de un navegador, en una instalación local (previa descarga), o en una plataforma en la nube. Se dan más detalles [abajo](#seguir_paper1).
</p>

<br>

### Tema

### Objetivos generales
<p align="justify">
Crear un modelo predictivo basado en técnicas de aprendizaje no supervisado y supervisado para segmentar clientes, identificar los factores determinantes en los ingresos y analizar la situación socioeconómica de la población adulta, con el fin de 
personalizar estrategias de marketing, mejorar los ingresos de la población objetivo y ofrecer información relevante para la toma de decisiones estratégicas.
</p>

<br>

### Objetivos específicos


#### Segmentación de clientes

*   Personalizar estrategias de marketing y servicios basado en la segmentación demográfica y económicamente de los individuos. 
*   Implementar un modelo de segmentación de clientes, utilizando técnicas de aprendizaje no supervisado como el análisis de clústeres basado en características demográficas y económicas. Este modelo permitirá identificar grupos de individuos con características similares, lo que facilitará la personalización de estrategias de marketing y servicios.

#### Identificación de factores influyentes en los ingresos

*   Determinar qué factores son más relevantes para alcanzar un nivel de ingresos más alto y así plantear estrategias para mejorar los ingresos de la población objetivo.
*   Aplicar técnicas de análisis de importancia de características para determinar qué factores tienen mayor influencia en los ingresos de los individuos. Esto ayudará a identificar variables críticas que contribuyen a alcanzar un nivel de ingresos más alto y a diseñar estrategias específicas para mejorar los ingresos de la población objetivo.

#### Análisis de la dinámica socioeconómica de la población

*   Identificar factores que puedan incidir en la sociedad en términos de aspectos sociales y económicos.
*   Emplear técnicas de análisis exploratorio de datos para identificar patrones y relaciones entre variables socioeconómicas. Esto permitirá comprender mejor la dinámica de la población en términos de aspectos sociales y económicos, 
proporcionando información valiosa para la toma de decisiones estratégicas.

<br>

### Contenidos 
Los temas que se abordan en este análisis son:

1.	Introducción y Generalidades
2.	Antecedentes
3.	Metodología
4.	Materiales
5.	Resultados
6.	Recomendaciones
7.	Conclusiones
8.	Resumen
9.	Bibliografía

<br>

### Introducción
<p align="justify">
En el ámbito de la ciencia de datos, la capacidad de predecir ingresos basados en características demográficas y socioeconómicas es crucial para el desarrollo de estrategias de mercado y políticas públicas. Este proyecto se centra en la creación de un modelo predictivo sobre los ingresos de la población adulta utilizando técnicas de aprendizaje automático. El análisis se fundamenta en datos del censo de ingresos del Repositorio de Machine Learning de UC Irvine, recolectados en 1994.
</p>

## Generalidades
<p align="justify">
Este proyecto se implementa siguiendo la metodología CRISP-DM, que estructura el proceso en fases definidas: comprensión del negocio, comprensión de los datos, preparación de los datos, modelado, evaluación y despliegue. A través de esta metodología, se busca no solo mejorar la precisión de los modelos predictivos, sino también obtener insights valiosos sobre los factores determinantes de los ingresos y cómo estos pueden influir en la toma de decisiones estratégicas para el bienestar socioeconómico de la población. El dataset original, extraído por Barry Becker, se compone de 32,561 registros con atributos que incluyen edad, clase de trabajo, nivel educativo, estado civil, ocupación, raza, sexo, ganancias y pérdidas de capital, horas trabajadas por semana y país de origen.
</p>

## Antecedentes
<p align="justify">
El proyecto se desarrolla utilizando la metodología CRISP-DM (Cross Industry Standard Process for Data Mining), que incluye las fases de comprensión del negocio, comprensión de los datos, preparación de los datos, modelado, evaluación y despliegue. El dataset original, extraído por Barry Becker, se compone de 32,561 registros con atributos que incluyen edad, clase de trabajo, nivel educativo, estado civil, ocupación, raza, sexo, ganancias y pérdidas de capital, horas trabajadas por semana y país de origen.
</p>

## Metodología
La metodología adoptada comprende las siguientes etapas:

1.   Comprensión del Negocio: Identificación de los objetivos del proyecto y definición de las preguntas clave a responder.
2.   Comprensión de los Datos: Análisis exploratorio para identificar patrones y anomalías.
3.   Preparación de los Datos: Limpieza de datos, tratamiento de valores nulos y reagrupamiento de categorías.
4.   Modelado: Entrenamiento de varios modelos de aprendizaje supervisado (AdaBoost, RandomForest, LogisticRegression y DecisionTree) y no supervisado (K-means para segmentación).
5.   Evaluación: Validación de los modelos mediante técnicas como GridSearchCV y RFE para ajuste de hiperparámetros y selección de características.
6.   Despliegue: Implementación del modelo final para su uso práctico.

## Materiales
Los materiales utilizados en el proyecto incluyen:

*   Dataset: "adult.csv" del UCI Machine Learning Repository.
*   Herramientas de Software: Python, bibliotecas de Machine Learning como scikit-learn y pandas para manipulación y análisis de datos.
*   Algoritmos: AdaBoost, RandomForest, LogisticRegression, 
*   DecisionTree PCA y K-means.

## Resultados
<p align="justify">
Los resultados iniciales mostraron que los modelos AdaBoost y RandomForest ofrecían mejores precisiones iniciales (86% y 84% respectivamente), en comparación con LogisticRegression y DecisionTree. Tras ajustar los hiperparámetros y reducir las características, se observó una ligera mejora en los modelos AdaBoost y RandomForest, mientras que LogisticRegression y DecisionTree mostraron una pequeña disminución en precisión.

Posteriormente, se utilizó SMOTE para balancear los datos, creando un conjunto de datos con 50 mil muestras. Los modelos fueron reentrenados y evaluados, resultando en mejoras significativas en todos los modelos, especialmente en LogisticRegression, que alcanzó una precisión del 100%.
</p>

## Recomendaciones

1.   Segmentación de Clientes:
  *   Focalizar campañas de marketing en individuos con altos niveles educativos y casados, utilizando técnicas de PCA y dispersión de datos.
2.   Mejora de Ingresos:
  *   Implementar políticas laborales y programas de formación que mejoren las habilidades y horas trabajadas.
  *   Incentivar el ahorro y la inversión.
3.   Inclusión y Equidad:
  *   Desarrollar políticas específicas para mujeres, minorías raciales y personas de diversos orígenes nacionales.

## Conclusiones
<p align="justify">
El proyecto demostró la efectividad de los modelos de aprendizaje automático en la predicción de ingresos, destacando la importancia de las características demográficas y financieras. El uso de SMOTE para balancear los datos fue crucial para mejorar significativamente la precisión de los modelos. Las estrategias recomendadas incluyen enfoques personalizados de marketing y políticas que promuevan la equidad y el bienestar socioeconómico.
</p>

## Resumen
<p align="justify">
Este proyecto de análisis de datos y modelado predictivo proporciona una comprensión profunda de los factores que influyen en los ingresos de la población adulta. Utilizando técnicas avanzadas de aprendizaje automático y métodos de balanceo de datos, se logró mejorar la precisión de los modelos, ofreciendo valiosas recomendaciones para políticas y estrategias de marketing más efectivas.
</p>

## Bibliografía


1.   UCI Machine Learning Repository: Adult Data Set. https://archive.ics.uci.edu/dataset/2/adult
2.   Vallalta Rueda, J. F. "CRISP-DM: una metodología para minería de datos en salud". Disponible en: https://healthdataminer.com/data-mining/crisp-dm-una-metodologia-para-mineria-de-datos-en-salud/.
3.   Aprende IA. "Algoritmo KMeans - Teoría". Disponible en: https://aprendeia.com/algoritmo-kmeans-clustering-machine-learning.
4.   Kaggle. "Introducción al clustering con Python y sklearn". Disponible en: https://www.kaggle.com/code/micheldc55/introduccion-al-clustering-con-python-y-sklearn.
5.   Zapata, J. R. "Aprendizaje automático con Scikit Learn". Disponible en: https://joserzapata.github.io/courses/python-ciencia-datos/ml/.
6.   DataScientest. "¿ Cómo aprovechar el rendimiento de la matriz de confusión". Disponible en: https://datascientest.com/es/matriz-de-confusion#:~:text=La%20matriz%20de%20confusi%C3%B3n%20indica,valor%20real%20es%20igualmente%20positiva.
7.   Scikit-learn. "Cross validation: evaluating estimator performance". Disponible en: https://scikit-learn.org/stable/modules/cross_validation.html.
8.   Brownlee, J. "Recursive feature elimination (RFE) para la selección de características en Python". Machine Learning Mastery. Disponible en: https://machinelearningmastery.com/rfe-feature-selection-in-python/.
9.   Brownlee, J. "Tune Hyperparameters for Classification Machine Learning Algorithms". Machine Learning Mastery. Disponible en: https://machinelearningmastery.com/hyperparameters-for-classification-machine-learning-algorithms/.
10.   Brownlee, J. "SMOTE for Imbalanced Classification with Python". Machine Learning Mastery. Disponible en: https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/.

