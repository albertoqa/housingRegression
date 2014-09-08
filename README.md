housingRegression
=================

##**Descripción del problema y base de datos analizada:**

* Boston Housing Data Set.
* Características de la base de datos
  - 506 patrones con 13 Atributos (Reales, categóricas y enteras) + Salida.
  - Predicción del valor medio de viviendas de Boston. 
* Metodologias usadas:
  - SVR (Support Vector Regressor) 
  - Decision Tree Regressor
  - Gradient Boosting Tree Regressor

**Ajuste de los modelos:**

* SVR
  - Kernel (linear, rbf, poly)
  - Ajuste de parámetros: C y épsilon
* Decision Tree
  - Número de características
  - Profundidad del árbol
  - Número de ejemplos para dividir un nodo 
  - Numero de ejemplos para que un nodo sea hoja
* Gradient Boosting
  - Función de pérdida (ls, lad, huber, queantile)
  - Ratio de aprendizaje
  - Número de árboles a generar
  - Profundidad de cada árbol
  - Porcentaje de elementos para entrenar cada árbol

**Detalles de la metodología de entrenamiento:**

* Validación cruzada con 10 particiones de datos para cada modelo.
  - Calculo de métricas en media de las 10 particiones del error medio absoluto, error medio cuadrático y r cuadrado (coeficiente de correlación) para cada modelo.
  - Cálculo de la importancia de variables para los modelos Gradient Boosting y Decision Tree Regressor.
  - Entrenamiento selectivo con las características mas idóneas.
  - Dependencia de algunas características con respecto al valor medio de las viviendas.

**Comparación de las distintas técnicas:**

* Resultados obtenidos (media de las 10 iteraciones de la validación cruzada):
|             | Error Medio Absoluto | Error Medio Cuadrático | R cuadrado |
|-------------|---------|-----------|--------------------------------------|
| SVR (linear) | a | b | c |
| Decision Tree Regressor | a | b | c |
| Gradient Boosting Tree Regressor | a | b | c |



