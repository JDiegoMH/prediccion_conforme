# 01 · Dígitos: el primer algoritmo

**Qué muestra:** conjuntos conformes de clasificación con el puntaje `1 - p̂_y(x)` (LAC). El tamaño del conjunto crece con la dificultad de la imagen.

- **Datos:** `sklearn.datasets.load_digits`, con 1,797 imágenes de 8×8. No hay que descargar nada. A cada imagen se le asigna al azar un nivel de ruido gaussiano (0, 4 u 8). Como el mismo mecanismo se aplica a todas las imágenes, los datos siguen siendo intercambiables.
- **Diseño:** logística multinomial, α = 0.10 y 100 particiones de 600 / 500 / 697.
- **Resultado:** cobertura de 0.899 (EMC 0.002) y tamaño medio del conjunto de 1.40. El 66% de los conjuntos tiene una etiqueta, el 28% dos y el 6% tres o más.

```sh
python imagenes.py      # escribe en resultados/
python figuras.py       # imagenes_ejemplos.png, imagenes_algoritmo.png
```

**Teoría:** secciones "Un primer algoritmo" y "Volvamos a los dígitos" de la guía.
