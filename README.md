# ClipVid

**ClipVid** es un paquete que permite indexar videos usando el modelo CLIP para obtener representaciones de sus frames y realizar búsquedas basadas en texto. Esto resulta útil en aplicaciones como búsqueda de contenido visual, etiquetado automático de videos o sistemas de recomendación multimedia.

## Instalación

1. Clona este repositorio:
   ```bash
   git clone <url_del_repositorio>
   cd ClipVid
   ```

2. Instala las dependencias utilizando Poetry:
   ```bash
   poetry install
   ```

3. Si no usas Poetry, puedes instalar las dependencias directamente con pip:
   ```bash
   pip install -r requirements.txt
   ```

## Requisitos
- Python 3.8 o superior
- GPU compatible con PyTorch (opcional pero recomendado para rendimiento)

## Uso

### Ejemplo básico

El siguiente ejemplo muestra cómo indexar videos y realizar búsquedas con texto:

```python
import os
import torch
import clip
from clipvid import ClipVidIndexer, ClipVidSearcher

def main():
    # Configuración
    videos_folder = "path/to/your/videos"  # Ruta a la carpeta de videos
    embeddings_folder = os.path.join(videos_folder, "embeddings")
    os.makedirs(embeddings_folder, exist_ok=True)

    # 1. Indexar videos
    print("Iniciando el proceso de indexación...")
    indexer = ClipVidIndexer()
    indexer.index_all_videos(videos_folder, freq=1.0)  # Indexar todos los videos en la carpeta
    indexer.save_embeddings(embeddings_folder)
    print("Indexación completada. Embeddings guardados.")

    # 2. Realizar búsquedas
    searcher = ClipVidSearcher(embeddings_folder=embeddings_folder, device=device)

    # Consulta de ejemplo
    query = "a person walking on the beach"  # Cambia esto según lo que quieras buscar
    print(f"Realizando búsqueda con la consulta: '{query}'")
    results = searcher.search(query, n=5)

    # Mostrar resultados
    print("\nResultados de la búsqueda:")
    for video, score in results:
        print(f"Video: {video}, Score: {score:.4f}")

if __name__ == "__main__":
    main()
```

### Parámetros de las clases

#### `ClipVidIndexer`

La clase `ClipVidIndexer` se utiliza para procesar y generar embeddings de los videos.

**Parámetros del constructor:**
- **`patch_size`** *(int)*: Tamaño de los parches extraídos de cada frame del video. Predeterminado: `224`.
- **`batch_size`** *(int)*: Tamaño del lote para el procesamiento de parches. Predeterminado: `128`.
- **`max_frames_per_video`** *(int)*: Número máximo de frames que se procesarán por video. Predeterminado: `10000`.

**Métodos principales:**
- **`index_all_videos(videos_folder, freq=1.0, video_timeout=300)`**:
   - `videos_folder` *(str)*: Ruta a la carpeta que contiene los videos.
   - `freq` *(float)*: Frecuencia en segundos entre los frames indexados (ej., `1.0` para un frame por segundo).
   - `video_timeout` *(int)*: Tiempo máximo (en segundos) para procesar un video antes de interrumpir.

- **`index_video(path, freq=1.0)`**:
   - `path` *(str)*: Ruta al archivo de video.
   - `freq` *(float)*: Frecuencia en segundos entre los frames indexados.

- **`save_embeddings(embeddings_folder)`**:
   - `embeddings_folder` *(str)*: Ruta a la carpeta donde se guardarán los embeddings y metadatos.

#### `ClipVidSearcher`

La clase `ClipVidSearcher` se utiliza para buscar contenido visual basado en texto.

**Parámetros del constructor:**
- **`embeddings_folder`** *(str)*: Carpeta donde se encuentran los embeddings generados y los metadatos.
- **`device`** *(str)*: Dispositivo de procesamiento (`'cuda'` o `'cpu'`).
- **`batch_size`** *(int)*: Tamaño del lote para las consultas. Predeterminado: `256`.

**Métodos principales:**
- **`load_all_embeddings()`**: Carga los embeddings y metadatos desde la carpeta especificada.
- **`search(query, n=10)`**:
   - `query` *(str)*: Consulta de texto para buscar videos.
   - `n` *(int)*: Número de resultados a devolver. Predeterminado: `10`.


## Contribuciones
Si deseas contribuir al proyecto, crea un fork del repositorio, realiza tus cambios y envía un pull request.

## Licencia
Este proyecto está licenciado bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.

