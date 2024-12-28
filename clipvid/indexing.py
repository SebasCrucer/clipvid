import os
import json
import cv2
import numpy as np
import torch
import clip
from PIL import Image
from patchify import patchify
from tqdm import tqdm
import logging
import sys
import time
import h5py
import concurrent.futures

def setup_logging(log_file='clip_index.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Manejador de archivo
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Manejador de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

setup_logging()

def video_duration(path):
    try:
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        video.release()
        duration = frame_count / fps if fps > 0 else 0
        logging.debug(f"Duración de {path}: {duration} segundos")
        return duration
    except Exception as e:
        logging.error(f"Error leyendo frames de {path}: {e}")
        return 0

def video_frames(path, max_frames=10000, log_interval=100):
    try:
        video = cv2.VideoCapture(path)
        if not video.isOpened():
            raise ValueError(f"No se pudo abrir el video: {path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        count = 0

        while count < max_frames:
            ret, frame = video.read()
            if not ret:
                logging.warning(f"Frame {count+1} no pudo ser leído en {path}. Fin del video.")
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logging.error(f"Error convirtiendo frame {count+1} en {path}: {e}")
                continue

            count += 1
            yield frame_rgb, count / fps

            if count % log_interval == 0:
                logging.debug(f"{path}: Procesados {count} frames.")

        video.release()

        if count >= max_frames:
            logging.warning(f"Se alcanzó el máximo de frames ({max_frames}) para {path}.")

    except Exception as e:
        logging.error(f"Error obteniendo frames de {path}: {e}")
        return

class ClipVidIndexer:
    def __init__(self, patch_size=224, batch_size=128, max_frames_per_video=10000):
        self.patch_size = patch_size
        self.patch_shape = (self.patch_size, self.patch_size, 3)
        self.patch_step = self.patch_size // 2
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info("Cargando modelo CLIP...")
        try:
            self.clip_model, self.clip_preprocess = clip.load('ViT-B/32', self.device, jit=False)
            self.clip_model.eval()
            logging.info("Modelo CLIP cargado exitosamente.")
        except Exception as e:
            logging.critical(f"Error cargando el modelo CLIP: {e}")
            raise e

        self.index_features = []
        self.index_metadata = []
        self.batch_size = batch_size
        self.max_frames_per_video = max_frames_per_video

    def index_all_videos(self, videos_folder, freq=1.0, video_timeout=300):
        """Indexa todos los videos que se encuentren en la carpeta `videos_folder`."""
        try:
            video_files = [
                f for f in os.listdir(videos_folder)
                if f.lower().endswith(('.mp4', '.avi', '.mov'))
            ]
            if not video_files:
                logging.warning(f"No se encontraron videos en la carpeta: {videos_folder}")
                return

            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_video = {
                    executor.submit(self.index_video, os.path.join(videos_folder, f), freq): f
                    for f in video_files
                }
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_video),
                    total=len(future_to_video),
                    desc="Indexando videos con CLIP",
                    unit="video"
                ):
                    video = future_to_video[future]
                    try:
                        future.result(timeout=video_timeout)
                        logging.info(f"Indexación completada para: {video}")
                    except concurrent.futures.TimeoutError:
                        logging.error(f"Timeout alcanzado para indexar el video: {video}")
                    except Exception as e:
                        logging.error(f"Error indexando {video}: {e}")
        except Exception as e:
            logging.critical(f"Error al listar los archivos en {videos_folder}: {e}")

    def index_video(self, path, freq=1.0, max_retries=3):
        """Indexa un único video en base a un `freq` (segundos por frame)."""
        last_index = 0
        duration = video_duration(path)
        if duration == 0:
            logging.warning(f"Duración de video inválida para {path}. Se omite indexación.")
            return
        video_name = os.path.splitext(os.path.basename(path))[0]
        logging.info(f'Indexando: {path}')

        with tqdm(total=duration, desc=f"Procesando {video_name}", unit="frames") as pbar:
            patches_batch = []
            metadata_batch = []
            frame_number = 0

            for frame, timestamp in video_frames(path, max_frames=self.max_frames_per_video):
                frame_number += 1
                if frame is None or frame.size == 0:
                    logging.warning(f"Frame vacío/ inválido en {path} (timestamp {timestamp}).")
                    continue

                if timestamp - last_index >= freq:
                    last_index = timestamp
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Dividir el frame en parches
                            patches = patchify(frame, self.patch_shape, step=self.patch_step)
                            patches = patches.reshape(-1, *self.patch_shape)

                            if patches.size == 0:
                                logging.warning(
                                    f"No se obtuvieron parches para el frame en {path} (t={timestamp})"
                                )
                                break

                            for p in patches:
                                pil_image = Image.fromarray(p)  # p ya está en RGB
                                preprocessed = self.clip_preprocess(pil_image)
                                metadata_batch.append({'video': video_name, 't': timestamp})
                                patches_batch.append(preprocessed)
                            break  # Salir del bucle de reintento si todo salió bien

                        except Exception as e:
                            retries += 1
                            logging.error(
                                f"Error dividiendo frame en parches para {path} (t={timestamp}). "
                                f"Reintento {retries}/{max_retries}: {e}"
                            )
                            time.sleep(1)

                    # Procesar lote si alcanzamos el batch_size
                    if len(patches_batch) >= self.batch_size:
                        try:
                            self._process_batch(patches_batch, metadata_batch)
                            patches_batch = []
                            metadata_batch = []
                            pbar.update(timestamp - pbar.n)
                        except Exception as e:
                            logging.error(
                                f"Error procesando lote en {path} (t={timestamp}): {e}"
                            )
                            patches_batch = []
                            metadata_batch = []
                            pbar.update(timestamp - pbar.n)

            # Procesar cualquier lote restante
            if patches_batch:
                try:
                    self._process_batch(patches_batch, metadata_batch)
                except Exception as e:
                    logging.error(f"Error procesando último lote en {path}: {e}")

            pbar.n = pbar.total
            pbar.close()
            logging.info(f"Video {path} procesado con {frame_number} frames.")

    def _process_batch(self, patches_batch, metadata_batch):
        """Procesa un lote de parches con CLIP."""
        if not patches_batch:
            logging.debug("Lote vacío, nada que procesar.")
            return

        try:
            tensor = torch.stack(patches_batch, dim=0).to(self.device)
        except Exception as e:
            logging.error(f"Error apilando parches en tensor: {e}")
            return

        with torch.no_grad():
            try:
                frame_features = self.clip_model.encode_image(tensor)
            except Exception as e:
                logging.error(f"Error codificando imágenes con CLIP: {e}")
                return

        self.index_features.append(frame_features.cpu())
        self.index_metadata.extend(metadata_batch)

    def save_embeddings(self, embeddings_folder):
        """Guarda los embeddings en un archivo .h5 y los metadatos en un .json."""
        try:
            os.makedirs(embeddings_folder, exist_ok=True)
            if not self.index_features:
                logging.warning("No hay features para guardar.")
                return

            index_features = torch.cat(self.index_features, dim=0).numpy().astype(np.float16)
            index_metadata = self.index_metadata

            # Guardar en HDF5
            embeddings_path = os.path.join(embeddings_folder, 'embeddings.h5')
            with h5py.File(embeddings_path, 'w') as hf:
                hf.create_dataset('features', data=index_features, compression="gzip")
            logging.info(f"Features guardados en {embeddings_path}")

            # Guardar metadatos en JSON
            metadata_path = os.path.join(embeddings_folder, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(index_metadata, f)
            logging.info(f"Metadatos guardados en {metadata_path}")

            print(f"Embeddings guardados en: {embeddings_path} y {metadata_path}")

        except Exception as e:
            logging.error(f"Error guardando embeddings: {e}")
