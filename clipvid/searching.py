import os
import json
import torch
import clip
import h5py
from collections import defaultdict


class ClipVidSearcher:
    def __init__(self, embeddings_folder, videos_folder, device, batch_size=256):
        """
        :param embeddings_folder: Carpeta donde se encuentran 'embeddings.h5' y 'metadata.json'
        :param videos_folder: Carpeta donde se encuentran los videos
        :param device: 'cuda' o 'cpu'
        :param batch_size: Tamaño de lote para procesamiento de consultas
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device, jit=False)
        self.embeddings_folder = embeddings_folder
        self.videos_folder = videos_folder
        self.batch_size = batch_size

        # Estructura donde se guardan features y metadata
        self.embeddings = {}
        self.load_all_embeddings()

    def load_all_embeddings(self):
        """Carga embeddings y metadatos desde disco."""
        embeddings_path = os.path.join(self.embeddings_folder, 'embeddings.h5')
        metadata_path = os.path.join(self.embeddings_folder, 'metadata.json')

        with h5py.File(embeddings_path, 'r') as hf:
            features = torch.tensor(hf['features'][:]).to(self.device)

        with open(metadata_path, 'r') as f:
            index_metadata = json.load(f)

        assert len(features) == len(index_metadata), (
            f"Desincronización: {len(features)} embeddings vs {len(index_metadata)} metadatos."
        )

        self.embeddings = {
            'features': features,
            'metadata': index_metadata
        }

    def search(self, query, n=10):
        """
        Realiza una búsqueda basada en una consulta de texto y devuelve
        los videos ordenados por la similitud promedio de todos sus parches.

        Retorna una lista de tuplas: (nombre_del_video, score, ruta_completa)
        """
        # Tokenizar y obtener embeddings de texto
        query_tokens = clip.tokenize([query]).to(self.device)
        with torch.no_grad():
            query_features = self.clip_model.encode_text(query_tokens)

        # Normalizar embedding de texto
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)

        # Normalizar las features indexadas (si no se hizo antes)
        emb_features = self.embeddings['features']
        emb_features = emb_features / emb_features.norm(dim=-1, keepdim=True)

        # Alinear dtype
        query_features = query_features.to(emb_features.dtype)

        # Calcular similitudes (batch único, se puede vectorizar)
        similarities = (query_features @ emb_features.T).squeeze(0)

        # Recolectar similitudes por video
        meta_data = self.embeddings['metadata']
        video_sims_map = defaultdict(list)

        for sim, meta in zip(similarities, meta_data):
            video = meta['video']  # nombre del archivo (con o sin extensión)
            video_sims_map[video].append(sim.item())

        # Calcular score promedio por video
        scores = []
        for video, sims in video_sims_map.items():
            avg_score = sum(sims) / len(sims)
            # Construir ruta completa
            full_path = os.path.join(self.videos_folder, f"{video}.mp4")

            scores.append((video, avg_score, full_path))

        # Ordenar descendente por score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Retornar los primeros n resultados
        return scores[:n]
