"""
Lógica del pipeline primary
"""

from typing import Any, Dict, List, Tuple

import polars as pl
from transformers import BertTokenizer

from src.natural_language_processing.utils import Utils
logger = Utils.setup_logging()

class PipelinePrimary:

    # 1. Recategorizar columnas
    @staticmethod
    def recategorize_players_pl(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Recategoriza los jugadores en un DataFrame agrupándolos por conversation_id.

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de polars que contiene los datos a recategorizar.
        params: Dict[str, Any]
            Diccionario de parámetros primary.

        Returns
        -------
        polars.DataFrame
            DataFrame con los jugadores recategorizados.
        """
        # Parámetros
        players_col = params['players_col']
        conversation_id_col = params['conversation_col']

        # Función para combinar jugadores por conversation_id
        def combine_players(players):
            return '-'.join(players.drop_nulls().unique())

        logger.info("Iniciando la recategorización de jugadores...")

        # Recategorizar la columna players
        combined_players = df.group_by(conversation_id_col).agg(
            pl.col(players_col).map_elements(combine_players, return_dtype=pl.Utf8).alias(players_col)
        )

        logger.info("Jugadores combinados por conversation_id.")

        # Unir el DataFrame original con el DataFrame recategorizado
        df_recategorized = df.join(combined_players, on=conversation_id_col, how="left")
        logger.info("DataFrame original unido con el DataFrame recategorizado.")

        # Eliminar la columna original
        df_recategorized = df_recategorized.with_columns(
            pl.col(players_col + "_right").alias(players_col)
        ).drop(players_col + "_right")

        logger.info("Columna original eliminada y sustituida por la recategorizada!")

        return df_recategorized
    
    # 2. Binarizar columnas
    @staticmethod
    def binary_encode_pl(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Binariza las columnas especificadas en un DataFrame de Polars según un valor verdadero específico.
        Para la segunda columna especificada, elimina los campos que son 'NOANNOTATION', los pasa a booleano,
        y los convierte en 1 (true), 0 (false), y 2 (null).

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de polars que contiene los datos a binarizar.
            params: Dict[str, Any]
                Diccionario de parámetros primary.

        Returns
        -------
        polars.DataFrame
            DataFrame con las columnas especificadas binarizadas.
        """
        # Obtención de los parámetros desde params
        value1 = params['value1']
        value2 = params['value2']

        logger.info("Iniciando la binarización de columnas...")
        
        # Procesar value1
        logger.info(f"Procesando columna: '{value1}'")
        df = df.with_columns(
            pl.when(pl.col(value1) == True)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .cast(pl.Int64)
            .alias(value1)
        )
        logger.info(f"Columna '{value1}' binarizada.")

        # Procesar value2
        logger.info(f"Procesando columna especial: '{value2}'")
        # Primero, eliminar los campos 'NOANNOTATION' y reemplazar por 'full'
        df = df.with_columns(
            pl.when(pl.col(value2) == 'NOANNOTATION')
            .then(pl.lit('full'))
            .otherwise(pl.col(value2))
            .alias(value2)
        )
        # Luego, convertir a 1 (true), 0 (false), 2 (null)
        df = df.with_columns(
            pl.when(pl.col(value2) == 'true')
            .then(pl.lit(1))
            .when(pl.col(value2) == 'false')
            .then(pl.lit(0))
            .otherwise(pl.lit(2))
            .cast(pl.Int64)
            .alias(value2)
        )
        logger.info(f"Columna '{value2}' binarizada.")

        logger.info("Binarización de columnas completada.")

        return df

    # 3. Tokenizar columnas
    @staticmethod
    def tokenize_messages_pl(df: pl.DataFrame, params: Dict[str, Any]) -> pl.DataFrame:
        """
        Tokeniza los mensajes en un DataFrame de polars utilizando BERT y agrega los campos tokenizados al DataFrame original.

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de polars que contiene los datos a tokenizar.
        params: Dict[str, Any]
            Diccionario de parámetros primary.

        Returns
        -------
        polars.DataFrame
            DataFrame original con las columnas tokenizadas agregadas.
        """
        # Obtener el nombre de la columna a tokenizar desde params
        messages_col = params['messages_col']

        # Inicializar el tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Función para tokenizar un mensaje
        def tokenize_message(message: str) -> Tuple[List[int], List[int]]:
            tokens = tokenizer.encode_plus(
                message,
                add_special_tokens=True,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            return tokens['input_ids'][0].tolist(), tokens['attention_mask'][0].tolist()

        logger.info("Iniciando la tokenización de mensajes...")

        # Aplicar la tokenización a la columna de mensajes
        tokenized_data = df.select([pl.col(messages_col)]).to_series().apply(tokenize_message, return_dtype=pl.Object).to_list()

        # Extraer input_ids y attention_masks
        input_ids, attention_masks = zip(*tokenized_data)

        # Agregar las columnas tokenizadas al DataFrame original
        df = df.with_columns([
            pl.Series(name="input_ids", values=list(input_ids)),
            pl.Series(name="attention_masks", values=list(attention_masks))
        ])

        logger.info("Columnas tokenizadas agregadas al DataFrame.")

        return df, tokenizer


