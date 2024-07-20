"""
Lógica del pipeline intermediate
"""

from typing import Any, Dict

import polars as pl

from src.natural_language_processing.utils import Utils
logger = Utils.setup_logging()


class PipelineIntermediate:

    # 1. Diferenciar mensajes de una misma conversación
    @staticmethod
    def expand_messages_pl(
            df: pl.DataFrame,
            params: Dict[str, Any],
    ) -> pl.DataFrame:
        """
        Expande las listas en la columna especificada en registros individuales y replica la lógica a las demás columnas.

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de Polars con una columna que contiene listas.
        params: Dict[str, Any]
            Diccionario de parámetros intermediate.

        Returns
        -------
        pl.DataFrame: DataFrame con los mensajes expandidos y columnas replicadas.
        """
        # Registra un mensaje de información indicando el inicio del proceso de expansión
        logger.info("Inicia el procesamiento inicial...")

        # Parámetros
        message_col = params["message_col"]
        name_new_col = params["name_new_col"]

        # Lista para almacenar los nuevos registros
        new_records = []

        # Iterar sobre cada fila del DataFrame
        for i, row in enumerate(df.rows()):
            # Asumimos que la columna 'message_col' tiene la lista de referencia
            num_elements = len(row[df.columns.index(message_col)])
            for j in range(num_elements):
                new_record = {}
                # Agregar cada columna con su valor correspondiente
                for col_name in df.columns:
                    col_index = df.columns.index(col_name)
                    if isinstance(row[col_index], list):
                        # Agregar el valor de la lista si el índice es válido
                        if j < len(row[col_index]):
                            new_record[col_name] = row[col_index][j]
                        else:
                            # Manejar el caso en el que la lista es más corta
                            new_record[col_name] = None
                    else:
                        # Replicar el valor para columnas que no son listas
                        new_record[col_name] = row[col_index]
                # Agregar columna 'conversation_id' para rastrear el registro original
                new_record[name_new_col] = i
                new_records.append(new_record)

        # Crear nuevo DataFrame a partir de los registros generados
        new_df = pl.DataFrame(new_records)

        # Registra un mensaje de información indicando el fin del proceso de expansión
        logger.info("Finalizado procesamiento inicial!")

        # Retorna el DataFrame resultante
        return new_df

    # 2. Cambio de tipado de columnas
    @staticmethod
    def columns_to_int_pl(
        df: pl.DataFrame, 
        params: Dict[str, Any]
        ) -> pl.DataFrame:
        """
        Convierte una lista de columnas de un DataFrame de polars a tipo float64.

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de polars que contiene los datos a convertir.
        params: Dict[str, Any]
            Diccionario de parámetros intermediate.

        Returns
        -------
        polars.DataFrame
            DataFrame con las columnas especificadas convertidas a tipo int64.
        """
        # Obtención de los nombres de las columnas desde params
        columns = params['colums_to_int']

        logger.info("Iniciando la conversión de columnas a int64...")

        # Conversión de las columnas a float64
        for column in columns:
            df = df.with_columns(pl.col(column).cast(pl.Int64).alias(column))
            logger.info(f"Columna '{column}' convertida a int64.")

        logger.info("Conversión de columnas a int64 completada.")

        return df
