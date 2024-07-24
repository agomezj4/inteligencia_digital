"""
Lógica del pipeline feature
"""

from typing import Any, Dict, List, Tuple

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

from src.natural_language_processing.utils import Utils
logger = Utils.setup_logging()

class PipelineModelInput:

    #1. Normalización de datos
    @staticmethod
    def min_max_scaler_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Estandariza las columnas numéricas (excluyendo binarias) de un DataFrame utilizando el método Min-Max Scaler.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas que se estandarizará.

        Returns
        -------
        pd.DataFrame
            DataFrame estandarizado.
        """
        logger.info("Iniciando la estandarización con Min-Max Scaler...")

        # Identificar las columnas numéricas
        numeric_cols = df.select_dtypes(include=['float32', 'float64', 'int32', 'int64']).columns

        # Filtrar solo las columnas numéricas no binarias (excluyendo aquellas que solo toman valores 0 y 1)
        numeric_cols = [col for col in numeric_cols if not ((df[col].nunique() == 2) & (df[col].isin([0, 1]).sum() == len(df)))]

        # Crear una copia del DataFrame para evitar el SettingWithCopyWarning
        df_copy = df.copy()

        # Aplicar Min-Max Scaler solo a las columnas numéricas no binarias
        for col in numeric_cols:
            min_val = df_copy[col].min()
            max_val = df_copy[col].max()
            range_val = max_val - min_val
            if range_val != 0:  # Evita la división por cero en caso de que todas las entradas en una columna sean iguales
                df_copy[col] = df_copy[col].astype('float64')  # Convertir la columna a float64
                df_copy.loc[:, col] = (df_copy[col] - min_val) / range_val

        logger.info("Estandarización con Min-Max Scaler completada!")

        return df_copy
    
    # #2. Balanceo de clases
    # def balance_target_variable_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    #     """
    #     Balancea la variable objetivo utilizando el método Synthetic Minority Over-sampling Technique (SMOTE).

    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         DataFrame de Pandas que se balanceará la target.
    #     params: Dict[str, Any]
    #         Diccionario de parámetros model input.

    #     Returns
    #     -------
    #     pd.DataFrame
    #         DataFrame con target balanceada balanceado.
    #     """
    #     logger.info("Iniciando el balanceo de la variable objetivo con SMOTE...")

    #     # Parámetros
    #     target = params['target_col']
    #     random_state = params['balance_target_variable']['random_state']
    #     sampling_strategy = params['balance_target_variable']['sampling_strategy']

    #     # Separar las características y la variable objetivo
    #     X = df.drop(columns=[target])
    #     y = df[target]

    #     # Guardar las listas de input_ids y attention_masks
    #     input_ids = np.stack(X['input_ids'].values)
    #     attention_masks = np.stack(X['attention_masks'].values)

    #     # Eliminar las columnas de listas temporales para aplicar SMOTE
    #     X = X.drop(columns=['input_ids', 'attention_masks'])

    #     # Contar las clases antes del balanceo
    #     counts_before = y.value_counts()
    #     logger.info(f"Conteo de clases antes del balanceo: {counts_before}")

    #     # Inicializar el objeto SMOTE
    #     smote = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy)

    #     # Aplicar SMOTE solo a las características numéricas/categóricas
    #     X_resampled, y_resampled = smote.fit_resample(X, y)

    #     # Recuperar los índices originales de las muestras sobresalidas
    #     resampled_indices = smote.sample_indices_

    #     # Añadir de nuevo las listas de input_ids y attention_masks
    #     input_ids_resampled = np.vstack([input_ids] * smote.ratio_[1])
    #     attention_masks_resampled = np.vstack([attention_masks] * smote.ratio_[1])

    #     # Crear un nuevo DataFrame con las características y la variable objetivo balanceada
    #     df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    #     df_resampled['input_ids'] = list(input_ids_resampled)
    #     df_resampled['attention_masks'] = list(attention_masks_resampled)
    #     df_resampled[target] = y_resampled

    #     # Contar las clases después del balanceo
    #     counts_after = y_resampled.value_counts()
    #     logger.info(f"Conteo de clases después del balanceo: {counts_after}")

    #     logger.info("Balanceo de la variable objetivo con SMote completado!")

    #     return df_resampled