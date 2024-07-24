"""
Lógica del pipeline feature
"""

from typing import Any, Dict, List, Tuple

from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import re

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from feature_engine.selection import DropConstantFeatures, SmartCorrelatedSelection

from src.natural_language_processing.utils import Utils
logger = Utils.setup_logging()

class PipelineFeature:

    #1. Creacion de features
    @staticmethod
    def features_new_pl_pd(
        df: pl.DataFrame,
        params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Calcula nuevas características para cada cliente.

        Parameters
        ----------
        df : polars.DataFrame
            DataFrame de polars que contiene las características del cliente.
        params: Dict[str, Any]
            Diccionario de parámetros feature.

        Returns
        -------
        pd.DataFrame: DataFrame con las nuevas columnas agregadas.
        """
        logger.info("Iniciando el cálculo de nuevas características ...")

        # Parámetros
        game = params['data_features']['game_id']
        conversation = params['data_features']['conversation_col']
        years = params['data_features']['years']
        messages = params['data_features']['messages_col']
        early_threshold = params['data_features']['early_threshold']
        mid_threshold = params['data_features']['mid_threshold']

        # Convertir DataFrame de polars a pandas
        df = df.to_pandas()

        # Leer parámetros desde el diccionario
        current_year = datetime.now().year

        # Convertir la columna 'years' a tipo numérico
        df[years] = pd.to_numeric(df[years], errors='coerce')

        # Crear la nueva columna 'game_conversation_id'
        df['game_conversation_id'] = (df[game].astype(str) + df[conversation].astype(str)).astype(int)

        # Crear variable 'years_since'
        df['years_since'] = (current_year - df[years]).astype(np.int64)

        # Crear variable 'message_length'
        df['message_length'] = df[messages].apply(len).astype(np.int64)

        # Crear variable 'game_stage'
        df['game_stage'] = pd.cut(df[years], bins=[-float('inf'), early_threshold, mid_threshold, float('inf')], labels=['early', 'mid', 'late']).astype(object)

        # Crear variable 'message_ratio'
        total_messages = df.groupby([game, conversation])[messages].transform('count')
        df['message_ratio'] = df[messages].groupby(df[game]).transform('count') / total_messages
        df['message_ratio'] = df['message_ratio'].astype(np.float64)

        # Crear variable 'conversation_length'
        df['conversation_length'] = df.groupby([game, conversation])[messages].transform('count').astype(np.int64)

        # Eliminar la columna original si es una buena práctica
        if params['drop_original'] is True:
            df.drop(messages, inplace=True, axis=1)
            logger.info(f"Columna '{messages}' eliminada del DataFrame.")

        logger.info("Finalizado el cálculo de nuevas características!")
        return df
    
    #2. Escalado de features
    @staticmethod
    def one_hot_encoding_pd(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        Aplica One Hot Encoding a las columnas especificadas en el diccionario de parámetros.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de Pandas al que se le aplicará One Hot Encoding
        params: Dict[str, Any]
            Diccionario de parámetros featuring

        Returns
        -------
        pd.DataFrame: DataFrame con las columnas transformadas.
        """
        logger.info("Iniciando One Hot Encoding...")

        # Parámetros
        tensor_cols = params["tensor_cols"]
        
        # Selección de columnas categóricas
        one_hot_encoder_columns = [nombre for nombre in df.columns if df[nombre].dtype == 'object']

        # Excluyen variables tensoriales
        one_hot_encoder_columns = [col for col in one_hot_encoder_columns if col not in tensor_cols]

        # Inicializamos el OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False, drop=None)
        encoded_columns = encoder.fit_transform(df[one_hot_encoder_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(one_hot_encoder_columns))

        # Eliminamos las columnas originales y unimos las nuevas columnas codificadas
        df = df.drop(columns=one_hot_encoder_columns).reset_index(drop=True)
        encoded_df = encoded_df.reset_index(drop=True)
        df = pd.concat([df, encoded_df], axis=1)

        logger.info("One Hot Encoding completado!")

        return df

    #3. Selección de features
    @staticmethod
    def add_random_variables_pd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega dos variables aleatorias al DataFrame

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame al que se agregarán las variables aleatorias

        Returns
        -------
        pd.DataFrame
            DataFrame con las variables aleatorias agregadas.
        """
        # Establecer la semilla fija en 42
        np.random.seed(42)

        # Agregar variables aleatorias
        df["var_aleatoria_uniforme"] = np.random.rand(len(df))
        df["var_aleatoria_entera"] = np.random.randint(1, 5, size=len(df))

        return df

    @staticmethod
    def feature_selection_pipeline_pd(
        df: pd.DataFrame, 
        params: Dict[str, Any]
        ) -> List[str]:
        """
        Ejecuta un pipeline de selección de características usando RandomForestClassifier y calcula la importancia de las características.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame que contiene el conjunto de datos a procesar.
        params: Dict[str, Any]
            Diccionario de parámetros feature.

        Returns
        -------
        List[str]
            Lista con los nombres de las características más relevantes seleccionadas por el modelo.
        """
        logger.info("Iniciando la selección de características ...")

        # Parámetros
        variance_threshold = params["variance_treshold"]
        corr_threshold = params["correlation_treshold"]
        target_col = params["target_col"]
        game_col = params["game_col"]
        game_conversation_col = params["game_conversation_col"]
        conversation_col = params["conversation_col"]
        tensor_cols = params["tensor_cols"]

        # Dejar como índice el id_column y date_column
        df.set_index([game_col, game_conversation_col, conversation_col], inplace=True)

        # Eliminar columnas tensoriales
        df_drop_tensor = df.drop(columns=tensor_cols).copy()

        # Dividir en X e y
        X = df_drop_tensor.drop(columns=[target_col])
        y = df_drop_tensor[target_col].values

        # Crear el pipeline de transformaciones
        pipeline = Pipeline(
            steps=[
                ("drop_constant_features", DropConstantFeatures(tol=variance_threshold, missing_values="include")),
                ("drop_correlated_features", SmartCorrelatedSelection(
                    method="pearson", threshold=corr_threshold, missing_values="raise", selection_method="model_performance",
                    estimator=DecisionTreeClassifier(random_state=0), scoring="roc_auc", cv=3)),
                ("model_selector", RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10, min_samples_leaf=20))
            ]
        )

        # Ejecutar el pipeline
        model = pipeline.fit(X, y)

        # Extraer información de características
        total_features = X.columns.tolist()
        constant_features = model.named_steps["drop_constant_features"].features_to_drop_
        correlated_features = model.named_steps["drop_correlated_features"].features_to_drop_
        features_selected = model.named_steps["model_selector"].feature_names_in_
        feature_importance = model.named_steps["model_selector"].feature_importances_

        # Compilar resultados en un DataFrame
        feature_importance_df = pd.DataFrame({"Feature": features_selected, "Importance": feature_importance})
        feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

        # Agregar las variables aleatorias al DataFrame
        X = PipelineFeature.add_random_variables_pd(X)

        # Recalcular la importancia de las características incluyendo las variables aleatorias
        model_with_random = RandomForestClassifier(random_state=0, n_estimators=100, max_depth=10, min_samples_leaf=20)
        model_with_random.fit(X, y)
        feature_importance_with_random = model_with_random.feature_importances_
        features_with_random = X.columns.tolist()

        # Compilar resultados en un DataFrame
        feature_importance_df_with_random = pd.DataFrame({"Feature": features_with_random, "Importance": feature_importance_with_random})
        feature_importance_df_with_random = feature_importance_df_with_random.sort_values(by="Importance", ascending=False)

        # Verificar la existencia de las variables aleatorias
        if "var_aleatoria_uniforme" not in feature_importance_df_with_random['Feature'].values:
            raise ValueError("La variable aleatoria 'var_aleatoria_uniforme' no se encuentra en el DataFrame de importancia de características.")
        if "var_aleatoria_entera" not in feature_importance_df_with_random['Feature'].values:
            raise ValueError("La variable aleatoria 'var_aleatoria_entera' no se encuentra en el DataFrame de importancia de características.")

        # Calcular la importancia de las variables aleatorias
        random_var_imp_0_1 = feature_importance_df_with_random.loc[feature_importance_df_with_random['Feature'] == 'var_aleatoria_uniforme', 'Importance'].values[0]
        random_var_imp_1_4 = feature_importance_df_with_random.loc[feature_importance_df_with_random['Feature'] == 'var_aleatoria_entera', 'Importance'].values[0]

        # Eliminar las variables con importancia menor que las variables aleatorias
        feature_importance_df_with_random = feature_importance_df_with_random[
            (feature_importance_df_with_random["Importance"] > random_var_imp_0_1) &
            (feature_importance_df_with_random["Importance"] > random_var_imp_1_4)
        ]

        # Imprimir la información sobre las variables eliminadas en cada paso
        logger.info("Selección de variables completada.")
        logger.info(f"Variables iniciales: {len(total_features)}")
        logger.info(f"Variables eliminadas por filtro de varianza: {len(constant_features)}")
        logger.info(f"Variables eliminadas por correlación: {len(correlated_features)}")
        logger.info(f"Variables finales: {len(feature_importance_df_with_random)}")

        features = feature_importance_df_with_random['Feature'].to_list()

        # Filtrar el DataFrame original con las características seleccionadas, las variables tensoriales y la variable objetivo
        features = features + tensor_cols + [target_col]
        df = df[features]

        return df