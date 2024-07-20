from typing import Tuple, Dict

import os
import sys
import yaml
import pickle
import logging
import pandas as pd
import json
import requests
from io import StringIO
import polars as pl


class Utils:

    @staticmethod
    def setup_logging() -> logging.Logger:
        """
        Configura el logging para la aplicación.

        Returns
        -------
        logging.Logger
            El logger configurado para la aplicación.
        """
        import logging
        logging.basicConfig()
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def add_src_to_path() -> None:
        """
        Agrega la ruta del directorio 'src' al sys.path para facilitar las importaciones.

        Returns
        -------
        None
        """
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        sys.path.append(project_root)

    @staticmethod
    def get_project_root() -> str:
        """
        Obtiene la ruta raíz del proyecto.

        Returns
        -------
        str
            Ruta raíz del proyecto.
        """
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    @staticmethod
    def load_parameters(parameters_directory: str) -> Dict[str, dict]:
        """
        Carga los archivos de parámetros en formato YAML desde un directorio específico.

        Parameters
        ----------
        parameters_directory : str
            Directorio donde se encuentran los archivos YAML.

        Returns
        -------
        Dict[str, dict]
            Diccionario con los parámetros cargados.
        """
        yaml_files = [f for f in os.listdir(parameters_directory) if f.endswith('.yml')]
        parameters = {}
        for yaml_file in yaml_files:
            with open(os.path.join(parameters_directory, yaml_file), 'r') as file:
                data = yaml.safe_load(file)
                key_name = f'parameters_{yaml_file.replace(".yml", "")}'
                parameters[key_name] = data
        return parameters

    @staticmethod
    def load_csv_pd(file_path: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo CSV usando Pandas.

        Parameters
        ----------
        file_path : str
            Ruta del archivo CSV a cargar.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def load_csv_pl(file_path: str) -> pl.DataFrame:
        """
        Cargar datos desde un archivo CSV usando Polars.

        Parameters
        ----------
        file_path : str
            Ruta del archivo CSV a cargar.

        Returns
        -------
        pl.DataFrame
            DataFrame con los datos cargados.
        """
        return pl.read_csv(file_path)

    @staticmethod
    def load_parquet_pd(file_path: str) -> pd.DataFrame:
        """
        Cargar datos desde un archivo Parquet usando Pandas.

        Parameters
        ----------
        file_path : str
            Ruta del archivo Parquet a cargar.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """
        return pd.read_parquet(file_path)

    @staticmethod
    def load_parquet_pl(file_path: str) -> pl.DataFrame:
        """
        Cargar datos desde un archivo Parquet usando Polars.

        Parameters
        ----------
        file_path : str
            Ruta del archivo Parquet a cargar.

        Returns
        -------
        pl.DataFrame
            DataFrame con los datos cargados.
        """
        return pl.read_parquet(file_path)

    @staticmethod
    def save_csv_pd(data: pd.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo CSV usando Pandas.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo CSV donde se guardará el DataFrame.
        """
        data.to_csv(path, index=False)

    @staticmethod
    def save_csv_pl(data: pl.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo CSV usando Polars.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo CSV donde se guardará el DataFrame.
        """
        data.write_csv(path)

    @staticmethod
    def save_parquet_pd(data: pd.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo Parquet usando Pandas.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo Parquet donde se guardará el DataFrame.
        """
        data.to_parquet(path)

    @staticmethod
    def save_parquet_pl(data: pl.DataFrame, path: str) -> None:
        """
        Guardar un DataFrame en un archivo Parquet usando Polars.

        Parameters
        ----------
        data : pl.DataFrame
            DataFrame a guardar.
        path : str
            Ruta del archivo Parquet donde se guardará el DataFrame.
        """
        data.write_parquet(path)

    @staticmethod
    def load_pickle(file_path: str) -> object:
        """
        Carga un objeto desde un archivo pickle.

        Parameters
        ----------
        file_path : str
            Ruta del archivo pickle.

        Returns
        -------
        object
            Objeto cargado desde el archivo pickle.
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    @staticmethod
    def save_pickle(data: object, file_path: str) -> None:
        """
        Guarda un objeto en un archivo pickle.

        Parameters
        ----------
        data : object
            Objeto a guardar.
        file_path : str
            Ruta del archivo pickle donde se guardará el objeto.

        Returns
        -------
        None
        """
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)

    @staticmethod
    def fetch_content(file_path: str) -> str:
        """
        Obtiene el contenido de una URL o lo lee desde un archivo local.

        Parameters
        ----------
        file_path : str
            La ruta al archivo o URL.

        Returns
        -------
        str
            El contenido del archivo o URL como una cadena.
        """
        if file_path.startswith('http://') or file_path.startswith('https://'):
            # Convertir la URL a la versión raw si es de GitHub
            if 'github.com' in file_path:
                file_path = file_path.replace('github.com',
                                              'raw.githubusercontent.com').replace('/blob/', '/')

            response = requests.get(file_path)
            if response.status_code == 200:
                # Verifica que el contenido no está vacío
                if response.text.strip():
                    return response.text
                else:
                    raise ValueError("El contenido de la URL está vacío")
            else:
                raise ValueError(f"Error al obtener los datos desde la URL: {response.status_code}")
        else:
            with open(file_path, 'r') as file:
                content = file.read()
            return content

    @staticmethod
    def load_json_pl(file_path: str) -> pl.DataFrame:
        """
        Carga datos desde un archivo JSON o una URL que contiene JSONL utilizando polars.

        Parameters
        ----------
        file_path : str
            Ruta del archivo JSON o URL a cargar.

        Returns
        -------
        pl.DataFrame
            DataFrame con los datos cargados.
        """
        content = Utils.fetch_content(file_path)
        try:
            df = pl.read_ndjson(StringIO(content))
            return df
        except Exception as e:
            print(f"Error al leer JSONL con polars: {e}")
            # Imprime las primeras líneas para depuración
            lines = content.splitlines()[:5]
            for line in lines:
                print(line)
            raise

    @staticmethod
    def load_json_pd(file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo JSON o una URL que contiene JSONL utilizando pandas.

        Parameters
        ----------
        file_path : str
            Ruta del archivo JSON o URL a cargar.

        Returns
        -------
        pd.DataFrame
            DataFrame con los datos cargados.
        """
        content = Utils.fetch_content(file_path)
        try:
            json_objects = [json.loads(line) for line in content.splitlines() if line.strip()]
            df = pd.DataFrame(json_objects)
            return df
        except ValueError as e:
            print(f"Error al leer JSONL con pandas: {e}")
            # Imprime las primeras líneas para depuración
            lines = content.splitlines()[:5]
            for line in lines:
                print(line)
            raise

    @staticmethod
    def load_and_save_data(data_paths, save_directory, save_paths):
        data = [Utils.load_json_pl(path) for path in data_paths]
        Utils.setup_logging().info('Lectura de datos completada...')
        
        for d, save_path in zip(data, save_paths):
            full_save_path = os.path.join(save_directory, save_path)
            Utils.save_parquet_pl(d, full_save_path)
            Utils.setup_logging().info(f'Guardado de datos en {full_save_path} completado...')
        
        return data

    @staticmethod
    def process_and_save_data(data, save_directory, save_paths, process_fn, save_fn, parameters=None):
        processed_data = [process_fn(d, parameters) for d in data]
        for d, save_path in zip(processed_data, save_paths):
            full_save_path = os.path.join(save_directory, save_path)
            save_fn(d, full_save_path)
            Utils.setup_logging().info(f'Guardado de datos en {full_save_path} completado...')

