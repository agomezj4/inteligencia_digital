from typing import Dict

import os
import sys
import yaml
import logging


class Utils:
    """
    Clase para funciones de utilidad.
    """

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