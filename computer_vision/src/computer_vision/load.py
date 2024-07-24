import json
import requests


class LoadJSON:
    """
    Clase para cargar datos desde un archivo JSON o una URL que contiene JSON.
    """

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
        try:
            if file_path.startswith('http://') or file_path.startswith('https://'):
                response = requests.get(file_path)
                response.raise_for_status()
                return response.text
            else:
                with open(file_path, 'r') as file:
                    return file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"El archivo {file_path} no se encontró.")
        except requests.RequestException as e:
            raise ValueError(f"Error al obtener los datos desde la URL: {e}")

    @staticmethod
    def load_json(file_path: str) -> dict:
        """
        Carga datos desde un archivo JSON o una URL.

        Parameters
        ----------
        file_path : str
            Ruta del archivo JSON o URL a cargar.

        Returns
        -------
        dict
            Diccionario con los datos cargados.
        """
        content = LoadJSON.fetch_content(file_path)
        return json.loads(content)