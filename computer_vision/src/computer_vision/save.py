import pandas as pd

class SaveInfo:
    """
    Clase para guardar información extraída en un archivo CSV.
    """

    @staticmethod
    def save_to_csv(file_path: str, data: dict) -> None:
        """
        Guarda los datos en un archivo CSV.

        Parameters
        ----------
        file_path : str
            Ruta del archivo CSV donde se guardarán los datos.

        data : dict
            Diccionario con los datos a guardar. Debe contener las claves:
            'numero_matricula', 'fecha_impresion', 'departamento', 'municipio',
            'localidad', y 'estado_folio'.

        """
        # Crear un DataFrame con los datos proporcionados
        df = pd.DataFrame([data])

        # Guardar el DataFrame en un archivo CSV
        df.to_csv(file_path, index=False)
