import sys
import os

from src.computer_vision.utils import Utils
from src.computer_vision.load import LoadJSON
from src.computer_vision.read import ReadJSON
from src.computer_vision.save import SaveInfo

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
parameters = Utils.load_parameters(parameters_directory)


def main():
    if len(sys.argv) > 1:
        step = sys.argv[1]
        if step == 'All Steps':
            logger.info("Lectura, extracción y guardado de información del certificado de propiedad...")
            load = LoadJSON.load_json(parameters['parameters_catalog']['certificado_propiedad_data_path'])
            read = ReadJSON.extract_info(load)
            SaveInfo.save_to_csv(parameters['parameters_catalog']['info_certificado_propiedad_data_path'], read)
            logger.info("Información del certificado de propiedad guardada en CSV!")

        elif step == 'Load JSON':
            logger.info("Cargando información del certificado de propiedad desde el JSON...")
            LoadJSON.load_json(parameters['parameters_catalog']['certificado_propiedad_data_path'])
            logger.info("Información del certificado de propiedad cargada!")

        elif step == 'Extract Info':
            logger.info("Extrayendo información del certificado de propiedad...")
            ReadJSON.extract_info()
            logger.info("Información del certificado de propiedad extraída!")

        else:
            print(f"Step '{step}' no reconocido.")
    else:
        print("No se especificó un step. Uso: python __main__.py [step]")


if __name__ == "__main__":
    main()

