import os
from .utils import Utils

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')


parameters = Utils.load_parameters(parameters_directory)


class PipelineOrchestration:

    # 1. Pipeline Raw
    @staticmethod
    def run_pipeline_raw():

        logger.info('Inicio Pipeline Raw')

        train_data_path = parameters['parameters_catalog']['raw_data_train_path_github']
        test_data_path = parameters['parameters_catalog']['raw_data_test_path_github']
        validation_data_path = parameters['parameters_catalog']['raw_data_validation_path_github']

        data_train = Utils.load_json_pl(train_data_path)
        data_test = Utils.load_json_pl(test_data_path)
        data_validation = Utils.load_json_pl(validation_data_path)

        logger.info('Lectura de datos raw desde Github completada...')

        raw_data_train_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_train_path'])
        Utils.save_data_pl(data_train, raw_data_train_path)

        raw_data_test_path = os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_test_path'])
        Utils.save_data_pl(data_test, raw_data_test_path)

        raw_data_validation_path = os.path.join(data_raw_directory,
                                                parameters['parameters_catalog']['raw_data_validation_path'])
        Utils.save_data_pl(data_validation, raw_data_validation_path)

        logger.info('Guardado de datos raw completado...')
        logger.info('Fin Pipeline Raw')



