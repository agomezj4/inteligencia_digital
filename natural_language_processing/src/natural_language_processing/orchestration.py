import os
from .utils import Utils

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_intermediate_directory = os.path.join(project_root, 'data', '02_intermediate')
data_primary_directory = os.path.join(project_root, 'data', '03_primary')

parameters = Utils.load_parameters(parameters_directory)


class PipelineOrchestration:
    
    # 1. Pipeline Raw
    @staticmethod
    def run_pipeline_raw():
        logger.info('Inicio Pipeline Raw')
        
        data_paths = [
            parameters['parameters_catalog']['raw_data_train_path_github'],
            parameters['parameters_catalog']['raw_data_test_path_github'],
            parameters['parameters_catalog']['raw_data_validation_path_github']
        ]
        
        save_paths = [
            parameters['parameters_catalog']['raw_data_train_path'],
            parameters['parameters_catalog']['raw_data_test_path'],
            parameters['parameters_catalog']['raw_data_validation_path']
        ]
        
        Utils.load_and_save_data(data_paths, data_raw_directory, save_paths)
        
        logger.info('Fin Pipeline Raw')

    # 2. Pipeline Intermediate
    @staticmethod
    def run_pipeline_intermediate():
        from .pipelines.intermediate import PipelineIntermediate

        logger.info('Inicio Pipeline Intermediate')
        
        raw_data_paths = [
            os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_train_path']),
            os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_test_path']),
            os.path.join(data_raw_directory, parameters['parameters_catalog']['raw_data_validation_path'])
        ]

        raw_data = [Utils.load_parquet_pl(path) for path in raw_data_paths]
        logger.info('Lectura de datos raw completada...')

        expanded_data = [PipelineIntermediate.expand_messages_pl(d, parameters['parameters_intermediate']) for d in raw_data]
        intermediate_data = [PipelineIntermediate.columns_to_int_pl(d, parameters['parameters_intermediate']) for d in expanded_data]
        
        intermediate_save_paths = [
            parameters['parameters_catalog']['intermediate_data_train_path'],
            parameters['parameters_catalog']['intermediate_data_test_path'],
            parameters['parameters_catalog']['intermediate_data_validation_path']
        ]
        
        Utils.process_and_save_data(intermediate_data, 
                                    data_intermediate_directory, 
                                    intermediate_save_paths, 
                                    process_fn=lambda d, p: d, 
                                    save_fn=Utils.save_parquet_pl,
                                    parameters=parameters['parameters_intermediate'])

        logger.info('Fin Pipeline Intermediate')
