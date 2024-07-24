import os
import pandas as pd
import torch
from .utils import Utils

logger = Utils.setup_logging()
Utils.add_src_to_path()
project_root = Utils.get_project_root()

parameters_directory = os.path.join(project_root, 'src', 'parameters')
data_raw_directory = os.path.join(project_root, 'data', '01_raw')
data_intermediate_directory = os.path.join(project_root, 'data', '02_intermediate')
data_primary_directory = os.path.join(project_root, 'data', '03_primary')
data_feature_directory = os.path.join(project_root, 'data', '04_feature')
data_model_input_directory = os.path.join(project_root, 'data', '05_model_input')

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

    # 3. Pipeline Primary
    @staticmethod
    def run_pipeline_primary():
        from .pipelines.primary import PipelinePrimary

        logger.info('Inicio Pipeline Primary')
        
        intermediate_data_paths = [
            os.path.join(data_intermediate_directory, parameters['parameters_catalog']['intermediate_data_train_path']),
            os.path.join(data_intermediate_directory, parameters['parameters_catalog']['intermediate_data_test_path']),
            os.path.join(data_intermediate_directory, parameters['parameters_catalog']['intermediate_data_validation_path'])
        ]

        intermediate_data = [Utils.load_parquet_pl(path) for path in intermediate_data_paths]
        logger.info('Lectura de datos intermediate completada...')

        recategorize_data = [PipelinePrimary.recategorize_players_pl(d, parameters['parameters_primary']) for d in intermediate_data]
        binary_data = [PipelinePrimary.binary_encode_pl(d, parameters['parameters_primary']) for d in recategorize_data]
        primary_data = [PipelinePrimary.tokenize_messages_pl(d, parameters['parameters_primary']) for d in binary_data]
        
        primary_save_paths = [
            parameters['parameters_catalog']['primary_data_train_path'],
            parameters['parameters_catalog']['primary_data_test_path'],
            parameters['parameters_catalog']['primary_data_validation_path']
        ]
        
        Utils.process_and_save_data(primary_data, 
                                    data_primary_directory, 
                                    primary_save_paths, 
                                    process_fn=lambda d, p: d, 
                                    save_fn=Utils.save_parquet_pl,
                                    parameters=parameters['parameters_primary'])

        logger.info('Fin Pipeline Primary')

    # 4. Pipeline Feature
    @staticmethod
    def run_pipeline_feature():
        from .pipelines.feature import PipelineFeature

        logger.info('Inicio Pipeline Feature')
        
        primary_data_paths = [
            os.path.join(data_primary_directory, parameters['parameters_catalog']['primary_data_train_path']),
            os.path.join(data_primary_directory, parameters['parameters_catalog']['primary_data_test_path']),
            os.path.join(data_primary_directory, parameters['parameters_catalog']['primary_data_validation_path'])
        ]

        primary_data = [Utils.load_parquet_pl(path) for path in primary_data_paths]
        logger.info('Lectura de datos primary completada...')

        new_columns = [PipelineFeature.features_new_pl_pd(d, parameters['parameters_feature']) for d in primary_data]

        concatenated_data = pd.concat(new_columns, axis=0, ignore_index=True)

        encoding_data = PipelineFeature.one_hot_encoding_pd(concatenated_data, parameters['parameters_feature'])
        feature_selected_data = PipelineFeature.feature_selection_pipeline_pd(encoding_data, parameters['parameters_feature'])

        train_size = len(primary_data[0])
        test_size = len(primary_data[1])

        feature_data = [
            feature_selected_data.iloc[:train_size].reset_index(drop=True),
            feature_selected_data.iloc[train_size:train_size + test_size].reset_index(drop=True),
            feature_selected_data.iloc[train_size + test_size:].reset_index(drop=True)
        ]
        
        feature_save_paths = [
            parameters['parameters_catalog']['feature_data_train_path'],
            parameters['parameters_catalog']['feature_data_test_path'],
            parameters['parameters_catalog']['feature_data_validation_path']
        ]
        
        Utils.process_and_save_data(feature_data, 
                                    data_feature_directory, 
                                    feature_save_paths, 
                                    process_fn=lambda d, p: d, 
                                    save_fn=Utils.save_parquet_pd,
                                    parameters=parameters['parameters_feature'])

        logger.info('Fin Pipeline Feature')

    # 5. Pipeline Model Input
    @staticmethod
    def run_pipeline_model_input():
        from .pipelines.model_input import PipelineModelInput

        logger.info('Inicio Pipeline Model Input')
        
        feature_data_paths = [
            os.path.join(data_feature_directory, parameters['parameters_catalog']['feature_data_train_path']),
            os.path.join(data_feature_directory, parameters['parameters_catalog']['feature_data_test_path']),
            os.path.join(data_feature_directory, parameters['parameters_catalog']['feature_data_validation_path'])
        ]

        feature_data = [Utils.load_parquet_pd(path) for path in feature_data_paths]
        logger.info('Lectura de datos feature completada...')

        model_input_data = [PipelineModelInput.min_max_scaler_pd(d) for d in feature_data]

        model_input_save_paths = [
            parameters['parameters_catalog']['model_input_data_train_path'],
            parameters['parameters_catalog']['model_input_data_test_path'],
            parameters['parameters_catalog']['model_input_data_validation_path']
        ]
        
        Utils.process_and_save_data(model_input_data, 
                                    data_model_input_directory, 
                                    model_input_save_paths, 
                                    process_fn=lambda d, p: d, 
                                    save_fn=Utils.save_parquet_pd,
                                    parameters=parameters['parameters_model_input'])

        logger.info('Fin Pipeline Model Input')

    # 6. Pipeline Model
    @staticmethod
    def run_pipeline_models():
        from .pipelines.models import PipelineModels

        logger.info('Inicio Pipeline Model')
        
        model_input_data_paths = [
            os.path.join(data_model_input_directory, parameters['parameters_catalog']['model_input_data_train_path']),
            os.path.join(data_model_input_directory, parameters['parameters_catalog']['model_input_data_test_path']),
            os.path.join(data_model_input_directory, parameters['parameters_catalog']['model_input_data_validation_path'])
        ]

        model_input_data = [Utils.load_parquet_pd(path) for path in model_input_data_paths]
        logger.info('Lectura de datos model input completada...')

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = PipelineModels.create_model_pd(parameters['parameters_models'])
        model = model.to(device)  # Mueve el modelo al dispositivo
        data_model = PipelineModels.prepare_datasets_pd(model_input_data[0], model_input_data[1], parameters['parameters_models'], device)
        training = PipelineModels.train_model_pd(model, data_model[0], data_model[1], parameters['parameters_models'])

        models_save_path = parameters['parameters_catalog']['model_training_data_path']

        Utils.save_pickle(training, models_save_path)

        logger.info('Fin Pipeline Model')