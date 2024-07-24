"""
Lógica del pipeline models
"""

from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from transformers import BertModel, Trainer, TrainingArguments
import torch.nn as nn

from src.natural_language_processing.utils import Utils, CustomNLPDataset, CustomTrainer
from sklearn.metrics import classification_report

logger = Utils.setup_logging()


class PipelineModels:
    
    # 1. Definición del modelo
    @staticmethod
    def create_model_pd(params: Dict[str, Any]) -> nn.Module:
        """
        Define y retorna un modelo BERT customizado para clasificación.

        Parameters
        ----------
        params: Dict[str, Any]
            Diccionario de parámetros models:
            Número de características adicionales (features) relativas al juego.

        Returns
        -------
        nn.Module
            Modelo BERT customizado.
        """

        # Parámetros
        num_features = params['num_features']

        class CustomBERTModel(nn.Module):
            def __init__(self, num_features):
                super(CustomBERTModel, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.classifier = nn.Linear(self.bert.config.hidden_size + num_features, 2)
                self.softmax = nn.Softmax(dim=1)
                self.config = self.bert.config
                self.config.num_labels = 2  # Asumiendo clasificación binaria

            def forward(self, input_ids, attention_mask, features):
                """
                Realiza el forward pass del modelo.

                Parameters
                ----------
                input_ids : torch.Tensor
                    IDs de los tokens de entrada.
                attention_mask : torch.Tensor
                    Máscaras de atención para los tokens de entrada.
                features : torch.Tensor
                    Características adicionales del juego.

                Returns
                -------
                torch.Tensor
                    Logits de salida del clasificador.
                """
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                combined = torch.cat((pooled_output, features), dim=1)
                logits = self.classifier(combined)
                return self.softmax(logits)

        logger.info("Modelo BERT customizado creado con éxito.")
        
        return CustomBERTModel(num_features=num_features)

    # 2. Entrenamiento del modelo
    @staticmethod
    def prepare_datasets_pd(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame, 
        params: Dict[str, Any],
        device: torch.device
    ) -> Tuple[CustomNLPDataset, CustomNLPDataset]:
        """
        Prepara los datasets de entrenamiento y prueba.

        Parameters
        ----------
        df_train : pd.DataFrame
            DataFrame de Pandas de entrenamiento.
        df_test : pd.DataFrame
            DataFrame de Pandas de testeo.
        params: Dict[str, Any]
            Diccionario de parámetros.

        Returns
        -------
        dict
            Diccionario con los datasets procesados.
        """

        # Parámetros
        tensor_col1 = params['tensor_cols'][0]
        tensor_col2 = params['tensor_cols'][1]
        feature_columns = [col for col in df_train.columns if col not in [params['target_col']] + params['tensor_cols']]
        label_column = params['target_col']

        # Convertir las características y etiquetas train a tensores
        train_features = df_train[feature_columns].values
        train_labels = df_train[label_column].values
        train_input_ids = df_train[tensor_col1].tolist()
        train_attention_mask = df_train[tensor_col2].tolist()

        # Convertir las características y etiquetas test a tensores
        test_features = df_test[feature_columns].values
        test_labels = df_test[label_column].values
        test_input_ids = df_test[tensor_col1].tolist()
        test_attention_mask = df_test[tensor_col2].tolist()

        # Crear los datasets
        train_dataset = CustomNLPDataset(
            feature_values=train_features, 
            labels=train_labels, 
            input_ids=train_input_ids, 
            attention_mask=train_attention_mask,
            device=device
        )

        test_dataset = CustomNLPDataset(
            feature_values=test_features, 
            labels=test_labels, 
            input_ids=test_input_ids, 
            attention_mask=test_attention_mask,
            device=device
        )

        return train_dataset, test_dataset

    @staticmethod
    def train_model_pd(model: nn.Module, df_train: CustomNLPDataset, df_test: CustomNLPDataset, params: Dict[str, Any]) -> Trainer:
        """
        Entrena el modelo BERT customizado utilizando el dataset proporcionado.

        Parameters
        ----------
        model : nn.Module
            Modelo BERT customizado a entrenar.
        df_train : CustomNLPDataset
            Dataset de entrenamiento.
        df_test : CustomNLPDataset
            Dataset de testeo.
        params: Dict[str, Any]
            Diccionario de parámetros models.

        Returns
        -------
        nn.Module
            El mejor modelo entrenado.
        """
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)

        # Parámetros
        training_args = TrainingArguments(
            output_dir=params['output_dir'],
            num_train_epochs=params['num_train_epochs'],
            per_device_train_batch_size=params['per_device_train_batch_size'],
            per_device_eval_batch_size=params['per_device_eval_batch_size'],
            warmup_steps=params['warmup_steps'],
            weight_decay=params['weight_decay'],
            logging_dir='./logs',
            logging_steps=10,
        )

        # Entrenador
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=df_train,
            eval_dataset=df_test,
        )

        logger.info("Iniciando el entrenamiento del modelo...")
        trainer.train()
        logger.info("Entrenamiento del modelo completado.")

        return trainer
    
    # # 3. Evaluación del modelo
    # @staticmethod
    # def evaluate_model_pd(model: nn.Module, df_validation: pd.DataFrame, params: Dict[str, Any]) -> None:
    #     """
    #     Evalúa el modelo BERT customizado utilizando el dataset de validación proporcionado.

    #     Parameters
    #     ----------
    #     model : nn.Module
    #         Modelo BERT customizado a evaluar.
    #     df_validation : pd.DataFrame
    #         DataFrame de Pandas de validación.
    #     params: Dict[str, Any]
    #         Diccionario de parámetros models.

    #     Returns
    #     -------
    #     None
    #     """
    #     def compute_metrics(p: EvalPrediction) -> Dict:
    #         """
    #         Calcula métricas de evaluación.

    #         Parameters
    #         ----------
    #         p : EvalPrediction
    #             Predicciones y etiquetas verdaderas.

    #         Returns
    #         -------
    #         dict
    #             Diccionario con las métricas calculadas.
    #         """
    #         preds = np.argmax(p.predictions, axis=1)
    #         return classification_report(p.label_ids, preds, output_dict=True)

    #     training_args = TrainingArguments(
    #         per_device_eval_batch_size=params['per_device_eval_batch_size'],
    #     )

    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         eval_dataset=df_validation,  # Se usa eval_dataset aquí para evaluación final
    #         data_collator=lambda data: {
    #             'input_ids': torch.stack([f['input_ids'] for f in data]),
    #             'attention_mask': torch.stack([f['attention_mask'] for f in data]),
    #             'features': torch.stack([f['features'] for f in data]),
    #             'labels': torch.stack([f['labels'] for f in data])
    #         },
    #         compute_metrics=compute_metrics
    #     )

    #     logger.info("Iniciando la evaluación del modelo...")
    #     results = trainer.evaluate()
    #     logger.info(f"Resultados de la evaluación: {results}")

    #     return results
