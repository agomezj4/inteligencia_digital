"""
Lógica del pipeline models
"""

from typing import Any, Dict, List, Tuple

import torch
import pandas as pd
from transformers import BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn as nn
from src.natural_language_processing.utils import Utils, CustomNLPDataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = Utils.setup_logging()


class PipelineModels:
    
    # 1. Definición del modelo
    @staticmethod
    def create_model_pd(params: Dict[str, Any]) -> nn.Module:
        """
        Crear el modelo BERT customizado.

        Parameters
        ----------
        params : Dict[str, Any]
            Parámetros del modelo.
        
        Returns
        -------
        nn.Module
            Modelo BERT customizado.
        """

        # Parámetros
        num_features = params['num_features']

        logger.info("Creando modelo BERT customizado...")
        
        # Modelo BERT customizado
        class CustomBERTModel(nn.Module):
            def __init__(self, num_features):
                super(CustomBERTModel, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-uncased')
                self.classifier = nn.Linear(self.bert.config.hidden_size + num_features, 2)
                self.config = self.bert.config
                self.config.num_labels = 2  # Asumiendo clasificación binaria

            # Forward pass
            def forward(self, input_ids, attention_mask, features):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                combined = torch.cat((pooled_output, features), dim=1)
                logits = self.classifier(combined)
                return logits

        logger.info("Modelo BERT customizado creado con éxito.")
        
        return CustomBERTModel(num_features=num_features)


    # 2. Preparar datasets
    @staticmethod
    def prepare_datasets_pd(
        df_train: pd.DataFrame, 
        df_test: pd.DataFrame,
        df_val: pd.DataFrame,
        params: Dict[str, Any],
        device: torch.device
    ) -> Tuple[CustomNLPDataset, CustomNLPDataset]:
        """
        Preparar los datasets de entrenamiento y testeo.

        Parameters
        ----------
        df_train : pd.DataFrame
            DataFrame de entrenamiento.
        df_test : pd.DataFrame
            DataFrame de testeo.
        df_val : pd.DataFrame
            DataFrame de validación.
        params : Dict[str, Any]
            Parámetros del modelo.
        device : torch.device
            Dispositivo de cómputo.

        Returns
        -------
        Tuple[CustomNLPDataset, CustomNLPDataset]
            Datasets de entrenamiento y testeo.
        """
       
        # Parámetros
        tensor_col1 = params['tensor_cols'][0]
        tensor_col2 = params['tensor_cols'][1]
        feature_columns = [col for col in df_train.columns if col not in [params['target_col']] + params['tensor_cols']]
        label_column = params['target_col']


        # Convertir a tensores
        train_features = df_train[feature_columns].values
        train_labels = df_train[label_column].values
        train_input_ids = df_train[tensor_col1].tolist()
        train_attention_mask = df_train[tensor_col2].tolist()

        # Convertir a tensores
        test_features = df_test[feature_columns].values
        test_labels = df_test[label_column].values
        test_input_ids = df_test[tensor_col1].tolist()
        test_attention_mask = df_test[tensor_col2].tolist()

        # Convertir a tensores
        val_features = df_val[feature_columns].values
        val_labels = df_val[label_column].values
        val_input_ids = df_val[tensor_col1].tolist()
        val_attention_mask = df_val[tensor_col2].tolist()


        # Crear datasets
        train_dataset = CustomNLPDataset(
            feature_values=train_features, 
            labels=train_labels, 
            input_ids=train_input_ids, 
            attention_mask=train_attention_mask,
            device=device
        )

        # Crear datasets
        test_dataset = CustomNLPDataset(
            feature_values=test_features, 
            labels=test_labels, 
            input_ids=test_input_ids, 
            attention_mask=test_attention_mask,
            device=device
        )

        # Crear datasets
        val_dataset = CustomNLPDataset(
            feature_values=val_features,
            labels=val_labels,
            input_ids=val_input_ids,
            attention_mask=val_attention_mask,
            device=device
        )

        return train_dataset, test_dataset, val_dataset

    # 3. Entrenar el modelo
    @staticmethod
    def train_model_pd(
        model: nn.Module, 
        train_dataset: CustomNLPDataset,
        params: Dict[str, Any]
        ) -> nn.Module:
        """
        Entrenar el modelo BERT customizado.

        Parameters
        ----------
        model : nn.Module
            Modelo BERT customizado.
        train_dataset : CustomNLPDataset
            Dataset de entrenamiento.
        params : Dict[str, Any]
            Parámetros del modelo.

        Returns
        -------
        nn.Module
            Modelo BERT customizado entrenado.
        """

        # Movimiento a dispositivo
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Optimizador y scheduler
        optimizer = AdamW(model.parameters(), lr=params['learning_rate'])
        num_training_steps = len(train_dataset) * params['num_train_epochs']
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

        # Dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

        # Gradiente scaler
        scaler = torch.cuda.amp.GradScaler()

        # Función de entrenamiento
        def train():
            
            # Modo de entrenamiento
            model.train()
            total_loss = 0
            all_preds = []
            all_labels = []
            
            # Iterar sobre el dataloader
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                # Gradiente scaler
                optimizer.zero_grad()

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, features)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(outputs, labels)

                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                all_preds.extend(torch.argmax(outputs, dim=1).tolist())
                all_labels.extend(labels.tolist())

            avg_loss = total_loss / len(train_dataloader)
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)

            return avg_loss, accuracy, f1, precision, recall

        # Entrenamiento
        for epoch in range(params['num_train_epochs']):
            train_loss, train_accuracy, train_f1, train_precision, train_recall = train()
            logger.info(f"Epoch {epoch + 1}/{params['num_train_epochs']}, Train Loss: {train_loss:.4f}, "
                        f"Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, "
                        f"Recall: {train_recall:.4f}")

        return model

    # 4. Evaluar el modelo
    @staticmethod
    def evaluate_model_pd(
        model: nn.Module, 
        test_dataset: CustomNLPDataset, 
        params: Dict[str, Any]
        ) -> Dict[str, float]:
        """
        Evaluar el modelo BERT customizado.
        
        Parameters
        ----------
        model : nn.Module
            Modelo BERT customizado.
        test_dataset : CustomNLPDataset
            Dataset de testeo.
        params : Dict[str, Any]
            Parámetros del modelo.

        Returns
        -------
        Dict[str, float]
            Métricas de evaluación.
        """

        # Movimiento a dispositivo
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Dataloader
        test_dataloader = DataLoader(test_dataset, batch_size=params['batch_size'])

        # Evaluación
        model.eval()
        preds, true_labels = [], []
        total_loss = 0

        # Iterar sobre el dataloader
        with torch.no_grad():
            for batch in test_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                features = batch['features'].to(device)
                labels = batch['labels'].to(device)

                # Forward pass
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask, features)
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(outputs, labels)

                # Loss
                total_loss += loss.item()
                preds.extend(torch.argmax(outputs, dim=1).tolist())
                true_labels.extend(labels.tolist())

        # Métricas
        recall = recall_score(true_labels, preds)
        precision = precision_score(true_labels, preds)
        accuracy = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)
        avg_loss = total_loss / len(test_dataloader)

        logger.info(f"Evaluation results - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Loss: {avg_loss:.4f}")
        
        return {
            'recall': recall,
            'precision': precision,
            'accuracy': accuracy,
            'f1': f1,
            'loss': avg_loss
        }
