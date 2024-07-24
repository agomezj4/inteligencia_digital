import sys
from src.natural_language_processing.utils import Utils
from src.natural_language_processing.orchestration import PipelineOrchestration

Utils.add_src_to_path()


def main():
    if len(sys.argv) > 1:
        pipeline = sys.argv[1]
        if pipeline == 'All Pipelines':
            PipelineOrchestration.run_pipeline_raw()
            PipelineOrchestration.run_pipeline_intermediate()
            PipelineOrchestration.run_pipeline_primary()
            PipelineOrchestration.run_pipeline_feature()
            PipelineOrchestration.run_pipeline_model_input()
            PipelineOrchestration.run_pipeline_models()

        elif pipeline == 'Pipeline Raw':
            PipelineOrchestration.run_pipeline_raw()

        elif pipeline == 'Pipeline Intermediate':
            PipelineOrchestration.run_pipeline_intermediate()

        elif pipeline == 'Pipeline Primary':
            PipelineOrchestration.run_pipeline_primary()

        elif pipeline == 'Pipeline Feature':
            PipelineOrchestration.run_pipeline_feature()

        elif pipeline == 'Pipeline Model Input':
            PipelineOrchestration.run_pipeline_model_input()

        elif pipeline == 'Pipeline Models':
            PipelineOrchestration.run_pipeline_models()

        else:
            print(f"Pipeline '{pipeline}' no reconocido.")
    else:
        print("No se especificó un pipeline. Uso: python __main__.py [pipeline]")


if __name__ == "__main__":
    main()

