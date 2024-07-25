# inteligencia_digital
Este repositorio contiene las soluciones para tres problemas específicos: procesamiento de lenguaje natural (NLP), visión por computadora y programación/SQL. 

## Computer Vision
Este módulo aborda la parte de visión por computadora del proyecto. Su estructura es la siguiente:

1. Data: Contiene los insumos necesarios para el procesamiento de los archivos JSON.

- 01-load: Carpeta con la información original de los certificados de propiedad a cargar.
- 02-save: Carpeta donde se guarda en formato CSV la información extraída de los JSON.


2. Src: Directorio que incluye los métodos y lógica del procesamiento.

- computer_vision: Contiene los archivos con los métodos para los pasos de carga (load), lectura (read) y guardado (save). También incluye métodos utilitarios que se usan recurrentemente.
- parameters: Incluye un archivo YAML para configurar la ruta de los archivos JSON a procesar.

3. Main: Explica cómo ejecutar todo el proceso desde la raíz del proyecto usando el comando python __main__.py [step]. Los pasos disponibles son:

- Load JSON: Solo carga los archivos JSON.
- Extract Info: Extrae la información requerida.
- All Steps: Ejecuta todo el proceso de carga, extracción y guardado.

4. Environment: Archivos para la creación de un entorno aislado de dependencias.

## Natural Language Processing (NLP)
Este módulo cubre el procesamiento de lenguaje natural. La estructura es:

1. Data: Almacena los datos en formato Parquet, organizados por las siguientes etapas del pipeline:

- raw, intermediate, primary, feature, model_input: Cada carpeta representa una etapa en el pipeline de procesamiento, donde los datos son transformados y pasados a la siguiente fase.

2. Src: Contiene el código fuente y los parámetros del proyecto.

- natural_language_processing: Incluye los archivos con los pipelines y la lógica de orquestación del procesamiento.

3. Main: Similar al módulo de visión, aquí se ejecuta la lógica principal usando python __main__.py [pipeline].

4- Environment: Configuración para crear un entorno de desarrollo con dependencias aisladas.

## SQL
Este módulo se encarga de la parte de SQL, especialmente la estandarización de datos.

1. Data: Contiene los archivos de datos que requieren estandarización.
2. Notebooks: Incluye notebooks con una explicación detallada paso a paso del proceso de estandarización.
