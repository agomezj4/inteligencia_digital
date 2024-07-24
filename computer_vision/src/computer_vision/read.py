import re

class ReadJSON:
    """
    Clase para extraer información de un certificado de propiedad a partir de un archivo JSON.
    """

    @staticmethod
    def extract_info(json_data: dict) -> dict:
        """
        Extrae el Número de Matrícula, fecha de impresión, departamento, municipio,
        localidad (vereda) y estado del folio del JSON de un certificado de propiedad.

        Parameters
        ----------
        json_data : dict
            Diccionario con los datos del archivo JSON.

        Returns
        -------
        dict
            Diccionario con la información extraída.
        """
        numero_matricula = None
        fecha_impresion = None
        departamento = None
        municipio = None
        localidad = None
        estado_folio = None
        estado_folio_found = False

        # Definición de patrones de búsqueda
        fecha_patron = re.compile(
            r'impreso el\s(\d{1,2})\sde\s(\w+)\sde\s(\d{4})', re.IGNORECASE
        )
        mes_traduccion = {
            'enero': '01', 'febrero': '02', 'marzo': '03', 'abril': '04',
            'mayo': '05', 'junio': '06', 'julio': '07', 'agosto': '08',
            'septiembre': '09', 'octubre': '10', 'noviembre': '11', 'diciembre': '12'
        }

        for i, block in enumerate(json_data.get('Blocks', [])):
            if block['BlockType'] == 'LINE':
                text = block.get('Text', '')

                # Buscar Número de Matrícula
                match = re.search(r'\bmatrícula\b[:\s]*([^\s]+)', text, re.IGNORECASE)
                if match:
                    numero_matricula = match.group(1).strip()

                # Buscar Fecha de Impresión
                match = fecha_patron.search(text.lower())
                if match:
                    dia, mes, anio = match.group(1), match.group(2), match.group(3)
                    mes_num = mes_traduccion.get(mes.lower(), '01')
                    fecha_impresion = f"{anio}-{mes_num}-{int(dia):02d}"

                # Buscar Departamento
                match = re.search(r'depto[:\s]+([^\s]+)', text, re.IGNORECASE)
                if match:
                    departamento = match.group(1).strip()

                # Buscar Municipio
                match = re.search(r'municipio[:\s]+([^\s]+)', text, re.IGNORECASE)
                if match:
                    municipio = match.group(1).strip()

                # Buscar Localidad (Vereda)
                match = re.search(r'vereda[:\s]+([^\s]*)', text, re.IGNORECASE)
                if match:
                    localidad = match.group(1).strip() or None

                # Buscar Estado del Folio
                if 'estado del folio' in text.lower():
                    estado_folio_found = True

                if estado_folio_found:
                    if i + 1 < len(json_data['Blocks']):
                        next_block = json_data['Blocks'][i + 1]
                        if next_block['BlockType'] == 'LINE':
                            estado_folio = next_block.get('Text', '').strip()
                            estado_folio_found = False

        return {
            'numero_matricula': numero_matricula,
            'fecha_impresion': fecha_impresion,
            'departamento': departamento,
            'municipio': municipio,
            'localidad': localidad,
            'estado_folio': estado_folio
        }

