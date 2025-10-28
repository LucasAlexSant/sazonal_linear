"""
Serviço para coleta de dados do Datadog
"""
import os
import re
import logging
import requests
import pandas as pd
from datetime import datetime
import urllib3
from multiprocessing import Pool, cpu_count

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class DatadogFetcher:
    """Classe responsável por buscar e processar dados do Datadog"""
    
    def __init__(self, url, headers):
        self.url = url
        self.headers = headers
    
    def fetch_data(self, query, start_time, end_time):
        """Faz uma requisição à API do Datadog para o período especificado."""
        try:
            params = {"from": start_time, "to": end_time, "query": query}
            logging.info(
                f"Enviando requisição para o período de {datetime.fromtimestamp(start_time)} "
                f"até {datetime.fromtimestamp(end_time)}..."
            )
            response = requests.get(
                self.url, headers=self.headers, params=params, verify=False
            )
            response.raise_for_status()
            logging.info("Requisição concluída com sucesso.")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Erro ao fazer a requisição: {e}")
            return None
    
    @staticmethod
    def fetch_data_for_day(args):
        """Função auxiliar para fazer requisições diárias (usado em multiprocessing)."""
        url, headers, query, start_time, end_time = args
        fetcher = DatadogFetcher(url, headers)
        return fetcher.fetch_data(query, start_time, end_time)
    
    @staticmethod
    def sanitize_filename(filename):
        """Remove ou substitui caracteres inválidos no nome do arquivo."""
        sanitized = re.sub(r'[\\/*?:"<>|]', '_', filename)
        return sanitized[:100]
    
    @staticmethod
    def save_series_to_csv(identifier, df, output_folder):
        """Salva a série temporal em um arquivo CSV."""
        sanitized_identifier = DatadogFetcher.sanitize_filename(identifier)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        file_name = f"{sanitized_identifier}.csv"
        file_path = os.path.join(output_folder, file_name)
        try:
            df.to_csv(file_path, index=False)
            logging.info(f"Série temporal salva em '{file_path}'.")
        except Exception as e:
            logging.error(f"Erro ao salvar o arquivo '{file_path}': {e}")
    
    @staticmethod
    def combine_pointlists(series_list):
        """Combina os pointlists de séries com base no valor do scope."""
        combined_data = {}
        for serie in series_list:
            scope = serie.get("scope", "unknown_scope")
            if scope not in combined_data:
                combined_data[scope] = []
            pointlist = serie.get("pointlist", [])
            combined_data[scope].extend(pointlist)
        return combined_data
    
    @staticmethod
    def process_and_save_combined_data(combined_data, output_folder):
        """Processa os dados combinados e salva em arquivos CSV."""
        for identifier, pointlist in combined_data.items():
            all_data = []
            for ponto in pointlist:
                timestamp, valor = ponto
                if valor is not None:
                    all_data.append({
                        "Timestamp": datetime.fromtimestamp(timestamp / 1000),
                        "Valor": valor,
                    })
            
            if all_data:
                df = pd.DataFrame(all_data)
                df = df.sort_values(by="Timestamp")
                DatadogFetcher.save_series_to_csv(identifier, df, output_folder)
            else:
                logging.warning(
                    f"Nenhum dado válido encontrado para o identificador '{identifier}'."
                )
    
    def collect_data_parallel(self, query, start_time, end_time, output_folder):
        """
        Coleta dados do Datadog em paralelo para múltiplos intervalos.
        Retorna uma tupla (success, message, combined_data)
        """
        # Dividir o período em intervalos diários
        intervals = []
        current_start = start_time
        while current_start < end_time:
            current_end = current_start + 86400  # Adiciona 1 dia (86400 segundos)
            intervals.append((self.url, self.headers, query, current_start, current_end))
            current_start = current_end
        
        # Usar multiprocessing para paralelizar as requisições
        results = []
        with Pool(cpu_count()) as pool:
            results = list(pool.imap(DatadogFetcher.fetch_data_for_day, intervals))
        
        # Combinar os dados das séries retornadas
        all_series = []
        for data in results:
            if data and "series" in data:
                all_series.extend(data["series"])
        
        # Combinar os pointlists de todas as séries
        combined_data = self.combine_pointlists(all_series)
        
        # Processar e salvar os dados combinados em arquivos CSV
        if combined_data:
            self.process_and_save_combined_data(combined_data, output_folder)
            return True, f"✅ Coleta concluída! Arquivos salvos na pasta '{output_folder}'", combined_data
        else:
            return False, "Nenhuma série foi retornada durante o período.", None