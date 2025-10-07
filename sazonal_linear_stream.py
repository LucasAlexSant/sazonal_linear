import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.stats import f_oneway
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import requests
import json
import re
import logging
import urllib3
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Datadog + Análise de Séries Temporais",
    page_icon="📊",
    layout="wide"
)

# Configurações do Datadog
DATADOG_URL = "https://api.proxy.datadog.prod.aws.cloud.ihf/api/v1/query"
DATADOG_HEADERS = {
    "Accept": "application/json",
    "DD-API-KEY": "org-itau-cc-prod",
    "DD-APPLICATION-KEY": "org-itau-cc-prod"
}

class DatadogFetcher:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    def parse_query(self, query):
        try:
            match = re.match(r"(?P<aggregation>\w+):(?P<metrica>[\w\.]+)\{(?P<params>.+)\}", query)
            if match:
                aggregation = match.group("aggregation")
                metrica = match.group("metrica")
                params = match.group("params")
                
                param_dict = {}
                for param in params.split(","):
                    if ":" in param:
                        key, value = param.split(":", 1)
                        param_dict[key.strip()] = value.strip()
                
                return {
                    "aggregation": aggregation,
                    "metrica": metrica,
                    "params": param_dict
                }
            return None
        except Exception as e:
            st.error(f"Erro ao processar query: {e}")
            return None
    
    def fetch_datadog_data(self, params):
        try:
            resp = requests.get(DATADOG_URL, headers=DATADOG_HEADERS, params=params, verify=False, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            
            if 'series' not in data or not data['series']:
                return []
            
            results = []
            for serie in data['series']:
                pontos = serie.get("pointlist", [])
                for ponto in pontos:
                    timestamp, valor = ponto
                    results.append({
                        "Timestamp": datetime.fromtimestamp(timestamp / 1000),
                        "Valor": valor
                    })
            return results
        except Exception as e:
            st.error(f"Erro ao fazer requisição: {e}")
            return []
    
    def process_metric(self, metric_info, days_back):
        aggregation = metric_info['aggregation']
        metrica = metric_info['metrica']
        params = metric_info['params']
        
        identifier = "_".join([f"{key}-{value}" for key, value in params.items()])
        query_params = ",".join([f"{key}:{value}" for key, value in params.items()])
        query = f"{aggregation}:{metrica}{{{query_params}}}"
        
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
        delta = 1 * 24 * 60 * 60
        all_data = []
        current_start = start_time
        
        while current_start < end_time:
            current_end = min(current_start + delta, end_time)
            params_req = {
                'from': current_start,
                'to': current_end,
                'query': query
            }
            data = self.fetch_datadog_data(params_req)
            all_data.extend(data)
            current_start = current_end
        
        if all_data:
            df = pd.DataFrame(all_data)
            df = df.sort_values(by="Timestamp")
            df.set_index("Timestamp", inplace=True)
            
            file_name = f"{identifier}.csv".replace("N/A", "NA").replace("/", "_")
            file_path = os.path.join(self.output_folder, file_name)
            df.to_csv(file_path)
            return file_name
        return None

class TimeSeriesAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.decomposition = None
        self.analysis_results = {}
        self.seasonality_results = {}

    def load_data(self, show_message=True):
        try:
            self.df = pd.read_csv(self.file_path, sep=None, engine='python')

            rename_map = {}
            if 'Timestamp' in self.df.columns:
                rename_map['Timestamp'] = 'datetime'
            if 'Valor' in self.df.columns:
                rename_map['Valor'] = 'value'
            self.df.rename(columns=rename_map, inplace=True)

            if 'datetime' not in self.df.columns or 'value' not in self.df.columns:
                if show_message:
                    st.error("O arquivo deve conter as colunas 'Timestamp' e 'Valor'")
                return False

            self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            self.df = self.df.dropna().sort_values('datetime')
            self.df.set_index('datetime', inplace=True)

            if show_message:
                file_name = os.path.basename(self.file_path)
                st.success(f"✅ Dados do arquivo '{file_name}' carregados com sucesso!")

                col1, col2, col3 = st.columns(3)
                col1.metric("Período Inicial", self.df.index.min().strftime('%Y-%m-%d'))
                col2.metric("Período Final", self.df.index.max().strftime('%Y-%m-%d'))
                col3.metric("Total de Observações", len(self.df))

        except Exception as e:
            if show_message:
                st.error(f"❌ Erro ao carregar dados: {e}")
            return False
        return True

    def basic_statistics(self):
        values = self.df['value']
        stats_dict = {
            'Contagem': len(values),
            'Média': values.mean(),
            'Desvio Padrão': values.std(),
            'Mínimo': values.min(),
            'Máximo': values.max(),
            'Mediana': values.median(),
            'Assimetria': stats.skew(values),
            'Curtose': stats.kurtosis(values)
        }
        self.analysis_results['basic_stats'] = stats_dict
        return stats_dict

    def detect_frequency(self):
        time_diffs = self.df.index.to_series().diff().dropna()
        most_common_diff = time_diffs.mode()[0]
        seconds = most_common_diff.total_seconds()

        if seconds < 60:
            freq_desc, freq_code = f"{seconds:.0f} segundos", f"{int(seconds)}S"
        elif seconds < 3600:
            freq_desc, freq_code = f"{seconds/60:.0f} minutos", f"{int(seconds/60)}min"
        elif seconds < 86400:
            freq_desc, freq_code = f"{seconds/3600:.0f} horas", f"{int(seconds/3600)}H"
        else:
            freq_desc, freq_code = f"{seconds/86400:.0f} dias", f"{int(seconds/86400)}D"

        self.analysis_results['frequency'] = {
            'description': freq_desc,
            'code': freq_code,
            'seconds': seconds
        }
        return freq_code, freq_desc

    def advanced_seasonality_detection(self):
        temp_df = self.df.copy()
        temp_df['minute'] = temp_df.index.minute
        temp_df['hour'] = temp_df.index.hour
        temp_df['day_of_week'] = temp_df.index.dayofweek
        temp_df['is_weekend'] = temp_df['day_of_week'].isin([5, 6])
        temp_df['is_business_day'] = ~temp_df['is_weekend']

        seasonality_tests = {}
        
        # Teste horário (por minuto)
        if temp_df['minute'].nunique() > 1:
            minute_groups = [group['value'].values for name, group in temp_df.groupby('minute') if len(group) >= 2]
            
            if len(minute_groups) >= 3:
                try:
                    f_stat, p_value_minute = f_oneway(*minute_groups)
                    minute_means = temp_df.groupby('minute')['value'].mean()
                    overall_mean = temp_df['value'].mean()
                    
                    temp_df_with_minute_mean = temp_df.copy()
                    temp_df_with_minute_mean['minute_mean'] = temp_df_with_minute_mean['minute'].map(minute_means)
                    ssb = ((temp_df_with_minute_mean['minute_mean'] - overall_mean) ** 2).sum()
                    sst = ((temp_df['value'] - overall_mean) ** 2).sum()
                    minute_variance_explained = (ssb / sst) * 100 if sst > 0 else 0
                    
                    seasonality_tests['hourly'] = {
                        'has_pattern': p_value_minute < 0.05,
                        'p_value': p_value_minute,
                        'f_statistic': f_stat,
                        'variance_explained': minute_variance_explained,
                        'peak_minute': minute_means.idxmax(),
                        'low_minute': minute_means.idxmin(),
                        'minute_range': minute_means.max() - minute_means.min()
                    }
                except Exception as e:
                    seasonality_tests['hourly'] = {'has_pattern': False, 'error': str(e)}
            else:
                seasonality_tests['hourly'] = {'has_pattern': False, 'reason': 'Dados insuficientes'}
        else:
            seasonality_tests['hourly'] = {'has_pattern': False, 'reason': 'Dados sem variação de minutos'}

        # Teste diário (por hora)
        if temp_df['hour'].nunique() > 1:
            hourly_groups = [group['value'].values for name, group in temp_df.groupby('hour') if len(group) >= 2]

            if len(hourly_groups) >= 3:
                try:
                    f_stat, p_value_hourly = f_oneway(*hourly_groups)
                    hourly_means = temp_df.groupby('hour')['value'].mean()
                    overall_mean = temp_df['value'].mean()

                    temp_df_with_hour_mean = temp_df.copy()
                    temp_df_with_hour_mean['hour_mean'] = temp_df_with_hour_mean['hour'].map(hourly_means)
                    ssb = ((temp_df_with_hour_mean['hour_mean'] - overall_mean) ** 2).sum()
                    sst = ((temp_df['value'] - overall_mean) ** 2).sum()
                    hourly_variance_explained = (ssb / sst) * 100 if sst > 0 else 0

                    seasonality_tests['daily'] = {
                        'has_pattern': p_value_hourly < 0.05,
                        'p_value': p_value_hourly,
                        'f_statistic': f_stat,
                        'variance_explained': hourly_variance_explained,
                        'peak_hour': hourly_means.idxmax(),
                        'low_hour': hourly_means.idxmin(),
                        'hour_range': hourly_means.max() - hourly_means.min()
                    }
                except Exception as e:
                    seasonality_tests['daily'] = {'has_pattern': False, 'error': str(e)}
            else:
                seasonality_tests['daily'] = {'has_pattern': False, 'reason': 'Dados insuficientes'}

        # Teste semanal
        if temp_df['day_of_week'].nunique() > 1:
            weekly_groups = [group['value'].values for name, group in temp_df.groupby('day_of_week') if len(group) >= 2]

            if len(weekly_groups) >= 3:
                try:
                    f_stat_weekly, p_value_weekly = f_oneway(*weekly_groups)
                    weekly_means = temp_df.groupby('day_of_week')['value'].mean()

                    temp_df_with_day_mean = temp_df.copy()
                    temp_df_with_day_mean['day_mean'] = temp_df_with_day_mean['day_of_week'].map(weekly_means)
                    ssb_weekly = ((temp_df_with_day_mean['day_mean'] - overall_mean) ** 2).sum()
                    sst_weekly = ((temp_df['value'] - overall_mean) ** 2).sum()
                    weekly_variance_explained = (ssb_weekly / sst_weekly) * 100 if sst_weekly > 0 else 0

                    business_days_values = temp_df[temp_df['is_business_day']]['value']
                    weekend_values = temp_df[temp_df['is_weekend']]['value']

                    weekday_weekend_diff = abs(business_days_values.mean() - weekend_values.mean())
                    weekday_weekend_pct_diff = (weekday_weekend_diff / overall_mean) * 100 if overall_mean != 0 else 0

                    seasonality_tests['weekly'] = {
                        'has_pattern': p_value_weekly < 0.05,
                        'p_value': p_value_weekly,
                        'f_statistic': f_stat_weekly,
                        'variance_explained': weekly_variance_explained,
                        'peak_day': weekly_means.idxmax(),
                        'low_day': weekly_means.idxmin(),
                        'weekday_weekend_diff': weekday_weekend_pct_diff,
                        'business_day_avg': business_days_values.mean(),
                        'weekend_avg': weekend_values.mean()
                    }
                except Exception as e:
                    seasonality_tests['weekly'] = {'has_pattern': False, 'error': str(e)}

        # Classificação final
        hourly_strong = seasonality_tests.get('hourly', {}).get('variance_explained', 0) > 10
        daily_strong = seasonality_tests.get('daily', {}).get('variance_explained', 0) > 10
        weekly_strong = seasonality_tests.get('weekly', {}).get('variance_explained', 0) > 10

        hourly_significant = seasonality_tests.get('hourly', {}).get('has_pattern', False)
        daily_significant = seasonality_tests.get('daily', {}).get('has_pattern', False)
        weekly_significant = seasonality_tests.get('weekly', {}).get('has_pattern', False)

        if hourly_significant and hourly_strong:
            main_seasonality = "SAZONAL_HORARIA"
            season_type = "horária (intra-hora)"
        elif daily_significant and daily_strong:
            if weekly_significant and weekly_strong:
                main_seasonality = "SAZONAL_SEMANAL"
                season_type = "mista (diária + semanal)"
            else:
                main_seasonality = "SAZONAL_DIARIA"
                season_type = "diária"
        elif weekly_significant and weekly_strong:
            main_seasonality = "SAZONAL_SEMANAL"
            season_type = "semanal"
        elif hourly_significant or daily_significant or weekly_significant:
            main_seasonality = "LINEAR"
            season_type = "fraca"
        else:
            main_seasonality = "LINEAR"
            season_type = "não sazonal"

        self.seasonality_results = {
            'classification': main_seasonality,
            'season_type': season_type,
            'hourly_test': seasonality_tests.get('hourly', {}),
            'daily_test': seasonality_tests.get('daily', {}),
            'weekly_test': seasonality_tests.get('weekly', {}),
            'patterns': self.get_detailed_patterns(temp_df)
        }

        return self.seasonality_results

    def get_detailed_patterns(self, temp_df):
        patterns = {}
        if temp_df['minute'].nunique() > 1:
            patterns['minute'] = temp_df.groupby('minute')['value'].agg(['mean', 'std', 'count'])
        if temp_df['hour'].nunique() > 1:
            patterns['hourly'] = temp_df.groupby('hour')['value'].agg(['mean', 'std', 'count'])
        if temp_df['day_of_week'].nunique() > 1:
            patterns['weekly'] = temp_df.groupby('day_of_week')['value'].agg(['mean', 'std', 'count'])
        return patterns

    def seasonal_decomposition(self):
        try:
            freq_seconds = self.analysis_results['frequency']['seconds']
            n = len(self.df)
            seasonality_info = self.seasonality_results

            if seasonality_info['classification'] == 'SAZONAL_HORARIA':
                period = 60 if freq_seconds <= 60 else (24 if freq_seconds <= 3600 else max(2, n // 10))
            elif seasonality_info['classification'] == 'SAZONAL_DIARIA':
                period = 24 if freq_seconds <= 3600 else (7 if freq_seconds <= 86400 else max(2, n // 10))
            elif seasonality_info['classification'] == 'SAZONAL_SEMANAL':
                period = 7 if freq_seconds <= 86400 else max(2, n // 10)
            else:
                period = max(2, min(n // 4, 12))

            period = min(period, n // 2)

            self.decomposition = seasonal_decompose(
                self.df['value'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )

            total_var = np.var(self.df['value'])
            seasonal_var = np.var(self.decomposition.seasonal.dropna())
            trend_var = np.var(self.decomposition.trend.dropna())
            residual_var = np.var(self.decomposition.resid.dropna())

            self.analysis_results['decomposition'] = {
                'period': period,
                'season_type': seasonality_info['season_type'],
                'total_variance': total_var,
                'seasonal_variance': seasonal_var,
                'trend_variance': trend_var,
                'residual_variance': residual_var,
                'seasonal_percentage': (seasonal_var / total_var) * 100,
                'trend_percentage': (trend_var / total_var) * 100,
                'residual_percentage': (residual_var / total_var) * 100,
                'classification': seasonality_info['classification']
            }
            return True
        except Exception as e:
            st.error(f"Erro na decomposição: {e}")
            return False

    def detect_jumps(self, threshold_std=3):
        diff = self.df['value'].diff()
        threshold = threshold_std * diff.std()
        jumps = diff[abs(diff) > threshold]

        self.analysis_results['jumps'] = {
            'count': len(jumps),
            'threshold': threshold,
            'jump_dates': jumps.index.tolist(),
            'jump_values': jumps.values.tolist()
        }
        return len(jumps), threshold, jumps

    def generate_plots(self):
        plots = {}
        fig1 = px.line(self.df, x=self.df.index, y='value', title='Série Temporal Original')
        fig1.update_layout(xaxis_title='Tempo', yaxis_title='Valor', height=500)
        plots['original'] = fig1

        fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Histograma', 'Box Plot'))
        fig2.add_trace(go.Histogram(x=self.df['value'], nbinsx=50, marker_color='skyblue'), row=1, col=1)
        fig2.add_trace(go.Box(y=self.df['value'], marker_color='lightcoral'), row=1, col=2)
        fig2.update_layout(title_text='Distribuição dos Valores', height=500)
        plots['distribution'] = fig2

        if self.decomposition is not None:
            fig3 = make_subplots(
                rows=4, cols=1, shared_xaxes=True,
                subplot_titles=('Série Original', 'Tendência', 'Componente Sazonal', 'Resíduos')
            )
            fig3.add_trace(go.Scatter(x=self.df.index, y=self.df['value'], name='Original'), row=1, col=1)
            fig3.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.trend, name='Tendência'), row=2, col=1)
            fig3.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.seasonal, name='Sazonal'), row=3, col=1)
            fig3.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.resid, name='Resíduos'), row=4, col=1)
            fig3.update_layout(height=900, title_text="Decomposição Sazonal", showlegend=False)
            plots['decomposition'] = fig3

        if hasattr(self, 'seasonality_results') and 'patterns' in self.seasonality_results:
            patterns = self.seasonality_results['patterns']
            available_patterns = [k for k in ['minute', 'hourly', 'weekly'] if k in patterns and len(patterns[k]) > 0]

            if available_patterns:
                n_plots = len(available_patterns)
                fig4 = make_subplots(
                    rows=1, cols=n_plots,
                    subplot_titles=[k.replace('minute', 'Por Minuto').replace('hourly', 'Por Hora').replace('weekly', 'Por Dia da Semana') for k in available_patterns]
                )

                for idx, pattern_type in enumerate(available_patterns, 1):
                    if pattern_type == 'minute':
                        x_vals = list(range(60))
                        y_vals = [patterns['minute'].loc[m, 'mean'] if m in patterns['minute'].index else 0 for m in range(60)]
                        fig4.add_trace(
                            go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='Média por Minuto', marker_color='lightblue'),
                            row=1, col=idx
                        )
                    elif pattern_type == 'hourly':
                        x_vals = list(range(24))
                        y_vals = [patterns['hourly'].loc[h, 'mean'] if h in patterns['hourly'].index else 0 for h in range(24)]
                        fig4.add_trace(
                            go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name='Média por Hora', marker_color='lightgreen'),
                            row=1, col=idx
                        )
                    elif pattern_type == 'weekly':
                        days = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                        y_vals = [patterns['weekly'].loc[d, 'mean'] if d in patterns['weekly'].index else 0 for d in range(7)]
                        fig4.add_trace(
                            go.Bar(x=days, y=y_vals, name='Média por Dia', marker_color='lightcoral'),
                            row=1, col=idx
                        )

                fig4.update_layout(height=500, title_text="Padrões Sazonais Detalhados")
                plots['patterns'] = fig4

        fig5 = px.line(self.df, x=self.df.index, y='value', title='Detecção de Saltos/Outliers')
        fig5.update_traces(line=dict(width=1.5))

        jumps = self.analysis_results.get('jumps', {})
        if jumps.get('jump_dates'):
            jump_dates = jumps['jump_dates']
            jump_values = [self.df.loc[date, 'value'] for date in jump_dates]
            fig5.add_trace(go.Scatter(
                x=jump_dates, y=jump_values, mode='markers',
                marker=dict(color='red', size=10), name='Saltos Detectados'
            ))

        fig5.update_layout(xaxis_title='Tempo', yaxis_title='Valor', height=500)
        plots['jumps'] = fig5

        return plots


def main():
    st.title("📊 Sistema Integrado Datadog + Análise de Séries Temporais")
    st.markdown("**Coleta de dados do Datadog e análise avançada de séries temporais**")
    st.markdown("---")

    # Criação de abas
    tab1, tab2 = st.tabs(["📥 Coletar Dados do Datadog", "📈 Analisar Séries Temporais"])

    # ABA 1: Coleta de Dados do Datadog
    with tab1:
        st.header("Coleta de Dados do Datadog")
        
        col1, col2 = st.columns(2)
        with col1:
            output_folder = st.text_input("Pasta de Saída", value="series_temporais")
        with col2:
            days_back = st.number_input("Dias Retroativos", min_value=1, max_value=365, value=90)
        
        queries_text = st.text_area(
            "Queries do Datadog (uma por linha)",
            height=200,
            placeholder="avg:trace.kafka.message.process_time{sigla:agtsch,servicename:agtsch-notifica-consumer-worker}\navg:trace.kafka.message.process_time{sigla:agtsch,servicename:agtsch-oferta-consumer-worker}"
        )
        
        if st.button("🚀 Iniciar Coleta", type="primary"):
            if queries_text.strip():
                queries = [q.strip() for q in queries_text.split('\n') if q.strip()]
                fetcher = DatadogFetcher(output_folder)
                
                st.info(f"Coletando dados de {len(queries)} queries...")
                progress_bar = st.progress(0)
                
                parsed_queries = []
                for query in queries:
                    parsed = fetcher.parse_query(query)
                    if parsed:
                        parsed_queries.append(parsed)
                
                success_count = 0
                for idx, metric_info in enumerate(parsed_queries):
                    result = fetcher.process_metric(metric_info, days_back)
                    if result:
                        success_count += 1
                    progress_bar.progress((idx + 1) / len(parsed_queries))
                
                st.success(f"✅ Coleta concluída! {success_count} arquivos criados na pasta '{output_folder}'")
            else:
                st.warning("Por favor, insira pelo menos uma query")

    # ABA 2: Análise de Séries Temporais
    with tab2:
        st.header("Análise de Séries Temporais")
        
        base_folder = "series_temporais"
        
        # Função para listar subpastas
        def list_subfolders(folder):
            if os.path.exists(folder):
                return [f.name for f in os.scandir(folder) if f.is_dir()]
            return []
        
        # Navegação por subpastas dentro de series_temporais
        current_folder = base_folder
        
        if os.path.exists(base_folder):
            # Listar primeiro nível de subpastas
            level1_folders = list_subfolders(base_folder)
            
            if level1_folders:
                selected_level1 = st.selectbox(
                    "Escolha a categoria principal:",
                    level1_folders
                )
                current_folder = os.path.join(base_folder, selected_level1)
                
                # Verificar se há subpastas no segundo nível
                level2_folders = list_subfolders(current_folder)
                
                if level2_folders:
                    selected_level2 = st.selectbox(
                        f"Escolha a subpasta dentro de '{selected_level1}':",
                        level2_folders
                    )
                    current_folder = os.path.join(current_folder, selected_level2)
                    
                    # Verificar se há mais níveis (opcional)
                    level3_folders = list_subfolders(current_folder)
                    if level3_folders:
                        selected_level3 = st.selectbox(
                            f"Escolha a subpasta dentro de '{selected_level2}':",
                            level3_folders
                        )
                        current_folder = os.path.join(current_folder, selected_level3)
            
            series_folder = current_folder
            st.info(f"📁 Pasta selecionada: `{series_folder}`")
        else:
            st.error(f"A pasta base '{base_folder}' não foi encontrada")
            series_folder = None
        
        if os.path.exists(series_folder):
            csv_files = [f for f in os.listdir(series_folder) if f.endswith('.csv')]
            
            if csv_files:
                st.info(f"Encontrados {len(csv_files)} arquivos CSV na pasta")
                
                # Opção de análise em lote
                col1, col2 = st.columns([1, 3])
                with col1:
                    analysis_mode = st.radio("Modo de Análise", ["Individual", "Lote"])
                
                with col2:
                    threshold_std = st.slider(
                        "Threshold para detecção de saltos (desvios padrão)",
                        min_value=1.0,
                        max_value=5.0,
                        value=3.0,
                        step=0.1
                    )
                
                # ANÁLISE EM LOTE
                if analysis_mode == "Lote":
                    if st.button("🚀 Analisar TODOS os Arquivos", type="primary"):
                        all_results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for idx, file_name in enumerate(csv_files, start=1):
                            status_text.text(f"Analisando {file_name} ({idx}/{len(csv_files)})...")
                            
                            file_path = os.path.join(series_folder, file_name)
                            analyzer = TimeSeriesAnalyzer(file_path)
                            
                            if not analyzer.load_data(show_message=False):
                                continue
                            
                            analyzer.basic_statistics()
                            analyzer.detect_frequency()
                            seasonality_results = analyzer.advanced_seasonality_detection()
                            analyzer.seasonal_decomposition()
                            analyzer.detect_jumps(threshold_std)
                            
                            basic = analyzer.analysis_results.get("basic_stats", {})
                            jumps = analyzer.analysis_results.get("jumps", {})
                            hourly_test = seasonality_results.get('hourly_test', {})
                            daily_test = seasonality_results.get('daily_test', {})
                            weekly_test = seasonality_results.get('weekly_test', {})
                            
                            all_results.append({
                                "arquivo": file_name,
                                "classificacao": seasonality_results.get('classification', 'N/A'),
                                "tipo_sazonalidade": seasonality_results.get('season_type', 'N/A'),
                                "variancia_sazonal_horaria": hourly_test.get('variance_explained', 0),
                                "variancia_sazonal_diaria": daily_test.get('variance_explained', 0),
                                "variancia_sazonal_semanal": weekly_test.get('variance_explained', 0),
                                "saltos_detectados": jumps.get("count", 0),
                                "media": basic.get("Média", np.nan),
                                "desvio_padrao": basic.get("Desvio Padrão", np.nan),
                            })
                            
                            progress_bar.progress(idx / len(csv_files))
                        
                        status_text.text("✅ Análise em lote concluída!")
                        progress_bar.empty()
                        
                        # Exibir resultados consolidados
                        st.header("📑 Resumo Consolidado")
                        results_df = pd.DataFrame(all_results)
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Resumo de classificação
                        st.subheader("📊 Resumo Geral de Classificação")
                        classification_counts = results_df['classificacao'].value_counts().to_dict()
                        
                        emoji_map = {
                            'SAZONAL_HORARIA': '🕐',
                            'SAZONAL_DIARIA': '🌅',
                            'SAZONAL_SEMANAL': '📅',
                            'SAZONAL_MISTA': '🔄',
                            'LINEAR': '📈'
                        }
                        
                        cols = st.columns(len(classification_counts))
                        for idx, (classification, count) in enumerate(classification_counts.items()):
                            emoji = emoji_map.get(classification, '❓')
                            cols[idx].metric(
                                f"{emoji} {classification.replace('_', ' ')}",
                                count,
                                delta=f"{(count/len(csv_files)*100):.1f}%"
                            )
                        
                        # Download consolidado
                        csv_data = results_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download Consolidado (CSV)",
                            csv_data,
                            "analise_consolidada.csv",
                            "text/csv"
                        )
                
                # ANÁLISE INDIVIDUAL
                else:
                    selected_file = st.selectbox("Escolha uma série temporal:", csv_files)
                    file_path = os.path.join(series_folder, selected_file)
                
                if st.button("🔍 Executar Análise", type="primary"):
                    analyzer = TimeSeriesAnalyzer(file_path)
                    
                    if analyzer.load_data():
                        with st.spinner("Processando análise..."):
                            basic_stats = analyzer.basic_statistics()
                            freq_code, freq_desc = analyzer.detect_frequency()
                            seasonality_results = analyzer.advanced_seasonality_detection()
                            analyzer.seasonal_decomposition()
                            jump_count, threshold, jumps = analyzer.detect_jumps(threshold_std)
                            plots = analyzer.generate_plots()
                        
                        st.success("✅ Análise concluída!")
                        
                        # Exibir resultados
                        st.subheader("📊 Estatísticas Básicas")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Média", f"{basic_stats['Média']:.4f}")
                        col2.metric("Desvio Padrão", f"{basic_stats['Desvio Padrão']:.4f}")
                        col3.metric("Frequência", freq_desc)
                        
                        st.subheader("🔬 Sazonalidade")
                        classification = seasonality_results['classification']
                        season_type = seasonality_results['season_type']
                        st.info(f"**Classificação:** {classification} | **Tipo:** {season_type}")
                        
                        col1, col2, col3 = st.columns(3)
                        hourly_test = seasonality_results.get('hourly_test', {})
                        daily_test = seasonality_results.get('daily_test', {})
                        weekly_test = seasonality_results.get('weekly_test', {})
                        
                        with col1:
                            st.markdown("**Sazonalidade Horária**")
                            if hourly_test.get('has_pattern', False):
                                st.success(f"Detectado: {hourly_test.get('variance_explained', 0):.1f}%")
                            else:
                                st.info("Não detectado")
                        
                        with col2:
                            st.markdown("**Sazonalidade Diária**")
                            if daily_test.get('has_pattern', False):
                                st.success(f"Detectado: {daily_test.get('variance_explained', 0):.1f}%")
                            else:
                                st.info("Não detectado")
                        
                        with col3:
                            st.markdown("**Sazonalidade Semanal**")
                            if weekly_test.get('has_pattern', False):
                                st.success(f"Detectado: {weekly_test.get('variance_explained', 0):.1f}%")
                            else:
                                st.info("Não detectado")
                        
                        st.subheader("📈 Visualizações")
                        st.plotly_chart(plots['original'], use_container_width=True)
                        st.plotly_chart(plots['distribution'], use_container_width=True)
                        
                        if 'decomposition' in plots:
                            st.plotly_chart(plots['decomposition'], use_container_width=True)
                        
                        if 'patterns' in plots:
                            st.plotly_chart(plots['patterns'], use_container_width=True)
                        
                        st.plotly_chart(plots['jumps'], use_container_width=True)
                        
                        if jump_count > 0:
                            st.warning(f"Foram detectados {jump_count} saltos significativos")
                        
                        # Download dos resultados
                        results_summary = {
                            'arquivo': selected_file,
                            'classificacao': classification,
                            'tipo_sazonalidade': season_type,
                            'variancia_sazonal_horaria': hourly_test.get('variance_explained', 0),
                            'variancia_sazonal_diaria': daily_test.get('variance_explained', 0),
                            'variancia_sazonal_semanal': weekly_test.get('variance_explained', 0),
                            'frequencia': freq_desc,
                            'saltos_detectados': jump_count,
                            'media': basic_stats['Média'],
                            'desvio_padrao': basic_stats['Desvio Padrão'],
                        }
                        
                        results_df = pd.DataFrame([results_summary])
                        csv_results = results_df.to_csv(index=False)
                        
                        st.download_button(
                            label="💾 Download Resumo da Análise (CSV)",
                            data=csv_results,
                            file_name=f"analise_{selected_file.replace('.csv', '')}_resumo.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("Nenhum arquivo CSV encontrado na pasta especificada")
        else:
            st.error("A pasta especificada não existe")

if __name__ == "__main__":
    main()
