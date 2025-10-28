"""
Serviço para análise de séries temporais
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from scipy.stats import f_oneway
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class TimeSeriesAnalyzer:
    """Classe responsável pela análise completa de séries temporais"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.decomposition = None
        self.analysis_results = {}
        self.seasonality_results = {}

    def load_data(self, show_message=True):
        """Carrega dados do arquivo CSV"""
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
        """Calcula estatísticas básicas da série temporal"""
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
        """Detecta a frequência da série temporal"""
        time_diffs = self.df.index.to_series().diff().dropna()
        
        if time_diffs.empty:
            self.analysis_results['frequency'] = {
                'description': "Dados insuficientes para detectar frequência",
                'code': None,
                'seconds': None
            }
            return None, "Dados insuficientes para detectar frequência"
        
        most_common_diff = time_diffs.mode()
        
        if most_common_diff.empty:
            self.analysis_results['frequency'] = {
                'description': "Frequência não detectada",
                'code': None,
                'seconds': None
            }
            return None, "Frequência não detectada"
        
        most_common_diff = most_common_diff[0]
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
        """Detecta padrões de sazonalidade avançados"""
        temp_df = self.df.copy()
        temp_df['minute'] = temp_df.index.minute
        temp_df['hour'] = temp_df.index.hour
        temp_df['day_of_week'] = temp_df.index.dayofweek
        temp_df['is_weekend'] = temp_df['day_of_week'].isin([5, 6])
        temp_df['is_business_day'] = ~temp_df['is_weekend']

        seasonality_tests = {}
        overall_mean = temp_df['value'].mean()
        
        # Teste horário (por minuto)
        seasonality_tests['hourly'] = self._test_seasonality_by_group(
            temp_df, 'minute', overall_mean
        )
        
        # Teste diário (por hora)
        seasonality_tests['daily'] = self._test_seasonality_by_group(
            temp_df, 'hour', overall_mean
        )
        
        # Teste semanal
        seasonality_tests['weekly'] = self._test_weekly_seasonality(
            temp_df, overall_mean
        )

        # Classificação final
        classification, season_type = self._classify_seasonality(seasonality_tests)

        self.seasonality_results = {
            'classification': classification,
            'season_type': season_type,
            'hourly_test': seasonality_tests.get('hourly', {}),
            'daily_test': seasonality_tests.get('daily', {}),
            'weekly_test': seasonality_tests.get('weekly', {}),
            'patterns': self.get_detailed_patterns(temp_df)
        }

        return self.seasonality_results

    def _test_seasonality_by_group(self, temp_df, group_col, overall_mean):
        """Testa sazonalidade para um grupo específico (minute/hour)"""
        if temp_df[group_col].nunique() <= 1:
            return {'has_pattern': False, 'reason': f'Dados sem variação de {group_col}'}
        
        groups = [group['value'].values for name, group in temp_df.groupby(group_col) if len(group) >= 2]
        
        if len(groups) < 3:
            return {'has_pattern': False, 'reason': 'Dados insuficientes'}
        
        try:
            f_stat, p_value = f_oneway(*groups)
            group_means = temp_df.groupby(group_col)['value'].mean()
            
            temp_df_with_mean = temp_df.copy()
            temp_df_with_mean['group_mean'] = temp_df_with_mean[group_col].map(group_means)
            ssb = ((temp_df_with_mean['group_mean'] - overall_mean) ** 2).sum()
            sst = ((temp_df['value'] - overall_mean) ** 2).sum()
            variance_explained = (ssb / sst) * 100 if sst > 0 else 0
            
            return {
                'has_pattern': p_value < 0.05,
                'p_value': p_value,
                'f_statistic': f_stat,
                'variance_explained': variance_explained,
                f'peak_{group_col}': group_means.idxmax(),
                f'low_{group_col}': group_means.idxmin(),
                f'{group_col}_range': group_means.max() - group_means.min()
            }
        except Exception as e:
            return {'has_pattern': False, 'error': str(e)}

    def _test_weekly_seasonality(self, temp_df, overall_mean):
        """Testa sazonalidade semanal"""
        if temp_df['day_of_week'].nunique() <= 1:
            return {'has_pattern': False, 'reason': 'Dados sem variação de dia da semana'}
        
        weekly_groups = [group['value'].values for name, group in temp_df.groupby('day_of_week') if len(group) >= 2]
        
        if len(weekly_groups) < 3:
            return {'has_pattern': False, 'reason': 'Dados insuficientes'}
        
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

            return {
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
            return {'has_pattern': False, 'error': str(e)}

    def _classify_seasonality(self, seasonality_tests):
        """Classifica o tipo de sazonalidade"""
        hourly_strong = seasonality_tests.get('hourly', {}).get('variance_explained', 0) > 10
        daily_strong = seasonality_tests.get('daily', {}).get('variance_explained', 0) > 10
        weekly_strong = seasonality_tests.get('weekly', {}).get('variance_explained', 0) > 9

        hourly_significant = seasonality_tests.get('hourly', {}).get('has_pattern', False)
        daily_significant = seasonality_tests.get('daily', {}).get('has_pattern', False)
        weekly_significant = seasonality_tests.get('weekly', {}).get('has_pattern', False)

        if hourly_significant and hourly_strong:
            return "SAZONAL_HORARIA", "horaria (intra-hora)"
        elif daily_significant and daily_strong:
            if weekly_significant and weekly_strong:
                return "SAZONAL_SEMANAL", "mista (diaria + semanal)"
            else:
                return "SAZONAL_DIARIA", "diaria"
        elif weekly_significant and weekly_strong:
            return "SAZONAL_SEMANAL", "semanal"
        elif hourly_significant or daily_significant or weekly_significant:
            return "LINEAR", "fraca"
        else:
            return "LINEAR", "não sazonal"

    def get_detailed_patterns(self, temp_df):
        """Obtém padrões detalhados por período"""
        patterns = {}
        if temp_df['minute'].nunique() > 1:
            patterns['minute'] = temp_df.groupby('minute')['value'].agg(['mean', 'std', 'count'])
        if temp_df['hour'].nunique() > 1:
            patterns['hourly'] = temp_df.groupby('hour')['value'].agg(['mean', 'std', 'count'])
        if temp_df['day_of_week'].nunique() > 1:
            patterns['weekly'] = temp_df.groupby('day_of_week')['value'].agg(['mean', 'std', 'count'])
        return patterns

    def seasonal_decomposition(self):
        """Realiza decomposição sazonal da série temporal"""
        try:
            freq_seconds = self.analysis_results['frequency']['seconds']
            n = len(self.df)
            seasonality_info = self.seasonality_results

            # Determinar período baseado na classificação de sazonalidade
            if seasonality_info['classification'] == 'SAZONAL_HORARIA':
                period = 12  # Exemplo: intra-hora
            elif seasonality_info['classification'] == 'SAZONAL_DIARIA':
                period = 288  # Padrões diários (24 horas * 60 minutos / 5 minutos)
            elif seasonality_info['classification'] == 'SAZONAL_SEMANAL':
                period = 2016  # Padrões semanais (288 períodos por dia * 7 dias)
            else:
                period = max(2, min(n // 4, 12))

            period = min(period, n // 2)

            self.decomposition = seasonal_decompose(
                self.df['value'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )

            # Calcular variâncias
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
                'seasonal_percentage': (seasonal_var / total_var) * 100 if total_var != 0 else 0,
                'trend_percentage': (trend_var / total_var) * 100 if total_var != 0 else 0,
                'residual_percentage': (residual_var / total_var) * 100 if total_var != 0 else 0,
                'classification': seasonality_info['classification']
            }
            return True
        except Exception as e:
            st.error(f"Erro na decomposição: {e}")
            return False

    def detect_jumps(self, threshold_std=3):
        """Detecta saltos e outliers na série temporal"""
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

    def detect_anomaly_algorithm(self):
        """
        Determina qual algoritmo de detecção de anomalias usar baseado nas características da série.
        
        Retorna:
            tuple: (algorithm_name, description, metrics_dict)
        
        Algoritmos:
            - BASIC: Para séries sem padrão sazonal (LINEAR)
            - ROBUST: Para séries sazonais estáveis (mudanças lentas são anomalias)
            - AGILE: Para séries sazonais com shifts frequentes (adaptação rápida)
        """
        classification = self.seasonality_results.get('classification', 'LINEAR')
        
        # Série LINEAR = BASIC
        if classification == 'LINEAR':
            metrics = {
                'algorithm': 'BASIC',
                'reason': 'Série sem padrão sazonal repetitivo',
                'characteristics': 'Use algoritmo simples baseado em rolling quantile'
            }
            self.analysis_results['anomaly_detection'] = metrics
            return 'BASIC', 'Série sem padrão sazonal repetitivo', metrics
        
        # Série SAZONAL = verificar estabilidade
        decomp = self.analysis_results.get('decomposition', {})
        
        # Métricas de estabilidade
        trend_variance_pct = decomp.get('trend_percentage', 0)
        
        # Calcular coeficiente de variação da tendência
        trend_cv = 0
        if self.decomposition is not None:
            trend_values = self.decomposition.trend.dropna()
            if len(trend_values) > 0 and trend_values.mean() != 0:
                trend_cv = trend_values.std() / abs(trend_values.mean())
        
        # Critérios de decisão para estabilidade
        # Uma série é considerada ESTÁVEL se:
        # 1. A tendência explica menos de 7% da variância total
        # 2. O coeficiente de variação da tendência é baixo (< 0.2)
        is_stable = (
            trend_variance_pct < 7 and  # Tendência tem baixa influência
            trend_cv < 0.2              # Tendência é pouco volátil
        )
        
        # Montar dicionário de métricas
        metrics = {
            'trend_variance_pct': round(trend_variance_pct, 2),
            'trend_coefficient_variation': round(trend_cv, 4),
            'is_stable': is_stable
        }
        
        if is_stable:
            algorithm = 'ROBUST'
            description = 'Métrica sazonal estável - mudanças lentas de nível são consideradas anomalias'
            characteristics = 'Algoritmo estável, previsões constantes mesmo com anomalias longas'
            metrics['algorithm'] = algorithm
            metrics['reason'] = description
            metrics['characteristics'] = characteristics
        else:
            algorithm = 'AGILE'
            description = 'Métrica sazonal com shifts - algoritmo se adapta rapidamente a mudanças de nível'
            characteristics = 'Ajuste rápido a shifts de nível, menos robusto a anomalias longas'
            metrics['algorithm'] = algorithm
            metrics['reason'] = description
            metrics['characteristics'] = characteristics
        
        self.analysis_results['anomaly_detection'] = metrics
        return algorithm, description, metrics

    def generate_plots(self):
        """Gera todos os gráficos da análise"""
        plots = {}
        
        # Gráfico da série original
        fig1 = px.line(self.df, x=self.df.index, y='value', title='Série Temporal Original')
        fig1.update_layout(xaxis_title='Tempo', yaxis_title='Valor', height=500)
        plots['original'] = fig1

        # Distribuição
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Histograma', 'Box Plot'))
        fig2.add_trace(go.Histogram(x=self.df['value'], nbinsx=50, marker_color='skyblue'), row=1, col=1)
        fig2.add_trace(go.Box(y=self.df['value'], marker_color='lightcoral'), row=1, col=2)
        fig2.update_layout(title_text='Distribuição dos Valores', height=500)
        plots['distribution'] = fig2

        # Decomposição sazonal
        if self.decomposition is not None:
            plots['decomposition'] = self._create_decomposition_plot()

        # Padrões sazonais
        if hasattr(self, 'seasonality_results') and 'patterns' in self.seasonality_results:
            patterns_plot = self._create_patterns_plot()
            if patterns_plot:
                plots['patterns'] = patterns_plot

        # Detecção de saltos
        plots['jumps'] = self._create_jumps_plot()

        return plots

    def _create_decomposition_plot(self):
        """Cria gráfico de decomposição sazonal"""
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True,
            subplot_titles=('Série Original', 'Tendência', 'Componente Sazonal', 'Resíduos')
        )
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['value'], name='Original'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.trend, name='Tendência'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.seasonal, name='Sazonal'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.decomposition.resid, name='Resíduos'), row=4, col=1)
        fig.update_layout(height=900, title_text="Decomposição Sazonal", showlegend=False)
        return fig

    def _create_patterns_plot(self):
        """Cria gráfico de padrões sazonais"""
        patterns = self.seasonality_results['patterns']
        available_patterns = [k for k in ['minute', 'hourly', 'weekly'] if k in patterns and len(patterns[k]) > 0]

        if not available_patterns:
            return None

        n_plots = len(available_patterns)
        fig = make_subplots(
            rows=1, cols=n_plots,
            subplot_titles=[
                k.replace('minute', 'Por Minuto')
                .replace('hourly', 'Por Hora')
                .replace('weekly', 'Por Dia da Semana') 
                for k in available_patterns
            ]
        )

        for idx, pattern_type in enumerate(available_patterns, 1):
            if pattern_type == 'minute':
                x_vals = list(range(60))
                y_vals = [patterns['minute'].loc[m, 'mean'] if m in patterns['minute'].index else 0 for m in range(60)]
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', 
                               name='Média por Minuto', marker_color='lightblue'),
                    row=1, col=idx
                )
            elif pattern_type == 'hourly':
                x_vals = list(range(24))
                y_vals = [patterns['hourly'].loc[h, 'mean'] if h in patterns['hourly'].index else 0 for h in range(24)]
                fig.add_trace(
                    go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', 
                               name='Média por Hora', marker_color='lightgreen'),
                    row=1, col=idx
                )
            elif pattern_type == 'weekly':
                days = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
                y_vals = [patterns['weekly'].loc[d, 'mean'] if d in patterns['weekly'].index else 0 for d in range(7)]
                fig.add_trace(
                    go.Bar(x=days, y=y_vals, name='Média por Dia', marker_color='lightcoral'),
                    row=1, col=idx
                )

        fig.update_layout(height=500, title_text="Padrões Sazonais Detalhados")
        return fig

    def _create_jumps_plot(self):
        """Cria gráfico de detecção de saltos"""
        fig = px.line(self.df, x=self.df.index, y='value', title='Detecção de Saltos/Outliers')
        fig.update_traces(line=dict(width=1.5))

        jumps = self.analysis_results.get('jumps', {})
        if jumps.get('jump_dates'):
            jump_dates = jumps['jump_dates']
            jump_values = [self.df.loc[date, 'value'] for date in jump_dates]
            fig.add_trace(go.Scatter(
                x=jump_dates, y=jump_values, mode='markers',
                marker=dict(color='red', size=10), name='Saltos Detectados'
            ))

        fig.update_layout(xaxis_title='Tempo', yaxis_title='Valor', height=500)
        return fig