import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from scipy.stats import f_oneway, kruskal
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Análise de Séries Temporais",
    page_icon="📈",
    layout="wide"
)

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

            # Padroniza os nomes para manter compatibilidade com o resto do código
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
        """
        Detecta padrões de sazonalidade horária, diária e semanal com testes estatísticos
        """
        temp_df = self.df.copy()
        temp_df['minute'] = temp_df.index.minute
        temp_df['hour'] = temp_df.index.hour
        temp_df['day_of_week'] = temp_df.index.dayofweek  # 0=Segunda, 6=Domingo
        temp_df['is_weekend'] = temp_df['day_of_week'].isin([5, 6])  # Sábado e Domingo
        temp_df['is_business_day'] = ~temp_df['is_weekend']

        seasonality_tests = {}
        
        # 0. TESTE DE SAZONALIDADE HORÁRIA (por minuto dentro da hora)
        if temp_df['minute'].nunique() > 1:
            # Agrupamento por minuto
            minute_groups = [group['value'].values for name, group in temp_df.groupby('minute') if len(group) >= 2]
            
            if len(minute_groups) >= 3:
                # ANOVA para testar se há diferenças significativas entre os minutos
                try:
                    f_stat, p_value_minute = f_oneway(*minute_groups)
                    
                    # Cálculo correto da variância explicada usando R²
                    minute_means = temp_df.groupby('minute')['value'].mean()
                    overall_mean = temp_df['value'].mean()
                    
                    # Soma dos quadrados entre grupos (SSB)
                    temp_df_with_minute_mean = temp_df.copy()
                    temp_df_with_minute_mean['minute_mean'] = temp_df_with_minute_mean['minute'].map(minute_means)
                    ssb = ((temp_df_with_minute_mean['minute_mean'] - overall_mean) ** 2).sum()
                    
                    # Soma dos quadrados total (SST)
                    sst = ((temp_df['value'] - overall_mean) ** 2).sum()
                    
                    # R² = SSB / SST (percentual de variância explicada)
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

        # 1. TESTE DE SAZONALIDADE DIÁRIA (por hora do dia)
        if temp_df['hour'].nunique() > 1:
            # Agrupamento por hora
            hourly_groups = [group['value'].values for name, group in temp_df.groupby('hour') if len(group) >= 2]

            if len(hourly_groups) >= 3:
                # ANOVA para testar se há diferenças significativas entre as horas
                try:
                    f_stat, p_value_hourly = f_oneway(*hourly_groups)

                    # Cálculo correto da variância explicada usando R²
                    hourly_means = temp_df.groupby('hour')['value'].mean()
                    overall_mean = temp_df['value'].mean()

                    # Soma dos quadrados entre grupos (SSB)
                    temp_df_with_hour_mean = temp_df.copy()
                    temp_df_with_hour_mean['hour_mean'] = temp_df_with_hour_mean['hour'].map(hourly_means)
                    ssb = ((temp_df_with_hour_mean['hour_mean'] - overall_mean) ** 2).sum()

                    # Soma dos quadrados total (SST)
                    sst = ((temp_df['value'] - overall_mean) ** 2).sum()

                    # R² = SSB / SST (percentual de variância explicada)
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

        # 2. TESTE DE SAZONALIDADE SEMANAL (por dia da semana)
        if temp_df['day_of_week'].nunique() > 1:
            weekly_groups = [group['value'].values for name, group in temp_df.groupby('day_of_week') if len(group) >= 2]

            if len(weekly_groups) >= 3:
                try:
                    f_stat_weekly, p_value_weekly = f_oneway(*weekly_groups)

                    # Cálculo correto da variância explicada para dias da semana usando R²
                    weekly_means = temp_df.groupby('day_of_week')['value'].mean()

                    # Soma dos quadrados entre grupos (SSB)
                    temp_df_with_day_mean = temp_df.copy()
                    temp_df_with_day_mean['day_mean'] = temp_df_with_day_mean['day_of_week'].map(weekly_means)
                    ssb_weekly = ((temp_df_with_day_mean['day_mean'] - overall_mean) ** 2).sum()

                    # Soma dos quadrados total (SST)
                    sst_weekly = ((temp_df['value'] - overall_mean) ** 2).sum()

                    # R² = SSB / SST (percentual de variância explicada)
                    weekly_variance_explained = (ssb_weekly / sst_weekly) * 100 if sst_weekly > 0 else 0

                    # Teste específico: dias úteis vs fins de semana
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
            else:
                seasonality_tests['weekly'] = {'has_pattern': False, 'reason': 'Dados insuficientes'}

        # 3. CLASSIFICAÇÃO FINAL DA SAZONALIDADE
        hourly_strong = seasonality_tests.get('hourly', {}).get('variance_explained', 0) > 10
        daily_strong = seasonality_tests.get('daily', {}).get('variance_explained', 0) > 15
        weekly_strong = seasonality_tests.get('weekly', {}).get('variance_explained', 0) > 10

        hourly_significant = seasonality_tests.get('hourly', {}).get('has_pattern', False)
        daily_significant = seasonality_tests.get('daily', {}).get('has_pattern', False)
        weekly_significant = seasonality_tests.get('weekly', {}).get('has_pattern', False)

        # Lógica de classificação refinada com sazonalidade horária
        if hourly_significant and hourly_strong:
            main_seasonality = "SAZONAL_HORARIA"
            season_type = "horária (intra-hora)"
        elif daily_significant and daily_strong:
            if weekly_significant and weekly_strong:
                main_seasonality = "SAZONAL_MISTA"
                season_type = "mista (diária + semanal)"
            else:
                main_seasonality = "SAZONAL_DIARIA"
                season_type = "diária"
        elif weekly_significant and weekly_strong:
            main_seasonality = "SAZONAL_SEMANAL"
            season_type = "semanal"
        elif hourly_significant or daily_significant or weekly_significant:
            main_seasonality = "SAZONAL_FRACA"
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
        """Obtém padrões detalhados para visualização"""
        patterns = {}

        # Padrões por minuto (intra-hora)
        if temp_df['minute'].nunique() > 1:
            patterns['minute'] = temp_df.groupby('minute')['value'].agg(['mean', 'std', 'count'])

        # Padrões por hora
        if temp_df['hour'].nunique() > 1:
            patterns['hourly'] = temp_df.groupby('hour')['value'].agg(['mean', 'std', 'count'])

        # Padrões por dia da semana
        if temp_df['day_of_week'].nunique() > 1:
            patterns['weekly'] = temp_df.groupby('day_of_week')['value'].agg(['mean', 'std', 'count'])

        # Padrões por mês
        if temp_df.index.month.nunique() > 1:
            patterns['monthly'] = temp_df.groupby(temp_df.index.month)['value'].agg(['mean', 'std', 'count'])

        return patterns

    def seasonal_decomposition(self):
        try:
            freq_seconds = self.analysis_results['frequency']['seconds']
            n = len(self.df)

            # Determina o período baseado na análise de sazonalidade avançada
            seasonality_info = self.seasonality_results

            if seasonality_info['classification'] == 'SAZONAL_HORARIA':
                # Para sazonalidade horária (intra-hora), usar 60 períodos (minutos)
                if freq_seconds <= 60:  # Dados por minuto ou menos
                    period = 60
                elif freq_seconds <= 3600:  # Dados por hora
                    period = 24
                else:
                    period = max(2, n // 10)

            elif seasonality_info['classification'] == 'SAZONAL_DIARIA':
                # Para sazonalidade diária, usar 24 períodos (horas) ou equivalente
                if freq_seconds <= 3600:  # Dados por hora ou menos
                    period = 24
                elif freq_seconds <= 86400:  # Dados diários
                    period = 7  # Fallback para semanal
                else:
                    period = max(2, n // 10)

            elif seasonality_info['classification'] == 'SAZONAL_SEMANAL':
                # Para sazonalidade semanal, usar 7 períodos
                if freq_seconds <= 86400:  # Dados diários ou menos
                    period = 7
                else:
                    period = max(2, n // 10)

            elif seasonality_info['classification'] == 'SAZONAL_MISTA':
                # Para sazonalidade mista, priorizar o período menor (diário)
                if freq_seconds <= 3600:
                    period = 24
                else:
                    period = 7
            elif seasonality_info['classification'] == 'SAZONAL_FRACA':
                # Para sazonalidade fraca, usar período moderado
                period = max(2, min(n // 6, 12))
            else:
                # Para casos lineares ou sazonalidade fraca
                period = max(2, min(n // 4, 12))

            # Garante que o período não seja maior que metade dos dados
            period = min(period, n // 2)

            self.decomposition = seasonal_decompose(
                self.df['value'],
                model='additive',
                period=period,
                extrapolate_trend='freq'
            )

            # Análise da variância dos componentes
            total_var = np.var(self.df['value'])
            seasonal_var = np.var(self.decomposition.seasonal.dropna())
            trend_var = np.var(self.decomposition.trend.dropna())
            residual_var = np.var(self.decomposition.resid.dropna())

            seasonal_pct = (seasonal_var / total_var) * 100
            trend_pct = (trend_var / total_var) * 100
            residual_pct = (residual_var / total_var) * 100

            self.analysis_results['decomposition'] = {
                'period': period,
                'season_type': seasonality_info['season_type'],
                'total_variance': total_var,
                'seasonal_variance': seasonal_var,
                'trend_variance': trend_var,
                'residual_variance': residual_var,
                'seasonal_percentage': seasonal_pct,
                'trend_percentage': trend_pct,
                'residual_percentage': residual_pct,
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

        # Série original
        fig1 = px.line(self.df, x=self.df.index, y='value', title='Série Temporal Original')
        fig1.update_layout(xaxis_title='Tempo', yaxis_title='Valor', height=500)
        plots['original'] = fig1

        # Distribuição
        fig2 = make_subplots(rows=1, cols=2, subplot_titles=('Histograma', 'Box Plot'))
        fig2.add_trace(go.Histogram(x=self.df['value'], nbinsx=50, marker_color='skyblue'), row=1, col=1)
        fig2.add_trace(go.Box(y=self.df['value'], marker_color='lightcoral'), row=1, col=2)
        fig2.update_layout(title_text='Distribuição dos Valores', height=500)
        plots['distribution'] = fig2

        # Decomposição
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

        # Padrões sazonais avançados
        if hasattr(self, 'seasonality_results') and 'patterns' in self.seasonality_results:
            patterns = self.seasonality_results['patterns']

            # Criar subplots baseados nos padrões disponíveis
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
                            go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                     name='Média por Minuto', marker_color='lightblue'),
                            row=1, col=idx
                        )

                    elif pattern_type == 'hourly':
                        x_vals = list(range(24))
                        y_vals = [patterns['hourly'].loc[h, 'mean'] if h in patterns['hourly'].index else 0 for h in range(24)]
                        fig4.add_trace(
                            go.Scatter(x=x_vals, y=y_vals, mode='lines+markers',
                                     name='Média por Hora', marker_color='lightgreen'),
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

        # Detecção de saltos
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


def show_seasonality_results(seasonality_results):
    """Exibe os resultados detalhados da análise de sazonalidade"""

    col1, col2 = st.columns(2)

    classification = seasonality_results['classification']
    season_type = seasonality_results['season_type']

    # Cores baseadas na classificação
    color_map = {
        'SAZONAL_HORARIA': '🟣',
        'SAZONAL_DIARIA': '🟢',
        'SAZONAL_SEMANAL': '🔵',
        'SAZONAL_MISTA': '🟡',
        'SAZONAL_FRACA': '🟠',
        'LINEAR': '⚪'
    }

    col1.metric("Classificação", f"{color_map.get(classification, '❓')} {classification}")
    col2.metric("Tipo de Sazonalidade", season_type.capitalize())

    # Detalhes dos testes estatísticos
    st.subheader("📊 Testes Estatísticos de Sazonalidade")

    col1, col2, col3 = st.columns(3)

    # Teste horário (intra-hora)
    with col1:
        st.markdown("**⏰ Sazonalidade Horária (por minuto)**")
        hourly_test = seasonality_results.get('hourly_test', {})

        if hourly_test.get('has_pattern', False):
            st.success("✅ Padrão horário detectado!")
            st.write(f"• Variância explicada: {hourly_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {hourly_test.get('p_value', 0):.6f}")
            st.write(f"• Pico no minuto: {hourly_test.get('peak_minute', 'N/A')}")
            st.write(f"• Vale no minuto: {hourly_test.get('low_minute', 'N/A')}")
        else:
            st.info("ℹ️ Sem padrão horário significativo")
            if 'error' in hourly_test:
                st.error(f"Erro: {hourly_test['error']}")

    # Teste diário
    with col2:
        st.markdown("**🌅 Sazonalidade Diária (por hora)**")
        daily_test = seasonality_results.get('daily_test', {})

        if daily_test.get('has_pattern', False):
            st.success("✅ Padrão diário detectado!")
            st.write(f"• Variância explicada: {daily_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {daily_test.get('p_value', 0):.6f}")
            st.write(f"• Pico às: {daily_test.get('peak_hour', 'N/A')}h")
            st.write(f"• Vale às: {daily_test.get('low_hour', 'N/A')}h")
        else:
            st.info("ℹ️ Sem padrão diário significativo")
            if 'error' in daily_test:
                st.error(f"Erro: {daily_test['error']}")

    # Teste semanal
    with col3:
        st.markdown("**📅 Sazonalidade Semanal (por dia)**")
        weekly_test = seasonality_results.get('weekly_test', {})

        if weekly_test.get('has_pattern', False):
            st.success("✅ Padrão semanal detectado!")
            st.write(f"• Variância explicada: {weekly_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {weekly_test.get('p_value', 0):.6f}")

            days_map = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta',
                       4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
            peak_day = days_map.get(weekly_test.get('peak_day'), 'N/A')
            low_day = days_map.get(weekly_test.get('low_day'), 'N/A')

            st.write(f"• Pico: {peak_day}")
            st.write(f"• Vale: {low_day}")
            st.write(f"• Diferença úteis/fds: {weekly_test.get('weekday_weekend_diff', 0):.1f}%")
        else:
            st.info("ℹ️ Sem padrão semanal significativo")
            if 'error' in weekly_test:
                st.error(f"Erro: {weekly_test['error']}").get('low_day'), 'N/A')

            st.write(f"• Pico: {peak_day}")
            st.write(f"• Vale: {low_day}")
            st.write(f"• Diferença úteis/fds: {weekly_test.get('weekday_weekend_diff', 0):.1f}%")
        else:
            st.info("ℹ️ Sem padrão semanal significativo")
            if 'error' in weekly_test:
                st.error(f"Erro: {weekly_test['error']}")


def show_summary(analyzer, basic_stats, freq_desc, jump_count, seasonality_results):
    col1, col2, col3 = st.columns(3)

    classification = seasonality_results['classification']

    # Determina a variância sazonal total
    hourly_var = seasonality_results.get('hourly_test', {}).get('variance_explained', 0)
    daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
    weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)
    max_seasonal_var = max(hourly_var, daily_var, weekly_var)

    col1.metric(
        "Classificação",
        classification.replace('_', ' '),
        delta=f"{max_seasonal_var:.1f}% variância sazonal"
    )
    col2.metric("Frequência", freq_desc)

    col3.metric("Saltos Detectados", jump_count)


def show_basic_stats(basic_stats):
    col1, col2 = st.columns(2)

    stats_df = pd.DataFrame({
        'Estatística': ['Média', 'Mediana'],
        'Valor': [f"{basic_stats['Média']:.4f}", f"{basic_stats['Mediana']:.4f}"]
    })

    disp_df = pd.DataFrame({
        'Estatística': ['Desvio Padrão', 'Mínimo', 'Máximo'],
        'Valor': [f"{basic_stats['Desvio Padrão']:.4f}", f"{basic_stats['Mínimo']:.4f}", f"{basic_stats['Máximo']:.4f}"]
    })

    col1.subheader("Medidas de Tendência Central")
    col1.dataframe(stats_df, use_container_width=True)

    col2.subheader("Medidas de Dispersão")
    col2.dataframe(disp_df, use_container_width=True)

    col1, col2 = st.columns(2)
    col1.metric("Assimetria", f"{basic_stats['Assimetria']:.4f}")
    col2.metric("Curtose", f"{basic_stats['Curtose']:.4f}")


def show_decomposition_metrics(decomp_data):
    col1, col2, col3 = st.columns(3)
    col1.metric("Variância Sazonal", f"{decomp_data['seasonal_percentage']:.2f}%")
    col2.metric("Variância de Tendência", f"{decomp_data['trend_percentage']:.2f}%")
    col3.metric("Variância Residual", f"{decomp_data['residual_percentage']:.2f}%")


def show_jumps(jump_count, jumps):
    if jump_count > 0:
        st.warning(f"⚠️ Foram detectados {jump_count} saltos significativos na série!")
        if len(jumps) > 0:
            st.subheader("Maiores Saltos Detectados")
            sorted_jumps = jumps.abs().sort_values(ascending=False).head(5)
            jump_data = [{
                'Data': date.strftime('%Y-%m-%d %H:%M'),
                'Direção': "↑ Aumento" if jumps[date] > 0 else "↓ Diminuição",
                'Magnitude': f"{abs(jumps[date]):.4f}"
            } for date in sorted_jumps.index]
            st.dataframe(pd.DataFrame(jump_data), use_container_width=True)


def show_download(selected_file, seasonality_results, freq_desc, jump_count, basic_stats):
    classification = seasonality_results['classification']
    season_type = seasonality_results['season_type']

    # Calcula variância sazonal máxima
    hourly_var = seasonality_results.get('hourly_test', {}).get('variance_explained', 0)
    daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
    weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)

    results_summary = {
        'arquivo': selected_file,
        'classificacao': classification,
        'tipo_sazonalidade': season_type,
        'variancia_sazonal_horaria': hourly_var,
        'variancia_sazonal_diaria': daily_var,
        'variancia_sazonal_semanal': weekly_var,
        'frequencia': freq_desc,
        'saltos_detectados': jump_count,
        'media': basic_stats['Média'],
        'desvio_padrao': basic_stats['Desvio Padrão'],
    }

    results_df = pd.DataFrame([results_summary])
    csv_results = results_df.to_csv(index=False)

    st.download_button(
        label="📊 Download Resumo da Análise (CSV)",
        data=csv_results,
        file_name=f"analise_{selected_file.replace('.csv', '')}_resumo.csv",
        mime="text/csv"
    )

def extract_sigla_servico(file_name):
    try:
        sigla = file_name.split("sigla-")[1].split("_")[0] if "sigla-" in file_name else "N/A"
        servico = file_name.split("servicename-")[1].rsplit(".csv", 1)[0] if "servicename-" in file_name else "N/A"
        return sigla, servico
    except Exception:
        return "N/A", "N/A"

def batch_analyze_all(threshold_std: float = 3.0):
    series_folder = "series_temporais/series_manhatam/series_temporais_tier_0"
    all_results = []
    csv_files = [f for f in os.listdir(series_folder) if f.endswith(".csv")]

    progress_bar = st.progress(0)
    step = 100 / len(csv_files) if csv_files else 100
    status_text = st.empty()

    for idx, file_name in enumerate(csv_files, start=1):
        status_text.text(f"Analisando {file_name} ({idx}/{len(csv_files)})...")

        file_path = os.path.join(series_folder, file_name)
        analyzer = TimeSeriesAnalyzer(file_path)

        if not analyzer.load_data(show_message=False):
            continue

        # Executa todas as análises
        analyzer.basic_statistics()
        analyzer.detect_frequency()

        # Nova análise de sazonalidade avançada
        seasonality_results = analyzer.advanced_seasonality_detection()

        analyzer.seasonal_decomposition()
        analyzer.detect_jumps(threshold_std)

        # Coleta resultados
        decomp = analyzer.analysis_results.get("decomposition", {})
        stat = analyzer.analysis_results.get("stationarity", {})
        basic = analyzer.analysis_results.get("basic_stats", {})
        jumps = analyzer.analysis_results.get("jumps", {})

        # Análise de sazonalidade detalhada
        hourly_var = seasonality_results.get('hourly_test', {}).get('variance_explained', 0)
        daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
        weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)

        all_results.append({
            "sigla": extract_sigla_servico(file_name)[0],
            "servico": extract_sigla_servico(file_name)[1],
            "classificacao": seasonality_results.get('classification', 'N/A'),
            "variancia_sazonal_horaria": hourly_var,
            "variancia_sazonal_diaria": daily_var,
            "variancia_sazonal_semanal": weekly_var,
            "saltos_detectados": jumps.get("count", 0),
            "media": basic.get("Média", np.nan),
            "desvio_padrao": basic.get("Desvio Padrão", np.nan),
        })

        progress_bar.progress(min(int(idx * step), 100))

    progress_bar.empty()
    status_text.text("✅ Análise em lote concluída!")

    df_results = pd.DataFrame(all_results)
    classification_counts = df_results['classificacao'].value_counts().to_dict()

    return df_results, classification_counts


def main():
    st.title("📈 Análise Avançada de Séries Temporais com Detecção de Sazonalidade")
    st.markdown("**Sistema aprimorado para detectar padrões sazonais horários, diários e semanais**")
    st.markdown("---")

    st.sidebar.header("🔧 Configurações")

    series_folder = "series_temporais/series_manhatam/series_temporais_tier_0"

    if os.path.exists(series_folder):
        csv_files = [f for f in os.listdir(series_folder) if f.endswith('.csv')]

        if csv_files:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🚀 Processamento em Lote")

            # Slider para lote
            threshold_std_batch = st.sidebar.slider(
                "Threshold para detecção de saltos (desvios padrão) - Lote:",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="threshold_batch"
            )

            if st.sidebar.button("Analisar TODOS os arquivos", type="primary"):
                results_df, summary_counts = batch_analyze_all(threshold_std_batch)

                st.header("📑 Resumo Consolidado")
                st.dataframe(results_df, use_container_width=True)

                # Resumo geral com nova classificação
                st.subheader("📊 Resumo Geral de Classificação")
                for classification, count in summary_counts.items():
                    emoji_map = {
                        'SAZONAL_HORARIA': '⏰',
                        'SAZONAL_DIARIA': '🌅',
                        'SAZONAL_SEMANAL': '📅',
                        'SAZONAL_MISTA': '🔄',
                        'SAZONAL_FRACA': '🔸',
                        'LINEAR': '📈'
                    }
                    emoji = emoji_map.get(classification, '❓')
                    st.write(f"{emoji} {classification.replace('_', ' ')}: {count}")

                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Consolidado (CSV)",
                    csv_data, "analise_consolidada.csv", "text/csv"
                )

            # Análise individual
            selected_file = st.sidebar.selectbox(
                "Escolha uma série temporal:",
                csv_files,
                help="Selecione um arquivo CSV da pasta series_temporaiscpu"
            )

            file_path = os.path.join(series_folder, selected_file)

            st.sidebar.header("⚙️ Parâmetros de Análise")
            threshold_std_single = st.sidebar.slider(
                "Threshold para detecção de saltos (desvios padrão) - Individual:",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="threshold_single"
            )

            if st.sidebar.button("🚀 Executar Análise", type="primary"):
                analyzer = TimeSeriesAnalyzer(file_path)

                progress_bar = st.progress(0)
                status_text = st.empty()

                # Processo de análise com progresso
                status_text.text('Carregando dados...')
                progress_bar.progress(10)

                if analyzer.load_data():
                    status_text.text('Calculando estatísticas básicas...')
                    progress_bar.progress(20)
                    basic_stats = analyzer.basic_statistics()

                    status_text.text('Detectando frequência...')
                    progress_bar.progress(30)
                    freq_code, freq_desc = analyzer.detect_frequency()

                    seasonality_results = analyzer.advanced_seasonality_detection()

                    status_text.text('Realizando decomposição sazonal...')
                    progress_bar.progress(60)
                    analyzer.seasonal_decomposition()

                    status_text.text('Detectando saltos...')
                    progress_bar.progress(75)
                    jump_count, threshold, jumps = analyzer.detect_jumps(threshold_std_single)

                    status_text.text('Testando estacionariedade...')
                    progress_bar.progress(85)

                    status_text.text('Gerando visualizações...')
                    progress_bar.progress(95)
                    plots = analyzer.generate_plots()

                    progress_bar.progress(100)
                    status_text.text('✅ Análise concluída!')
                    progress_bar.empty()
                    status_text.empty()

                    # Exibição dos resultados
                    st.header("🎯 Resumo Executivo")
                    show_summary(analyzer, basic_stats, freq_desc, jump_count, seasonality_results)

                    # Nova seção de sazonalidade detalhada
                    show_seasonality_results(seasonality_results)

                    st.header("📊 Estatísticas Básicas")
                    show_basic_stats(basic_stats)

                    st.header("📈 Visualizações")
                    st.subheader("Série Temporal Original")
                    st.plotly_chart(plots['original'], use_container_width=True)

                    st.subheader("Análise da Distribuição")
                    st.plotly_chart(plots['distribution'], use_container_width=True)

                    if 'decomposition' in plots:
                        st.subheader("Decomposição Sazonal")
                        st.plotly_chart(plots['decomposition'], use_container_width=True)
                        show_decomposition_metrics(analyzer.analysis_results['decomposition'])

                    if 'patterns' in plots:
                        st.subheader("Padrões Sazonais")
                        st.plotly_chart(plots['patterns'], use_container_width=True)

                    st.subheader("Detecção de Saltos/Outliers")
                    st.plotly_chart(plots['jumps'], use_container_width=True)
                    show_jumps(jump_count, jumps)

                    st.header("💾 Download dos Resultados")
                    show_download(selected_file, seasonality_results, freq_desc,
                                jump_count, basic_stats)
        else:
            st.error("❌ Nenhum arquivo CSV encontrado na pasta 'series_temporaiscpu'")
    else:
        st.error("❌ Pasta 'series_temporaiscpu' não encontrada")
        st.info("💡 Certifique-se de que a pasta 'series_temporaiscpu' existe e contém arquivos CSV com colunas 'datetime' e 'value'")


if __name__ == "__main__":
    main()
