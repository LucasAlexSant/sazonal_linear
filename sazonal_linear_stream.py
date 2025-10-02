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
        Detecta padrões de sazonalidade diária, semanal e mensal com testes estatísticos
        """
        temp_df = self.df.copy()
        temp_df['hour'] = temp_df.index.hour
        temp_df['day_of_week'] = temp_df.index.dayofweek
        temp_df['month'] = temp_df.index.month
        temp_df['is_weekend'] = temp_df['day_of_week'].isin([5, 6])
        temp_df['is_business_day'] = ~temp_df['is_weekend']
        
        seasonality_tests = {}
        
        # 1. TESTE DE SAZONALIDADE DIÁRIA (por hora)
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
        
        # 2. TESTE DE SAZONALIDADE SEMANAL (por dia da semana)
        if temp_df['day_of_week'].nunique() > 1:
            weekly_groups = [group['value'].values for name, group in temp_df.groupby('day_of_week') if len(group) >= 2]
            
            if len(weekly_groups) >= 3:
                try:
                    f_stat_weekly, p_value_weekly = f_oneway(*weekly_groups)
                    
                    weekly_means = temp_df.groupby('day_of_week')['value'].mean()
                    overall_mean = temp_df['value'].mean()
                    
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
            else:
                seasonality_tests['weekly'] = {'has_pattern': False, 'reason': 'Dados insuficientes'}

        # 3. TESTE DE SAZONALIDADE MENSAL (NOVO)
        if temp_df['month'].nunique() > 1:
            monthly_groups = [group['value'].values for name, group in temp_df.groupby('month') if len(group) >= 2]
            
            if len(monthly_groups) >= 3:
                try:
                    f_stat_monthly, p_value_monthly = f_oneway(*monthly_groups)
                    
                    monthly_means = temp_df.groupby('month')['value'].mean()
                    overall_mean = temp_df['value'].mean()
                    
                    temp_df_with_month_mean = temp_df.copy()
                    temp_df_with_month_mean['month_mean'] = temp_df_with_month_mean['month'].map(monthly_means)
                    ssb_monthly = ((temp_df_with_month_mean['month_mean'] - overall_mean) ** 2).sum()
                    
                    sst_monthly = ((temp_df['value'] - overall_mean) ** 2).sum()
                    
                    monthly_variance_explained = (ssb_monthly / sst_monthly) * 100 if sst_monthly > 0 else 0
                    
                    seasonality_tests['monthly'] = {
                        'has_pattern': p_value_monthly < 0.05,
                        'p_value': p_value_monthly,
                        'f_statistic': f_stat_monthly,
                        'variance_explained': monthly_variance_explained,
                        'peak_month': monthly_means.idxmax(),
                        'low_month': monthly_means.idxmin(),
                        'month_range': monthly_means.max() - monthly_means.min()
                    }
                    
                except Exception as e:
                    seasonality_tests['monthly'] = {'has_pattern': False, 'error': str(e)}
            else:
                seasonality_tests['monthly'] = {'has_pattern': False, 'reason': 'Dados insuficientes'}

        # 4. CLASSIFICAÇÃO FINAL DA SAZONALIDADE (ATUALIZADA)
        daily_strong = seasonality_tests.get('daily', {}).get('variance_explained', 0) > 15
        weekly_strong = seasonality_tests.get('weekly', {}).get('variance_explained', 0) > 10
        monthly_strong = seasonality_tests.get('monthly', {}).get('variance_explained', 0) > 10
        
        daily_significant = seasonality_tests.get('daily', {}).get('has_pattern', False)
        weekly_significant = seasonality_tests.get('weekly', {}).get('has_pattern', False)
        monthly_significant = seasonality_tests.get('monthly', {}).get('has_pattern', False)
        
        # Lógica de classificação refinada com sazonalidade mensal
        patterns_detected = []
        if daily_significant and daily_strong:
            patterns_detected.append('diária')
        if weekly_significant and weekly_strong:
            patterns_detected.append('semanal')
        if monthly_significant and monthly_strong:
            patterns_detected.append('mensal')
        
        if len(patterns_detected) >= 2:
            main_seasonality = "SAZONAL_MISTA"
            season_type = " + ".join(patterns_detected)
        elif 'diária' in patterns_detected:
            main_seasonality = "SAZONAL_DIARIA"
            season_type = "diária"
        elif 'semanal' in patterns_detected:
            main_seasonality = "SAZONAL_SEMANAL"
            season_type = "semanal"
        elif 'mensal' in patterns_detected:
            main_seasonality = "SAZONAL_MENSAL"
            season_type = "mensal"
        else:
            main_seasonality = "LINEAR"
            season_type = "não sazonal"
        
        self.seasonality_results = {
            'classification': main_seasonality,
            'season_type': season_type,
            'daily_test': seasonality_tests.get('daily', {}),
            'weekly_test': seasonality_tests.get('weekly', {}),
            'monthly_test': seasonality_tests.get('monthly', {}),
            'patterns': self.get_detailed_patterns(temp_df)
        }
        
        return self.seasonality_results

    def get_detailed_patterns(self, temp_df):
        """Obtém padrões detalhados para visualização"""
        patterns = {}
        
        if temp_df['hour'].nunique() > 1:
            patterns['hourly'] = temp_df.groupby('hour')['value'].agg(['mean', 'std', 'count'])
        
        if temp_df['day_of_week'].nunique() > 1:
            patterns['weekly'] = temp_df.groupby('day_of_week')['value'].agg(['mean', 'std', 'count'])
            
        if temp_df.index.month.nunique() > 1:
            patterns['monthly'] = temp_df.groupby(temp_df.index.month)['value'].agg(['mean', 'std', 'count'])
            
        return patterns

    def seasonal_decomposition(self):
        try:
            freq_seconds = self.analysis_results['frequency']['seconds']
            n = len(self.df)
            
            seasonality_info = self.seasonality_results
            
            if seasonality_info['classification'] == 'SAZONAL_DIARIA':
                if freq_seconds <= 3600:
                    period = 24
                elif freq_seconds <= 86400:
                    period = 7
                else:
                    period = max(2, n // 10)
                    
            elif seasonality_info['classification'] == 'SAZONAL_SEMANAL':
                if freq_seconds <= 86400:
                    period = 7
                else:
                    period = max(2, n // 10)
                    
            elif seasonality_info['classification'] == 'SAZONAL_MENSAL':
                if freq_seconds <= 86400:
                    period = 30
                else:
                    period = 12
                    
            elif seasonality_info['classification'] == 'SAZONAL_MISTA':
                if freq_seconds <= 3600:
                    period = 24
                elif freq_seconds <= 86400:
                    period = 7
                else:
                    period = 12
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
            
            available_patterns = [k for k in ['hourly', 'weekly', 'monthly'] if k in patterns and len(patterns[k]) > 0]
            
            if available_patterns:
                n_plots = len(available_patterns)
                titles = {
                    'hourly': 'Por Hora',
                    'weekly': 'Por Dia da Semana',
                    'monthly': 'Por Mês'
                }
                subplot_titles = [titles[k] for k in available_patterns]
                
                fig4 = make_subplots(rows=1, cols=n_plots, subplot_titles=subplot_titles)
                
                for idx, pattern_type in enumerate(available_patterns, 1):
                    if pattern_type == 'hourly':
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
                    
                    elif pattern_type == 'monthly':
                        months = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                                 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
                        y_vals = [patterns['monthly'].loc[m, 'mean'] if m in patterns['monthly'].index else 0 for m in range(1, 13)]
                        fig4.add_trace(
                            go.Bar(x=months, y=y_vals, name='Média por Mês', marker_color='lightskyblue'),
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


def show_seasonality_results(seasonality_results):
    """Exibe os resultados detalhados da análise de sazonalidade"""
    
    col1, col2 = st.columns(2)
    
    classification = seasonality_results['classification']
    season_type = seasonality_results['season_type']
    
    color_map = {
        'SAZONAL_DIARIA': '🟢',
        'SAZONAL_SEMANAL': '🔵', 
        'SAZONAL_MENSAL': '🟣',
        'SAZONAL_MISTA': '🟡',
        'LINEAR': '⚪'
    }
    
    col1.metric("Classificação", f"{color_map.get(classification, '❓')} {classification}")
    col2.metric("Tipo de Sazonalidade", season_type.capitalize())
    
    st.subheader("📊 Testes Estatísticos de Sazonalidade")
    
    col1, col2, col3 = st.columns(3)
    
    # Teste diário
    with col1:
        st.markdown("**🌅 Sazonalidade Diária**")
        daily_test = seasonality_results.get('daily_test', {})
        
        if daily_test.get('has_pattern', False):
            st.success("✅ Padrão diário detectado!")
            st.write(f"• Variância: {daily_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {daily_test.get('p_value', 0):.6f}")
            st.write(f"• Pico: {daily_test.get('peak_hour', 'N/A')}h")
            st.write(f"• Vale: {daily_test.get('low_hour', 'N/A')}h")
        else:
            st.info("ℹ️ Sem padrão diário")
    
    # Teste semanal  
    with col2:
        st.markdown("**📅 Sazonalidade Semanal**")
        weekly_test = seasonality_results.get('weekly_test', {})
        
        if weekly_test.get('has_pattern', False):
            st.success("✅ Padrão semanal detectado!")
            st.write(f"• Variância: {weekly_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {weekly_test.get('p_value', 0):.6f}")
            
            days_map = {0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui', 
                       4: 'Sex', 5: 'Sáb', 6: 'Dom'}
            peak_day = days_map.get(weekly_test.get('peak_day'), 'N/A')
            low_day = days_map.get(weekly_test.get('low_day'), 'N/A')
            
            st.write(f"• Pico: {peak_day}")
            st.write(f"• Vale: {low_day}")
        else:
            st.info("ℹ️ Sem padrão semanal")
    
    # Teste mensal (NOVO)
    with col3:
        st.markdown("**📆 Sazonalidade Mensal**")
        monthly_test = seasonality_results.get('monthly_test', {})
        
        if monthly_test.get('has_pattern', False):
            st.success("✅ Padrão mensal detectado!")
            st.write(f"• Variância: {monthly_test.get('variance_explained', 0):.1f}%")
            st.write(f"• P-valor: {monthly_test.get('p_value', 0):.6f}")
            
            months_map = {1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
                         7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'}
            peak_month = months_map.get(monthly_test.get('peak_month'), 'N/A')
            low_month = months_map.get(monthly_test.get('low_month'), 'N/A')
            
            st.write(f"• Pico: {peak_month}")
            st.write(f"• Vale: {low_month}")
        else:
            st.info("ℹ️ Sem padrão mensal")


def show_summary(analyzer, basic_stats, freq_desc, jump_count, seasonality_results):
    col1, col2, col3 = st.columns(3)
    
    classification = seasonality_results['classification']
    
    daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
    weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)
    monthly_var = seasonality_results.get('monthly_test', {}).get('variance_explained', 0)
    max_seasonal_var = max(daily_var, weekly_var, monthly_var)
    
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
    
    daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
    weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)
    monthly_var = seasonality_results.get('monthly_test', {}).get('variance_explained', 0)
    
    results_summary = {
        'arquivo': selected_file,
        'classificacao': classification,
        'tipo_sazonalidade': season_type,
        'variancia_sazonal_diaria': daily_var,
        'variancia_sazonal_semanal': weekly_var,
        'variancia_sazonal_mensal': monthly_var,
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
            
        analyzer.basic_statistics()
        analyzer.detect_frequency()
        
        seasonality_results = analyzer.advanced_seasonality_detection()
        
        analyzer.seasonal_decomposition()
        analyzer.detect_jumps(threshold_std)
        
        decomp = analyzer.analysis_results.get("decomposition", {})
        basic = analyzer.analysis_results.get("basic_stats", {})
        jumps = analyzer.analysis_results.get("jumps", {})
        
        daily_var = seasonality_results.get('daily_test', {}).get('variance_explained', 0)
        weekly_var = seasonality_results.get('weekly_test', {}).get('variance_explained', 0)
        monthly_var = seasonality_results.get('monthly_test', {}).get('variance_explained', 0)
        
        all_results.append({
            "sigla": extract_sigla_servico(file_name)[0],
            "servico": extract_sigla_servico(file_name)[1],  
            "classificacao": seasonality_results.get('classification', 'N/A'),
            "variancia_sazonal_diaria": daily_var,
            "variancia_sazonal_semanal": weekly_var,
            "variancia_sazonal_mensal": monthly_var,
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
    st.markdown("**Sistema aprimorado para detectar padrões sazonais diários, semanais e mensais**")
    st.markdown("---")
    
    st.sidebar.header("🔧 Configurações")
    
    series_folder = "series_temporais/series_manhatam/series_temporais_tier_0"
    
    if os.path.exists(series_folder):
        csv_files = [f for f in os.listdir(series_folder) if f.endswith('.csv')]
        
        if csv_files:
            st.sidebar.markdown("---")
            st.sidebar.subheader("🚀 Processamento em Lote")
            
            threshold_std_batch = st.sidebar.slider(
                "Threshold para detecção de saltos (desvios padrão) - Lote:",
                min_value=1.0, max_value=5.0, value=3.0, step=0.1, key="threshold_batch"
            )
            
            if st.sidebar.button("Analisar TODOS os arquivos", type="primary"):
                results_df, summary_counts = batch_analyze_all(threshold_std_batch)
                
                st.header("📑 Resumo Consolidado")
                st.dataframe(results_df, use_container_width=True)
                
                st.subheader("📊 Resumo Geral de Classificação")
                for classification, count in summary_counts.items():
                    emoji_map = {
                        'SAZONAL_DIARIA': '🌅',
                        'SAZONAL_SEMANAL': '📅', 
                        'SAZONAL_MENSAL': '🟣',
                        'SAZONAL_MISTA': '🔄',
                        'LINEAR': '📈'
                    }
                    emoji = emoji_map.get(classification, '❓')
                    st.write(f"{emoji} {classification.replace('_', ' ')}: {count}")
                
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Consolidado (CSV)",
                    csv_data, "analise_consolidada.csv", "text/csv"
                )
            
            selected_file = st.sidebar.selectbox(
                "Escolha uma série temporal:",
                csv_files,
                help="Selecione um arquivo CSV da pasta series_temporais"
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
                
                status_text.text('Carregando dados...')
                progress_bar.progress(10)
                
                if analyzer.load_data():
                    status_text.text('Calculando estatísticas básicas...')
                    progress_bar.progress(20)
                    basic_stats = analyzer.basic_statistics()
                    
                    status_text.text('Detectando frequência...')
                    progress_bar.progress(30)
                    freq_code, freq_desc = analyzer.detect_frequency()
                    
                    status_text.text('Analisando sazonalidade...')
                    progress_bar.progress(50)
                    seasonality_results = analyzer.advanced_seasonality_detection()
                    
                    status_text.text('Realizando decomposição sazonal...')
                    progress_bar.progress(60)
                    analyzer.seasonal_decomposition()
                    
                    status_text.text('Detectando saltos...')
                    progress_bar.progress(75)
                    jump_count, threshold, jumps = analyzer.detect_jumps(threshold_std_single)
                    
                    status_text.text('Gerando visualizações...')
                    progress_bar.progress(95)
                    plots = analyzer.generate_plots()
                    
                    progress_bar.progress(100)
                    status_text.text('✅ Análise concluída!')
                    progress_bar.empty()
                    status_text.empty()
                    
                    st.header("🎯 Resumo Executivo")
                    show_summary(analyzer, basic_stats, freq_desc, jump_count, seasonality_results)
                    
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
            st.error("❌ Nenhum arquivo CSV encontrado na pasta")
    else:
        st.error("❌ Pasta não encontrada")
        st.info("💡 Certifique-se de que a pasta existe e contém arquivos CSV")


if __name__ == "__main__":
    main()
