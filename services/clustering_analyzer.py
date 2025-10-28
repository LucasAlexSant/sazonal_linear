"""
Serviço para análise de clustering de séries temporais
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ClusteringAnalyzer:
    """Classe responsável por agrupar séries temporais em clusters"""
    
    def __init__(self, df):
        """
        Inicializa o analisador de clustering
        
        Args:
            df: DataFrame com os resultados da análise em lote
        """
        self.df = df.copy()
        self.df_encoded = None
        self.cluster_labels = None
        self.n_clusters = None
        self.silhouette_avg = None
        self.encoders = {}
        
    def prepare_data(self, columns_to_cluster):
        """
        Prepara os dados para clustering, fazendo encoding e normalização
        
        Args:
            columns_to_cluster: Lista de colunas a serem usadas no clustering
        """
        df_cluster = self.df[columns_to_cluster].copy()
        
        # Separar colunas categóricas e numéricas
        categorical_cols = df_cluster.select_dtypes(include=['object', 'bool']).columns
        numerical_cols = df_cluster.select_dtypes(include=['number']).columns
        
        # Fazer encoding das categóricas
        encoded_data = df_cluster.copy()
        for col in categorical_cols:
            le = LabelEncoder()
            encoded_data[col] = le.fit_transform(df_cluster[col].astype(str))
            self.encoders[col] = le
        
        # Normalizar todas as colunas
        scaler = StandardScaler()
        self.df_encoded = pd.DataFrame(
            scaler.fit_transform(encoded_data),
            columns=encoded_data.columns,
            index=encoded_data.index
        )
        
        return self.df_encoded
    
    def find_optimal_clusters(self, max_clusters=10):
        """
        Encontra o número ótimo de clusters usando método do cotovelo e silhouette
        
        Args:
            max_clusters: Número máximo de clusters a testar
            
        Returns:
            tuple: (número ótimo de clusters, scores)
        """
        n_samples = len(self.df_encoded)
        max_clusters = min(max_clusters, n_samples - 1)
        
        if max_clusters < 2:
            return 2, {}
        
        inertias = []
        silhouette_scores = []
        K_range = range(2, max_clusters + 1)
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(self.df_encoded)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_encoded, kmeans.labels_))
        
        # Encontrar o melhor k baseado no silhouette score
        best_k_idx = np.argmax(silhouette_scores)
        optimal_k = list(K_range)[best_k_idx]
        
        scores = {
            'k_range': list(K_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k, scores
    
    def perform_clustering(self, n_clusters=None):
        """
        Executa o clustering K-Means
        
        Args:
            n_clusters: Número de clusters (None para determinar automaticamente)
            
        Returns:
            tuple: (cluster_labels, silhouette_score, cluster_info)
        """
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters()
        
        self.n_clusters = n_clusters
        
        # Executar K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.df_encoded)
        
        # Calcular silhouette score
        if len(np.unique(self.cluster_labels)) > 1:
            self.silhouette_avg = silhouette_score(self.df_encoded, self.cluster_labels)
        else:
            self.silhouette_avg = 0
        
        # Adicionar labels ao DataFrame original
        self.df['cluster'] = self.cluster_labels
        
        # Gerar informações dos clusters
        cluster_info = self._generate_cluster_info()
        
        return self.cluster_labels, self.silhouette_avg, cluster_info
    
    def _generate_cluster_info(self):
        """Gera informações descritivas sobre cada cluster"""
        cluster_info = []
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.df['cluster'] == cluster_id
            cluster_data = self.df[cluster_mask]
            
            info = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'characteristics': self._describe_cluster(cluster_data)
            }
            
            cluster_info.append(info)
        
        return cluster_info
    
    def _describe_cluster(self, cluster_data):
        """Gera descrição textual das características do cluster"""
        characteristics = {}
        
        # Características categóricas (moda)
        categorical_cols = ['classificacao', 'tipo_sazonalidade', 'algoritmo_anomalia']
        for col in categorical_cols:
            if col in cluster_data.columns:
                mode_value = cluster_data[col].mode()
                if len(mode_value) > 0:
                    characteristics[col] = mode_value.iloc[0]
        
        # Características numéricas (média)
        if 'saltos_detectados' in cluster_data.columns:
            characteristics['saltos_detectados_media'] = cluster_data['saltos_detectados'].mean()
            characteristics['saltos_detectados_max'] = cluster_data['saltos_detectados'].max()
        
        if 'media' in cluster_data.columns:
            characteristics['valor_medio'] = cluster_data['media'].mean()
        
        return characteristics
    
    def create_cluster_visualization(self):
        """
        Cria visualizações dos clusters
        
        Returns:
            dict: Dicionário com os gráficos plotly
        """
        plots = {}
        
        # 1. Gráfico de distribuição de clusters
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        fig1 = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            labels={'x': 'Cluster', 'y': 'Número de Métricas'},
            title='Distribuição de Métricas por Cluster',
            color=cluster_counts.values,
            color_continuous_scale='Viridis'
        )
        fig1.update_layout(showlegend=False, height=400)
        plots['distribution'] = fig1
        
        # 2. Composição dos clusters por características
        fig2 = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Classificação por Cluster',
                'Algoritmo de Anomalia por Cluster',
                'Tipo de Sazonalidade por Cluster',
                'Saltos Detectados por Cluster'
            ),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'box'}]]
        )
        
        # Classificação
        if 'classificacao' in self.df.columns:
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                class_counts = cluster_data['classificacao'].value_counts()
                fig2.add_trace(
                    go.Bar(name=f'Cluster {cluster_id}', x=class_counts.index, y=class_counts.values),
                    row=1, col=1
                )
        
        # Algoritmo de Anomalia
        if 'algoritmo_anomalia' in self.df.columns:
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                algo_counts = cluster_data['algoritmo_anomalia'].value_counts()
                fig2.add_trace(
                    go.Bar(name=f'Cluster {cluster_id}', x=algo_counts.index, y=algo_counts.values, showlegend=False),
                    row=1, col=2
                )
        
        # Tipo de Sazonalidade
        if 'tipo_sazonalidade' in self.df.columns:
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                season_counts = cluster_data['tipo_sazonalidade'].value_counts()
                fig2.add_trace(
                    go.Bar(name=f'Cluster {cluster_id}', x=season_counts.index, y=season_counts.values, showlegend=False),
                    row=2, col=1
                )
        
        # Saltos Detectados
        if 'saltos_detectados' in self.df.columns:
            for cluster_id in sorted(self.df['cluster'].unique()):
                cluster_data = self.df[self.df['cluster'] == cluster_id]
                fig2.add_trace(
                    go.Box(name=f'Cluster {cluster_id}', y=cluster_data['saltos_detectados'], showlegend=False),
                    row=2, col=2
                )
        
        fig2.update_layout(height=800, title_text="Características dos Clusters", barmode='group')
        plots['characteristics'] = fig2
        
        # 3. Scatter plot 2D (se possível fazer PCA ou usar 2 features)
        if len(self.df_encoded.columns) >= 2:
            # Usar as 2 primeiras componentes normalizadas
            fig3 = px.scatter(
                self.df,
                x=self.df_encoded.iloc[:, 0],
                y=self.df_encoded.iloc[:, 1],
                color=self.df['cluster'].astype(str),
                hover_data=['metrica'] if 'metrica' in self.df.columns else None,
                labels={'color': 'Cluster', 'x': 'Componente 1', 'y': 'Componente 2'},
                title='Visualização dos Clusters (2D)',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig3.update_layout(height=600)
            plots['scatter'] = fig3
        
        return plots
    
    def generate_cluster_summary(self):
        """
        Gera um resumo textual dos clusters
        
        Returns:
            dict: Resumo dos clusters
        """
        summary = {
            'n_clusters': self.n_clusters,
            'silhouette_score': round(self.silhouette_avg, 3),
            'total_metrics': len(self.df),
            'clusters': []
        }
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = self.df['cluster'] == cluster_id
            cluster_data = self.df[cluster_mask]
            
            cluster_summary = {
                'id': cluster_id,
                'size': len(cluster_data),
                'percentage': round((len(cluster_data) / len(self.df)) * 100, 1),
                'description': self._generate_cluster_description(cluster_data)
            }
            
            summary['clusters'].append(cluster_summary)
        
        return summary
    
    def _generate_cluster_description(self, cluster_data):
        """Gera descrição em texto natural do cluster"""
        desc_parts = []
        
        # Classificação predominante
        if 'classificacao' in cluster_data.columns:
            top_class = cluster_data['classificacao'].mode()
            if len(top_class) > 0:
                class_pct = (cluster_data['classificacao'] == top_class.iloc[0]).sum() / len(cluster_data) * 100
                desc_parts.append(f"{int(class_pct)}% {top_class.iloc[0]}")
        
        # Algoritmo predominante
        if 'algoritmo_anomalia' in cluster_data.columns:
            top_algo = cluster_data['algoritmo_anomalia'].mode()
            if len(top_algo) > 0:
                algo_pct = (cluster_data['algoritmo_anomalia'] == top_algo.iloc[0]).sum() / len(cluster_data) * 100
                if algo_pct >= 70:  # Só mencionar se for predominante
                    desc_parts.append(f"algoritmo {top_algo.iloc[0]}")
        
        # Saltos
        if 'saltos_detectados' in cluster_data.columns:
            avg_jumps = cluster_data['saltos_detectados'].mean()
            if avg_jumps > 5:
                desc_parts.append(f"alto número de saltos (média: {avg_jumps:.1f})")
            elif avg_jumps < 1:
                desc_parts.append("poucos saltos")
        
        return ", ".join(desc_parts) if desc_parts else "Características mistas"