"""
Configurações centralizadas do sistema
"""

# Configurações do Datadog
DATADOG_URL = "https://api.proxy.datadog.prod.aws.cloud.ihf/api/v1/query"
DATADOG_HEADERS = {
    "Accept": "application/json",
    "DD-API-KEY": "org-itau-cc-prod",
    "DD-APPLICATION-KEY": "org-itau-cc-prod"
}

# Configurações de análise
DEFAULT_THRESHOLD_STD = 3.0
DEFAULT_DAYS_BACK = 90
DEFAULT_OUTPUT_FOLDER = "series_temporais"

# Configurações de interface
PAGE_CONFIG = {
    "page_title": "Análise de Séries Temporais",
    "page_icon": "📊",
    "layout": "wide"
}

# Mapeamento de emojis para classificações
EMOJI_MAP = {
    'SAZONAL_HORARIA': '🕐',
    'SAZONAL_DIARIA': '🌅',
    'SAZONAL_SEMANAL': '📅',
    'SAZONAL_MISTA': '🔄',
    'LINEAR': '📈',
}