#!/usr/bin/env python3
# Application configuration settings
from typing import Dict, Any

class Config:
    """Application configuration constants."""
    
    # Timeout settings for preventing freezes
    TIMEOUTS = {
        'api': 10,  # Gemini API timeout
        'pubchem_3d': 10,  # PubChem 3D record fetch timeout
    }
    
    # Random sample configuration
    RANDOM_QUERY = {
        'count': 30,  # Number of random samples to display
        'columns': 2,  # Number of columns for random samples
    }
    
    # Cache configuration
    CACHE = {
        'enabled': False,  # Enable/disable cache functionality (can be overridden by secrets.toml)
        'base_directory': 'cache',  # Base cache directory name
        'max_size_mb': 100,  # Maximum cache size in MB
        'max_age_days': 360,  # Maximum age of cache entries in days
        'data_sources': {
            'pubchem': {
                'enabled': True,
                'directory': 'pubchem',
                'max_age_days': 36500,
            },
            'queries': {
                'enabled': True,
                'directory': 'queries',
                'max_age_days': 36500,
                'max_items_per_file': 25,
            },
            'descriptions': {
                'enabled': True,
                'directory': 'descriptions',
                'max_age_days': 36500,
                'max_items_per_file': 25,
            },
            'analysis': {
                'enabled': True,
                'directory': 'analysis',
                'max_age_days': 180,
                'max_items_per_file': 25,
            },
            'similar': {
                'enabled': True,
                'directory': 'similar',
                'max_age_days': 180,  #
                'max_items_per_file': 50,
                'max_items_per_data': 25,
            },
            'failed_molecules': {
                'enabled': True,
                'directory': 'failed_molecules',
                'max_age_days': 365,
                'max_items_per_file': 1000,
            },
        }
    }
        
    # 3D Molecular Viewer Configuration
    # Responsive viewer size based on window size
    VIEWER = {
        'width_pc': 700,
        'height_pc': 450,
        'width_mobile': 340,
        'height_mobile': 180,
        'zoom_min': 0.1,
        'zoom_max': 50,
        'rotation_speed': 1,
    }
    
    # Default AI Model Configuration
    DEFAULT_MODEL_NAME = "gemini-2.5-flash-lite"
    
    # Gemini API Configuration by Query Type
    GEMINI_CONFIG = {
        'molecular_search': {
            'temperature': 1.0,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1024,
            'use_google_search': True,
            'timeout': 15
        },
        'molecular_analysis': {
            'temperature': 0.2,
            'top_p': 0.8,
            'top_k': 20,
            'max_output_tokens': 2048,
            'use_google_search': False,
            'timeout': 10
        },
        'molecular_search': {
            'temperature': 1.0,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 1024,
            'use_google_search': True,
            'timeout': 15
        },
    }
    
    # Error messages - simplified to essential ones only
    ERROR_MESSAGES = {
        # API related errors
        'api_error': "API接続エラーが発生しました。しばらく待ってから再試行してください。",
        'timeout': "操作がタイムアウトしました。",
        
        # Data retrieval errors
        'molecule_not_found': "分子データが見つかりませんでした。",
        'invalid_data': "無効なデータが返されました。",
        
        # Molecular processing errors
        'processing_error': "分子データの処理中にエラーが発生しました。",
        
        # General errors
        'parse_error': "データの解析に失敗しました。",
        'display_error': "表示中にエラーが発生しました。",
        'no_data': "データが見つかりません。最初からやり直してください。",
        'general_error': "予期しないエラーが発生しました。",
    }

# Announcement Configuration
ANNOUNCEMENT_MESSAGE: str = """

アプリを作っている **[山本 典史](https://yamlab.jp)** です、こんにちは。

大学教員。専門は計算化学。化学の学びを身近にすることにも興味を持っています。

[![Image from Gyazo](https://i.gyazo.com/9dcdd96f968748d29c1667516660ad55.png)](https://yamlab.jp)

このアプリの開発をご支援いただける方は、**[こちら](https://buymeacoffee.com/yamnor)** からお願いします！

![](https://i.gyazo.com/602d3779cdf830009e0c5f7dcc9d6d63.png)
"""

MENU_ITEMS_ABOUT: str = '''
**ChatMOL** was created by [yamnor](https://yamnor.me),
a chemist 🧪 specializing in molecular simulation 🖥️ living in Japan 🇯🇵.

If you have any questions, thoughts, or comments,
feel free to [contact me](https://letterbird.co/yamnor) ✉️
or find me on [X (Twitter)](https://x.com/yamnor) 🐦.

GitHub: [yamnor/chatmol](https://github.com/yamnor/chatmol)
'''
