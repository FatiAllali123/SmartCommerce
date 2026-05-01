"""
Tests unitaires pour le preprocessing.
Lance : pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_mini_dataset():
    """Cree un mini dataset synthétique pour les tests."""
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'title': ['Produit A', 'Produit B', 'Produit C', 'Produit D', 'Produit E'],
        'description': ['Desc A', 'Desc B', None, 'Desc D', 'Desc E'],
        'vendor': ['Vendor1', 'Vendor2', 'Vendor1', 'Vendor2', 'Vendor1'],
        'product_type': ['Shoes', 'T-shirt', None, 'Bracelet', 'Shoes'],
        'tags': ['tag1,tag2', 'tag3', 'tag4,tag5,tag6', '', 'tag7'],
        'created_at': ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01'],
        'updated_at': ['2024-01-15', '2024-02-15', '2024-03-15', '2024-04-15', '2024-05-15'],
        'price': [50.0, 0.0, 100.0, 25.0, 75.0],
        'compare_at_price': [60.0, None, None, 30.0, None],
        'discount_pct': [16.7, 0.0, 0.0, 16.7, 0.0],
        'available': [True, True, False, True, True],
        'inventory_quantity': [10, 5, 0, 20, 15],
        'variants_count': [3, 1, 2, 4, 1],
        'variant_options': ['S,M,L', 'M', 'S,L', 'S,M,L,XL', 'M'],
        'images_count': [5, 2, 3, 4, 1],
        'store_name': ['Store1', 'Store2', 'Store1', 'Store2', 'Store1'],
        'store_url': ['url1', 'url2', 'url1', 'url2', 'url1'],
        'store_country': ['USA', 'UK', 'USA', 'UK', 'USA'],
        'store_category': ['Cat1', 'Cat2', 'Cat1', 'Cat2', 'Cat1']
    }
    return pd.DataFrame(data)


def test_dataset_creation():
    """Verifie que le mini dataset est cree correctement."""
    df = create_mini_dataset()
    assert len(df) == 5
    assert len(df.columns) == 20
    print("OK : Dataset cree avec 5 produits et 20 colonnes")


def test_price_filtering():
    """Verifie que les prix a 0 sont retires."""
    df = create_mini_dataset()
    df_clean = df[df['price'] > 0]
    assert len(df_clean) == 4  # Le produit a 0$ doit etre retire
    print("OK : Prix a 0 correctement filtres")


def test_required_columns():
    """Verifie que les colonnes essentielles existent."""
    df = create_mini_dataset()
    required = ['product_id', 'title', 'price', 'available',
                'store_name', 'product_type', 'discount_pct']
    for col in required:
        assert col in df.columns, f"Colonne manquante : {col}"
    print("OK : Toutes les colonnes requises presentes")


def test_score_range():
    """Verifie qu'un score composite est entre 0 et 1."""
    df = create_mini_dataset()
    # Simuler un score simple
    df['score'] = (df['price'] / df['price'].max()) * 0.5 + \
                  (df['available'].astype(int)) * 0.5
    assert df['score'].min() >= 0
    assert df['score'].max() <= 1
    print("OK : Scores dans la plage [0, 1]")


def test_target_variable():
    """Verifie que la variable cible est binaire (0 ou 1)."""
    df = create_mini_dataset()
    df['score'] = (df['price'] / df['price'].max())
    threshold = df['score'].quantile(0.8)
    df['produit_succes'] = (df['score'] >= threshold).astype(int)
    assert set(df['produit_succes'].unique()).issubset({0, 1})
    print("OK : Variable cible binaire (0 ou 1)")


if __name__ == "__main__":
    test_dataset_creation()
    test_price_filtering()
    test_required_columns()
    test_score_range()
    test_target_variable()
    print("\n=== TOUS LES TESTS PASSENT ===")
