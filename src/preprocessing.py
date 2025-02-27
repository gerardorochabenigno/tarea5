"""
Módulo para el preprocesamiento del modelo.

Este módulo contiene funciones para cargar los datos, entrenar un modelo y guardarlo.
"""

# src/preprocessing.py
import pandas as pd
import os
import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from deep_translator import GoogleTranslator
import numpy as np

logger = logging.getLogger(__name__)


# Función para reducir el número de categorías de items
def reduce_categories(df_item_categories: pd.DataFrame) -> pd.DataFrame:
    """
    Traduce y reduce las categorías de items agrupándolas por tipo principal.
    Args:
        df_item_categories (pd.DataFrame): DataFrame con las categorías de items.
    Returns:
        pd.DataFrame: DataFrame con las categorías de items reducidas.
    """
    try:
        # Inicializamos el traductor
        translator = GoogleTranslator(source='ru', target='en')
    except Exception as e:
        logging.error(f"Error al importar GoogleTranslator: {e}")

    try:
        # Creamos una copia para no modificar el original
        df_item_categories_copy = df_item_categories.copy()
        
        # Traducimos los campos
        df_item_categories_copy["item_cat_name_t"] = df_item_categories_copy["item_category_name"].apply(lambda x: translator.translate(x))

        # Pasamos a minúsculas
        df_item_categories_copy["item_cat_name_t"] = df_item_categories_copy["item_cat_name_t"].str.lower()

        # Vamos a clasificar las categorías
        # Accessories : 1
        # Consoles : 2
        # Games : 3
        # Books : 4
        # Cinema : 5
        # Gifts : 6
        # Payment cards : 7
        # Programs : 8
        # Music : 9
        # Others : 10

        # Secuencialmente asignamos las categorías principales, ya que el traductor no es perfecto
        # Identificamos si el valor contiene "accesories", "headsets/headphones"
        df_item_categories_copy["main_category_id"] = df_item_categories_copy["item_cat_name_t"].apply(lambda x: 1 if "accessories" in x or "headsets/headphones" in x else 
                                                        2 if "game consoles" in x or "game consoles accessories" in x else 
                                                        3 if "games" in x or "games accessories" in x else 
                                                        4 if "books" in x else 
                                                        5 if "cinema" in x else 
                                                        6 if "gifts" in x else 
                                                        7 if "payment" in  x else 
                                                        8 if "programs" in x else 
                                                        9 if "music" in x else 
                                                        10)
        
        # Asignamos el nombre de la categoría principal
        df_item_categories_copy["main_category"] = df_item_categories_copy["main_category_id"].apply(lambda x: "Accessories" if x == 1 else 
                                                        "Consolas" if x == 2 else 
                                                        "Videojuegos" if x == 3 else 
                                                        "Libros" if x == 4 else 
                                                        "Películas" if x == 5 else 
                                                        "Regalos" if x == 6 else 
                                                        "Tarjetas de prepago" if x == 7 else 
                                                        "Programas" if x == 8 else 
                                                        "Música" if x == 9 else 
                                                        "Otros")
        logger.info("Categorías reducidas correctamente")
        return df_item_categories_copy
    
    
    except Exception as e:
        logger.error(f"Error al reducir las categorías: {e}")
        raise e

# Función para pegar la información de items y item_categories
def paste_item_info(series_data: pd.DataFrame, items_data: pd.DataFrame, item_categories_data: pd.DataFrame) -> pd.DataFrame: 
    """
    Preprocesa los datos crudos pegando los datos de items y item_categories.

    Args:
        series_data (pd.DataFrame): Dataframe con los datos crudos.
        items_data (pd.DataFrame): Dataframe con los datos de items.
        item_categories_data (pd.DataFrame): Dataframe con los datos de item_categories.

    Returns:
        pd.DataFrame: Dataframe con los datos preprocesados.
    """
    
    try:
        # Copiamos el dataframe para no modificar los originales
        series_data_copy = series_data.copy()
        
        # Pegamos los datos de items y item_categories
        series_data_copy = pd.merge(series_data_copy, items_data, on='item_id', how='left')
        series_data_copy = pd.merge(series_data_copy, item_categories_data, on='item_category_id', how='left')
        
        logger.info("Información de items e item_categories pegada correctamente")
        return series_data_copy

    
    except Exception as e:
        logger.error(f"Error al pegar los datos de items y item_categories: {e}")
        raise e

# Función para cambiar el formato de fechas
def change_date_format(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Cambia el formato de las fechas a datetime.
    Args:
        daily_data (pd.DataFrame): Dataframe con los datos crudos.

    Returns:
        pd.DataFrame: Dataframe con los datos preprocesados.
    """
    try:
        # Copiamos el dataframe para no modificar los originales
        daily_data_copy = daily_data.copy()

        # Cambiamos el formato de las fechas
        daily_data_copy['date'] = pd.to_datetime(daily_data_copy['date'], format='%d.%m.%Y')
        logger.info("Fechas cambiadas de formato correctamente")
        return daily_data_copy
    
    except Exception as e:
        logger.error(f"Error al cambiar el formato de las fechas: {e}")
        raise e

def create_week_variable(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable de semana.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy['week'] = daily_data_copy['date'].dt.isocalendar().week
        logger.info("Variable de semana creada correctamente")
        return daily_data_copy
    
    except Exception as e:
        logger.error(f"Error al crear la variable de semana: {e}")
        raise e

def create_year_variable(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
        Crea la variable de año.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy['year'] = daily_data_copy['date'].dt.year
        daily_data_copy['year'] = daily_data_copy['year'].astype('category')
        logger.info("Variable de año creada correctamente")
        return daily_data_copy  
    
    except Exception as e:
        logger.error(f"Error al crear la variable de año: {e}")
        raise e
    
def calculate_weekly_metrics(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el número items vendidos y el precio promedio.
    Args:
        daily_data (pd.DataFrame): Dataframe con los datos crudos.

    Returns:
        pd.DataFrame: Dataframe con los datos preprocesados.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy['weekly_items_sold'] = daily_data_copy.groupby(['week', 'year', 'item_id'])['item_cnt_day'].transform('sum')
        daily_data_copy['weekly_price_avg'] = daily_data_copy.groupby(['week', 'year', 'item_id'])['item_price'].transform('mean')
        logger.info("Métricas de ventas calculadas correctamente")
        return daily_data_copy
    
    except Exception as e:
        logger.error(f"Error al calcular las métricas de ventas: {e}")
        raise e

def transform_week_cyclical(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte la variable 'week' en su representación cíclica usando seno y coseno.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy["week_sin"] = np.sin(2 * np.pi * daily_data_copy["week"] / 52)
        daily_data_copy["week_cos"] = np.cos(2 * np.pi * daily_data_copy["week"] / 52)
        daily_data_copy.drop(columns=["week"], inplace=True)  # Eliminamos "week" porque ya está transformada
        logger.info("Variable week transformada en cíclica correctamente")
        return daily_data_copy
    except Exception as e:
        logger.error(f"Error al transformar la variable week en cíclica: {e}")
        raise e

def create_category_dummies(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Crea dummies para la categoría principal.
    Args:
        daily_data (pd.DataFrame): Dataframe con los datos crudos.

    Returns:
        pd.DataFrame: Dataframe con los datos preprocesados.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy = pd.get_dummies(daily_data_copy, columns=['main_category'], drop_first=True)
        logger.info("Dummies para la categoría principal creados correctamente")
        return daily_data_copy  
    
    except Exception as e:
        logger.error(f"Error al crear dummies para la categoría principal: {e}")
        raise e

def eliminate_duplicates(daily_data: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina duplicados.
    """
    try:
        daily_data_copy = daily_data.copy()
        daily_data_copy = daily_data_copy.drop_duplicates(subset=["year", "week_sin", "week_cos", "item_id"])
        logger.info("Duplicados eliminados correctamente")
        return daily_data_copy
    
    except Exception as e:
        logger.error(f"Error al eliminar duplicados: {e}")
        raise e


def crear_train_test(PATH_DATA, FILE_RAW):
    try:
        df = pd.read_csv(os.path.join(PATH_DATA, FILE_RAW))
        train = df[df['date_block_num'] <= 23]
        test = df[df['date_block_num']  > 23]

        train.to_csv(os.path.join(PATH_DATA, "train2.csv"), index=False)
        test.to_csv(os.path.join(PATH_DATA, "test2.csv"), index=False)
        logger.info("Train y test creados correctamente")

    except Exception as e:
        logger.error(f"Error al crear train y test: {e}")
        raise e
