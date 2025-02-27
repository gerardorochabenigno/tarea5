# training.py
import joblib
import os
import pandas as pd
import logging
from catboost import CatBoostRegressor
logger = logging.getLogger(__name__)

def load_data_train(data_path: str) -> pd.DataFrame:
    """
    Carga el archivo de datos preprocesados.
    
    Args:
        data_path (str): La ruta del archivo de datos preprocesados.
        
    Returns:
        pd.DataFrame: El DataFrame con los datos preprocesados.
    """
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        logger.error(f"El archivo {data_path} no existe.")
        raise FileNotFoundError(f"El archivo {data_path} no existe.")

def split_y_x(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separa los datos en variables dependientes e independientes.
    
    Args:
        df_train (pd.DataFrame): El DataFrame con los datos preprocesados.
        
    Returns:
        tuple[pd.DataFrame, pd.Series]: Una tupla con el DataFrame con los datos preprocesados y el vector de variables dependientes.
    """
    # Cargar los datos preprocesados
    try:
        X_train, y_train = df_train[['week_sin', 'item_id', 'year', 'date_block_num', 'weekly_price_avg', 
                                     'main_category_Libros', 'main_category_Música', 'main_category_Otros',
                                     'main_category_Películas', 'main_category_Programas',
                                     'main_category_Regalos', 'main_category_Tarjetas de prepago',
                                     'main_category_Videojuegos']],df_train['weekly_items_sold']
        logger.info("Características y variable dependiente separadas correctamente.")
        return X_train, y_train
    except Exception as e:
        logger.error("Error al separar las características y la variable dependiente.", e)
        raise e

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> CatBoostRegressor:
    """
    Entrena un modelo de CatBoostRegressor.
    
    Args:
        X_train (pd.DataFrame): El DataFrame con los datos de entrenamiento.
        y_train (pd.Series): El vector de variables dependientes.

    Returns:
        CatBoostRegressor: El modelo entrenado.
    """
    try:
        model = CatBoostRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logger.info("Modelo entrenado correctamente.")
        return model
    except Exception as e:
        logger.error("Error al entrenar el modelo.", e)
        raise e

def save_model(model: CatBoostRegressor, model_path: str) -> None:
    """
    Guarda el modelo entrenado en un archivo.
    
    Args:
        model (CatBoostRegressor): El modelo entrenado.
        model_path (str): La ruta del archivo donde se guardará el modelo.
    """
    try:
        joblib.dump(model, model_path)
        logger.info(f"Modelo guardado correctamente en {model_path}.")
    except Exception as e:
        logger.error("Error al guardar el modelo.", e)
        raise e

