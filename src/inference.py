"""
Módulo para el inferencia del modelo.

Este módulo contiene funciones para cargar un modelo, realizar predicciones y guardarlas.
"""

# inference.py

import logging
import joblib
import pandas as pd
from catboost import CatBoostRegressor

logger = logging.getLogger(__name__)



def load_data_inference(data_path: str) -> pd.DataFrame:
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

def split_y_x_inference(df_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
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

def load_model(model_path: str) -> CatBoostRegressor:
    """
    Carga un modelo CatBoostRegressor desde un archivo.

    Args:
        model_path (str): La ruta del archivo donde se encuentra el modelo.

    Returns:
        CatBoostRegressor: El modelo CatBoostRegressor cargado.
    """
    try:
        model = joblib.load(model_path)
        logger.info(f"Modelo cargado correctamente desde {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error al cargar el modelo desde {model_path}: {e}")
        raise e
    

def predict(model: CatBoostRegressor, data: pd.DataFrame) -> pd.Series:
    """
    Realiza predicciones utilizando un modelo CatBoostRegressor.

    Args:
        model (CatBoostRegressor): El modelo CatBoostRegressor.
        data (pd.DataFrame): El DataFrame con los datos de entrada.

    Returns:
        pd.Series: Las predicciones realizadas.
    """
    try:
        predictions = model.predict(data)
        logger.info(f"Predicciones realizadas correctamente")
        return predictions
    except Exception as e:
        logger.error(f"Error al realizar predicciones: {e}")
        raise e


def save_predictions(predictions, output_path: str) -> None:
    """
    Guarda las predicciones en un archivo CSV.

    Args:
        predictions (numpy.ndarray | pd.Series): Las predicciones realizadas.
        output_path (str): La ruta del archivo donde se guardarán las predicciones.
    """
    try:
        # Convertir a Series si es un ndarray
        predictions_series = pd.Series(predictions)
        predictions_series.to_csv(output_path, index=False)
        logging.info(f"Predicciones guardadas correctamente en {output_path}")
    except Exception as e:
        logging.error(f"Error al guardar las predicciones en {output_path}: {e}")
        raise e


