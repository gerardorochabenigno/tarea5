from .preprocessing import (
    paste_item_info,
    reduce_categories,
    change_date_format,
    create_week_variable,
    create_year_variable,
    calculate_weekly_metrics,
    eliminate_duplicates,
    create_category_dummies,
    transform_week_cyclical,
    crear_train_test
)
from .training import (
    load_data_train,
    split_y_x,
    train_model,
    save_model
)



from .inference import (
    split_y_x_inference,
    load_model,
    predict,
    save_predictions,
    load_data_inference
)

__all__ = [
    "paste_item_info",
    "reduce_categories",
    "change_date_format",
    "create_week_variable",
    "create_year_variable",
    "calculate_weekly_metrics",
    "eliminate_duplicates",
    "create_category_dummies",
    "transform_week_cyclical",
    "crear_train_test",
    "load_data_train",
    "split_y_x",
    "train_model",
    "save_model",
    "load_model",
    "predict",
    "save_predictions",
    "load_data_inference",
    "split_y_x_inference"
]

CONFIG = {
    "DATA_RAW_PATH": "data/raw/",
    "DATA_PREP_PATH": "data/prep/",
    "DATA_INFERENCE_PATH": "data/inference/",
    "DATA_PREDICTIONS_PATH": "data/predictions/",
    "MODEL_PATH": "model.joblib"
}
