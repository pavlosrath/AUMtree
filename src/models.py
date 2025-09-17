def get_model_default_kwargs_for_ds(ds_name):
    # tiny datsets
    if ds_name in ('mushrooms', 'cardiotocography',):
        return dict(
            n_estimators = 30,
            max_depth = 3,
        )
    # small and easy datasets
    elif ds_name in (
        'satelite', 'digits', 'spirals', 'sensorless_drive', 'credit_card_fraud',
        'balanced_credit_card_fraud'
        ):
        return dict(
            n_estimators = 50,
            max_depth = 5,
        )
    # large or difficult datasets
    # 'human_activity' performs slightly better with `max_depth=10`, 
    # but this would make training way slower
    elif ds_name in ('letters', 'human_activity_recognition'):
        return dict(
            n_estimators = 100,
            max_depth = 5,
        )
    elif ds_name in ('mnist', 'fashion_mnist'):
        return dict(
            n_estimators = 100,
            max_depth = 10,
        )
    else:
        raise ValueError(f'Unknown dataset {ds_name}')