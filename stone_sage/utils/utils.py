

def update_configs_with_dict(config_obj, override_dict):
    for key, value in override_dict.items():
        if hasattr(config_obj, key):
            setattr(config_obj, key, value)
    return config_obj

