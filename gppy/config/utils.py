def merge_dicts(base: dict, updates: dict) -> dict:
    """
    Recursively merge the updates dictionary into the base dictionary.

    For each key in the updates dictionary:
    - If the key exists in base and both values are dictionaries,
        then merge these dictionaries recursively.
    - Otherwise, set or override the value in base with the one from updates.

    Args:
        base (dict): The original base configuration dictionary.
        updates (dict): The new configuration dictionary with updated values.

    Returns:
        dict: The merged dictionary containing updates from the new configuration.
    """
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

