import pandas as pd

def get_val(params: pd.DataFrame, name: str, ptype: str) -> float:
    v = params[(params['name'] == name) & (params['type'] == ptype)]['value']
    return v.iloc[0] if not v.empty else 0