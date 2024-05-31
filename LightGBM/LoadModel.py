import lightgbm as lgb
import numpy as np

def loadmodel():
    bst = lgb.Booster(model_file='LightGBM/lightgbm_model.bin')
    return bst
