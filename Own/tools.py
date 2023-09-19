
import pandas as pd
def insert_generated_feature_to_original_feas(feas, f,f_name=None):
    """
    将新生成的feature插入到原来的dataframe中
    :param feas: 原dataframe
    :param f: 新feature
    :param f_name: 新feature的name，有代表feature为ndarray
    :return: 新生成的dataframe
    """
    if f_name:
        f = pd.DataFrame(f,columns=[f_name])
    y_label = pd.DataFrame(feas[feas.columns[len(feas.columns) - 1]])
    y_label.columns = [feas.columns[len(feas.columns) - 1]]
    feas = feas.drop(columns=feas.columns[len(feas.columns) - 1])
    final_data = pd.concat([feas, f, y_label], axis=1)
    return final_data