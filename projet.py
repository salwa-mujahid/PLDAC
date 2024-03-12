from sklearn import datasets # pour importer le dataset iris
import pandas as pd # pour la transformation en DataFrame
import re

# Fonctions intermédiaires pour la conversion de samples en texte


def text_type_select(value, feature, select=0) :
    """ Retourne un type de texte selon la sélection
    0. <value> <feature>
    1. <value> <unit> of <name> (avec feature = (<name>, <unit>))

    Args:
        value (float): la valeur du feature
        feature (str): le nom de feature
        select (int, optional): Type de texte à sélectionner. Defaults to 0.

    Returns:
        str: Texte retourné
    """
    if select == 0 :
        return str(value) + " " + str(feature)
    if select == 1 :
        name, unit = feature
        return str(value) + " " + str(unit) + " of " + str(name)

def df_row_to_text(df:pd.DataFrame, row_num, label_num=-1, subject_name = "subject", has_unit=False) :
    """ Pour une ligne d'un DataFrame, génère un texte expliquant la ligne

    Args:
        `df` (pd.DataFrame): Le DataFrame
        `row_num` (int): La ligne du DataFrame dont on veut générer la description en texte
        `label_num` (int, optional): La colonne contenant le label. Defaults to -1 (la dernière colonne).
        `subject_name` (str, optional): Le nom du sujet pour l'affichage. Defaults to "subject".
        `has_unit` (bool, optional): _description_. Defaults to True.

    Returns:
        str: Le texte généré
    """

    # features
    values = [i for i in df.iloc[row_num]]
    feature_names = list(df.columns)

    # label
    label_value = values.pop(label_num)
    feature_names.pop(label_num)

    # features name - unit
    if has_unit :
        regex_unit_pattern = r'([\w ]+)\s+\((\w+)\)$'
        feature_names = [re.search(regex_unit_pattern, feature).groups() for feature in feature_names] 
        has_unit_vector = [True if len(tuple) == 2 else False for tuple in feature_names]
    else :
        has_unit_vector = [False for _ in feature_names]

    # generate text type
    value_feature_text_list = [text_type_select(value, feature, select=1) if has_unit_vector[i] \
                               else text_type_select(value, feature, select=0) \
                               for i, (value, feature) in enumerate(zip(values, feature_names))]

    # generate text
    text = "The " + subject_name + " with "
    for vf in value_feature_text_list[:-1] :
        text += vf + ", "
    text = text[:-2]
    text += " and " + value_feature_text_list[-1]
    text += " is a " + str(label_value) # a / an 

    return text

def df_texts_list(df:pd.DataFrame, label_num=-1, subject_name = "subject", has_unit=False) :
    return [df_row_to_text(df, i, label_num, subject_name, has_unit) for i in range(len(df))]
