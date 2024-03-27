import numpy as np
import pandas as pd # pour la transformation en DataFrame
import re

# Importations datasets
from sklearn import datasets

# Pré-modèles !
from train_premodel import *
from test_finetuned import *

# -------------------------------------------
# TRANSFORMATION DATAFRAME

# Transformation de la valeur en texte
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
    
# Transformation d'une ligne en texte
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

# Liste de textes
def df_texts_list(df:pd.DataFrame, **kwargs) :
    """_summary_

    Args:
        df (pd.DataFrame): _description_
        label_num (int, optional): _description_. Defaults to -1.
        subject_name (str, optional): _description_. Defaults to "subject".
        has_unit (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    return [df_row_to_text(df, i, **kwargs) for i in range(len(df))]

# DataFrame Texte (Main)
def df_to_df_text(df:pd.DataFrame, **kwargs) :
    label_num = kwargs.get("label_num", -1) # à tester
    text_list = df_texts_list(df, **kwargs)
    text_array = np.array(text_list).reshape(-1, 1)
    label_array = np.array(df.iloc[:, label_num]).reshape(-1, 1)
    text_df = pd.DataFrame(np.hstack((text_array, label_array)), columns=["text", "label"])
    return text_df

# -------------------------------------------
# TRAINING

# Training (Main)
def train_clf(clf, df_train, label_num=-1) :
    selection = [True for _ in range(len(df_train.columns))]
    selection[label_num] = False
    clf.fit(df_train.iloc[:,selection], df_train.iloc[:,label_num]) 
    print(clf.predict(df_train.iloc[:,:-1]))

# -------------------------------------------
# TRAITEMENT

# Récupérer la liste de features et le nom de label de la question
def question_to_list(q) :
    pattern = r'be (.*) \?'
    match = re.search(pattern, q)
    a_label = match.group(1)

    pattern = r'have (.*), what'
    match = re.search(pattern, q.strip())
    a_features = match.group(1)
    a_list_features = a_features.split(',')
    a_list_features = [feature.strip() for feature in a_list_features]

    return a_list_features, a_label

# Transformer en DataFrame pour faire passer dans le calcul de prédiction
def question_to_df(q) :
    a_list_features, _ = question_to_list(q)
    a_list_features_split = [feature.split('=') for feature in a_list_features]
    a_features_names = [feature[0] for feature in a_list_features_split]
    try : 
        a_features_values = [float(feature[1]) for feature in a_list_features_split]
    except :
        a_features_values = [feature[1] for feature in a_list_features_split]
    return pd.DataFrame(data=np.array([a_features_values]), columns=a_features_names)

# Prédiction de la réponse
def q_df_to_answer(clf, df) :
    return clf.predict(df)[0]

# Prédiction de la réponse
def answer_to_text(q, a) :
    _, a_label = question_to_list(q)
    pattern = r'what .* \?'
    qa = re.sub(pattern, '', q.strip())
    return qa.strip() + " " + a_label + " is " + str(a)

# Réponse du programme à partir de la question
def traitement_question(clf, q) :
    q_df = question_to_df(q)
    a = q_df_to_answer(clf, q_df)
    return answer_to_text(q, a)


"""
# ------
# Tâches
# ------

- Training (Done)
- Fonction avec entrée question et sortie réponse (Done)

- Autres problèmes à régler pour la généralisation :
    - Réglage de types (on considère str ou float) (Done)
    - Ordre des features
    - Features manquants

- A partir du DataFrame de texte à entraîner, faire la même chose

# ------
# Notes
# ------

- Les fonctions sont 'Main' lorsqu'il ne s'agit pas de fonctions intermédiaires
- format question : When we have x1=r.x1, x2=r.x2, . . . , xp=r.xp, what should be y ?
- format réponse : y = r.y


"""