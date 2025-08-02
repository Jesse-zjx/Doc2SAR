import base64
import io
import fitz
import re
import pandas as pd
import numpy as np
from typing import Union
from rdkit import Chem
from Levenshtein import distance as levenshtein_distance


synonyms = {
    'Hydrogen': 'H',
    'Helium': 'He',
    'Lithium': 'Li',
    'Beryllium': 'Be',
    'Boron': 'B',
    'Carbon': 'C',
    'Nitrogen': 'N',
    'Oxygen': 'O',
    'Fluorine': 'F',
    'Neon': 'Ne',
    'Sodium': 'Na',
    'Magnesium': 'Mg',
    'Aluminium': 'Al',
    'Aluminium(aluminum)': 'Al',
    'Silicon': 'Si',
    'Phosphorus': 'P',
    'Sulfur': 'S',
    'Chlorine': 'Cl',
    'Argon': 'Ar',
    'Potassium': 'K',
    'Calcium': 'Ca',
    'Scandium': 'Sc',
    'Titanium': 'Ti',
    'Vanadium': 'V',
    'Chromium': 'Cr',
    'Manganese': 'Mn',
    'Iron': 'Fe',
    'Cobalt': 'Co',
    'Nickel': 'Ni',
    'Copper': 'Cu',
    'Zinc': 'Zn',
    'Gallium': 'Ga',
    'Germanium': 'Ge',
    'Arsenic': 'As',
    'Selenium': 'Se',
    'Bromine': 'Br',
    'Krypton': 'Kr',
    'Rubidium': 'Rb',
    'Strontium': 'Sr',
    'Yttrium': 'Y',
    'Zirconium': 'Zr',
    'Niobium': 'Nb',
    'Molybdenum': 'Mo',
    'Technetium': 'Tc',
    'Ruthenium': 'Ru',
    'Rhodium': 'Rh',
    'Palladium': 'Pd',
    'Silver': 'Ag',
    'Cadmium': 'Cd',
    'Indium': 'In',
    'Tin': 'Sn',
    'Antimony': 'Sb',
    'Tellurium': 'Te',
    'Iodine': 'I',
    'Xenon': 'Xe',
    'Cesium': 'Cs',
    'Barium': 'Ba',
    'Lanthanum': 'La',
    'Cerium': 'Ce',
    'Praseodymium': 'Pr',
    'Neodymium': 'Nd',
    'Promethium': 'Pm',
    'Samarium': 'Sm',
    'Europium': 'Eu',
    'Gadolinium': 'Gd',
    'Terbium': 'Tb',
    'Dysprosium': 'Dy',
    'Holmium': 'Ho',
    'Erbium': 'Er',
    'Thulium': 'Tm',
    'Ytterbium': 'Yb',
    'Lutetium': 'Lu',
    'Hafnium': 'Hf',
    'Tantalum': 'Ta',
    'Tungsten': 'W',
    'Rhenium': 'Re',
    'Osmium': 'Os',
    'Iridium': 'Ir',
    'Platinum': 'Pt',
    'Gold': 'Au',
    'Mercury': 'Hg',
    'Thallium': 'Tl',
    'Lead': 'Pb',
    'Bismuth': 'Bi',
    'Polonium': 'Po',
    'Astatine': 'At',
    'Radon': 'Rn',
    'Francium': 'Fr',
    'Radium': 'Ra',
    'Actinium': 'Ac',
    'Thorium': 'Th',
    'Protactinium': 'Pa',
    'Uranium': 'U',
    'Neptunium': 'Np',
    'Plutonium': 'Pu',
    'Americium': 'Am',
    'Curium': 'Cm',
    'Berkelium': 'Bk',
    'Californium': 'Cf',
    'Einsteinium': 'Es',
    'Fermium': 'Fm'
}



def fuzzy_compare_name(a: str, b: str, metric="EditDistance", **kwargs) -> Union[bool, float]:
    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    a = a.strip()
    b = b.strip()

    if a == "" or b == "" and not a + b == "":
        return False
    if is_float(a) and is_float(b):
        return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)

    if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
            (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
        return True
    else:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / max(len(a), len(b))
            # return 1 - Levenshtein.distance(a.lower(), b.lower()) / (len(a) + len(b))
        elif metric == "Word2Vec":
            pass


def fuzzy_normalize_name(s):
    s = str(s)
    if s.startswith("Unnamed"):
        return ""
    else:
        """ Standardize name or index string. """
        # # 定义需要移除的单位和符号
        # units = ["µM", "µg/mL", "nM", "%", "wt.%", "at.%", "at%", "wt%"]
        # for unit in units:
        #     s = s.replace(unit, "")

        # 定义特定关键字
        keywords = ["pIC50", "IC50", "EC50", "TC50", "GI50", "Ki", "Kd", "Kb", "pKb"]

        # 移除非字母数字的字符，除了空格
        s = re.sub(r'[^\w\s%.\-\(\)]', '', s)
        if s in synonyms:
            s = synonyms[s]

        # 分割字符串为单词列表
        words = s.split()

        # 将关键字移到末尾
        reordered_words = [word for word in words if word not in keywords]
        keywords_in_string = [word for word in words if word in keywords]
        reordered_words.extend(keywords_in_string)
        # 重新组合为字符串
        return ' '.join(reordered_words)


def match_list_bipartite(ind0: list, ind1: list, threshold: float = 0.9,
                         similarity_method=fuzzy_compare_name) -> dict:
    """
    Match the 2 list of string or tuple. Maybe indices of two dataframes.
    """
    from munkres import Munkres

    renames = {}
    similarities = np.array(np.ones([len(ind0) + 15, len(ind1) + 15]), dtype=np.float64)

    if similarity_method == fuzzy_compare_name:
        name2query = lambda name: name if type(name) != tuple else name[0] if len(name) == 1 or name[-1] == "" else \
            name[-1]
        querys0 = [fuzzy_normalize_name(name2query(name)) for name in ind0]
        querys1 = [fuzzy_normalize_name(name2query(name)) for name in ind1]
    else:
        name2query = lambda name: name
        querys0 = ind0
        querys1 = ind1
    for i, name_i in enumerate(ind0):
        query_i = querys0[i]
        for j, name_j in enumerate(ind1):
            query_j = querys1[j]
            if query_i == "" or query_j == "":
                similarities[i, j] = 0
            result = similarity_method(query_i, query_j)
            if type(result) == bool:
                similarities[i, j] = 1 if result else 0
            elif type(result) == float:
                similarities[i, j] = result

    for k in range(15):
        for i in range(len(ind0)):
            similarities[i][len(ind1) + k] = threshold
        for j in range(len(ind1)):
            similarities[len(ind0) + k][j] = threshold
    dists = 1 - similarities
    # print(pd.DataFrame(dists, index=querys0 + ["v"] * 15, columns=querys1 + ["v"] * 15))

    # Kuhn-Munkres algorithm for useful solving the rectangular Assignment Problem
    mu = Munkres()
    indexes = mu.compute(dists.tolist())

    # 根据最优匹配下标输出映射
    for i, j in indexes:
        if (i < len(ind0)) and (j < len(ind1)):
            renames[name2query(ind1[j])] = name2query(ind0[i])
    return renames


def fuzzy_compare_value(a: str, b: str, metric="EditDistance", **kwargs) -> Union[bool, float]:
    """
    Compare two strings with fuzzy matching.
    """

    def standardize_unit(s: str) -> str:
        """
        Standardize a (affinity) string to common units.
        """
        mark = "" if re.search(r"[><=]", s) is None else re.search(r"[><=]", s).group()
        unit = s.rstrip()[-2:]
        number = float(re.search(r"[\+\-]*[0-9.]+", s).group())

        if unit in ["µM", "uM"]:
            unit = "nM"
            number *= 1000
        elif unit in ["mM", "mm"]:
            unit = "nM"
            number *= 1000000

        if mark == "=":
            mark = ""
        return f"{mark}{number:.1f} {unit}"

    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    try:
        unit_str = ["nM", "uM", "µM", "mM", "%", " %", "wt.%", "at.%", "at%", "wt%"]
        nan_str = ["n/a", "nan", "na", "n.a.", "nd", "not determined", "not tested", "inactive"]
        a = a.strip()
        b = b.strip()
        if is_float(a) and is_float(b):
            return np.allclose(float(a), float(b), equal_nan=True, atol=1e-2, rtol=1e-2)
        elif fuzzy_normalize_value(a) == "bal" or fuzzy_normalize_value(b) == "bal":
            return True
        elif fuzzy_normalize_value(a) == fuzzy_normalize_value(b):
            return True
        elif ((a[-2:] in unit_str or a[-1] in unit_str or a.split()[-1] in unit_str) and
              (b[-2:] in unit_str or b[-1] in unit_str or b.split()[-1] in unit_str)):
            a = standardize_unit(a)
            b = standardize_unit(b)
            return a == b
        elif a.lower() in nan_str and b.lower() in nan_str:
            return True
        if ((a.lower().startswith(b.lower()) or a.lower().endswith(b.lower())) or
                (b.lower().startswith(a.lower()) or b.lower().endswith(a.lower()))):
            return True
        else:
            if metric == "EditDistance":
                import Levenshtein
                return 1 - Levenshtein.distance(a.lower(), b.lower()) / max(len(a), len(b))
            elif metric == "Word2Vec":
                pass
    except:
        if metric == "EditDistance":
            import Levenshtein
            return 1 - Levenshtein.distance(a.lower(), b.lower()) / max(len(a), len(b))
        elif metric == "Word2Vec":
            pass


def fuzzy_normalize_value(vi):
    try:
        vi = str(vi).lower()

        if "bal" in vi or "remainder" in vi or "bas" in vi:
            vi = "bal"
            return "bal"

        if ("nan" in vi and not "–" in vi) or "/" == vi or "n/a" in vi or "na" in vi or vi == "":
            vi = "0"
        vi = vi.replace("nan", "–").replace("~", "-")

        pattern = r"\d+(?:\.\d+)?"
        matches = re.findall(pattern, vi)
        if len(matches) == 2:
            vi = f"{matches[0]}-{matches[1]}"
        elif len(matches) == 1:
            vi = matches[0]

        if "<" in vi:
            vi = vi.replace("<", "")
        if ">" in vi:
            vi = vi.replace(">", "")

        try:
            vi = float(vi)
            vi = round(vi, 3)
        except:
            # print(vi)
            pass
    except:
        pass

    return vi


def compare_molecule_strict(smi1, smi2, **kwargs) -> bool:
    from rdkit import Chem

    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    if mol1 is None or mol2 is None:
        return False
    else:
        Chem.RemoveStereochemistry(mol1)
        Chem.RemoveStereochemistry(mol2)
        return Chem.MolToSmiles(Chem.RemoveHs(mol1), isomericSmiles=False) == Chem.MolToSmiles(Chem.RemoveHs(mol2),
                                                                                               isomericSmiles=False)


def affinity_match(df_ref, df_prompt, index='Compound', compare_fields=[]):
    assert len(df_ref) > 0, "Prompt table is empty."
    if df_prompt is None or len(df_prompt) == 0:
        return {"recall_field": 0.0, "recall_index": 0.0, "recall_value": 0.0, "recall_value_strict": 0.0,
                "accuracy_value": 0.0, "accuracy_value_strict": 0.0}
    metrics = {}
    index_names = ["Compound", "Name", "SMILES", "Nickname", "Substrate", "AlloyName"]
    # ref_df, prompt_df = df_ref, df_prompt


    if index not in [None, ""]:
        df_ref[index] = df_ref[index].astype(str)
        df_ref = df_ref.set_index(index)
        df_prompt[index] = df_prompt[index].astype(str)
        df_prompt = df_prompt.set_index(index)

    renames = match_list_bipartite(compare_fields, df_prompt.columns)
    renames = {key: value for key, value in renames.items() if key not in index_names}
    if len(renames) > 0:
        df_prompt.rename(columns=renames, inplace=True)

    renames = match_list_bipartite(df_ref.index, df_prompt.index)
    renames = {key: value for key, value in renames.items() if key not in index_names}
    if len(renames) > 0:
        # print("Find similar indices between answer and correct:", renames)
        df_prompt.rename(index=renames, inplace=True)

    compare_fields_ = [col for col in compare_fields if
                       col not in [index] + ([index[0]] if type(index) == tuple else [])]
    metrics["recall_field"] = max(
        len([item for item in compare_fields_ if item in df_prompt.columns]) / len(compare_fields_), 1.0)
    metrics["recall_index"] = max(len([item for item in df_ref.index if item in df_prompt.index]) / df_ref.shape[0],
                                  1.0)

    match_score, total_match_score, smiles_match_score = 0.0, 0.0, 0.0
    N, M = len(df_ref.index), len(compare_fields_)
    for idx in df_ref.index:
        _total_matching = 1.0
        _tmp_match_score = 0.0
        for col in compare_fields_:
            gtval = df_ref.loc[idx, col]
            gt = str(gtval.iloc[0]) if type(gtval) == pd.Series else str(gtval)
            try:
                pval = df_prompt.loc[idx, col]
                p = str(pval.iloc[0]) if type(pval) == pd.Series else str(pval)
            except:
                p = 'not found'
            _is_matching = fuzzy_compare_value(gt, p) if col != "SMILES" else compare_molecule_strict(gt, p)
            if float(_is_matching) > 0 and float(_is_matching) < 1:
                _is_matching = 0.0
            if col == "SMILES":
                smiles_match_score += float(_is_matching)
            _total_matching *= float(_is_matching)
            _tmp_match_score += float(_is_matching) / M
        match_score += float(_tmp_match_score)
        total_match_score += _total_matching
        _total_matching = 1.0
        _tmp_match_score = 0.0
    metrics['smiles_match_rate'] = smiles_match_score / N
    metrics = {
        **metrics,
        "recall_value": match_score / N,
        "recall_value_strict": total_match_score / N,
        "accuracy_value": match_score / N * metrics["recall_index"],
        "accuracy_value_strict": total_match_score / N * metrics["recall_index"],
    }
    return metrics
