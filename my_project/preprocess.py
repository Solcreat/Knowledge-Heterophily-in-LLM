from my_project.config import REQUIRED_HEADERS
# REQUIRED_HEADERS is a dictionary that consists of str : [str, ... str]:
# REQUIRED_HEADERS = {
#     'triplet_id': ['triplet_id', 'id', 'tid', 'triple_id'],
# ...


def uniform_headers(df):  # df - DataFrame
    # for each required header, find which actual column it matches and rename
    mapping = {}
    for std, variants in REQUIRED_HEADERS.items():  # .items() returns a view-object consisting of (key, value)
        for v in variants:  # variants consists of aliases like ['triplet_id', 'id', 'tid', 'triple_id']
            if v in df.columns:  # df.columns can be, for example, in codex: Index(['triplet_id', 'sub_label', 'rel_label', 'obj_label'])
                mapping[v] = std  # this can result in, for instance, {'id': 'triplet_id'}
                break  # we standardized that column. Go to the 'sub_label' : ['subject', ... 'sub'] etc.
        else:
            raise KeyError(f"No column found for '{std}' among {variants}")
    return df.rename(columns=mapping)[list(REQUIRED_HEADERS.keys())]  # df.rename(columns=mapping) will change the old name (key) to the new standardized name (value)
# because of mapping[variant] = standard e.g. mapping['tid'] = 'triplet_id', we'll get a uniform columns in DataFrame.