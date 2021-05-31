CLASSES_LIST = [
    'survivor',
    'fire extinguisher',
    'cellphone',
    'drill',
    'backpack',
    'vent',
    'helmet',
    'rope',
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == len(CLASSES_LIST):
        return {i: n for i, n in enumerate(CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
