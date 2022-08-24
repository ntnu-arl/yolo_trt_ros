CLASSES_LIST = [
    'Survivor',
    'Fire Extinguisher',
    'Cell Phone',
    'Drill',
    'Backpack',
    'Vent',
    'Helmet',
    'Rope',
]

def get_cls_dict(category_num):
    """Get the class ID to name translation dictionary."""
    if category_num == len(CLASSES_LIST):
        return {i: n for i, n in enumerate(CLASSES_LIST)}
    else:
        return {i: 'CLS%d' % i for i in range(category_num)}
def get_class_name(class_num):
    if class_num >= len(CLASSES_LIST):
        return "UNKNOWN"
    return CLASSES_LIST[class_num]
