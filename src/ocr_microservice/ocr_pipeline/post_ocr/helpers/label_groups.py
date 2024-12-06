class LabelGroups:
    def __init__(self,id2label,label2id) -> None:
        self.id2label = id2label
        self.label2id = label2id
        self.label_groups = { "O": [0], \
                        "datetime": [1,2,3,4,5], \
                        "header": [6], \
                        "unused1": [7,8,9], \
                        "tax_table_header": [10], \
                        "tax_table": [11,12,13,14], \
                        "unused2": [15,16,17,18,19], \
                        "store": [20,21,22,23,24,25,26,27,28,29], \
                        "item_table_header": [30], \
                        "item_table": [31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49], \
                        "subtotal": [50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69], \
                        "total": [70,71,72,73,74,75,76,77,78,79], \
                        "payment": [80,81,82,83,84,85,86,87,88,89]
                    }
        self.group2color = { "O": "lightyellow", \
                        "datetime": "darkgreen", \
                        "header": "purple", \
                        "unused1": "", \
                        "tax_table_header": "darkviolet", \
                        "tax_table": "brown", \
                        "unused2": "", \
                        "store": "green", \
                        "item_table_header": "indigo", \
                        "item_table": "magenta", \
                        "subtotal": "blue", \
                        "total": "darkred", \
                        "payment": "darkblue"
                    }
    
    def getGroups(self):
        return self.label_groups.keys()
    
    def getColor(self,group):
        return self.group2color[group]

    def getGroupByLabelId(self,label_id):
        for g_name,g_label_list in self.label_groups.items():
            if label_id in g_label_list:
                return g_name
        return None
    
    def getGroupByLabelStr(self,label_str):
        return self.getGroupByLabelId(self.label2id[label_str])
    
    def doBelongToSameGroup(self,labelStrList):
        group_name = None
        for l in labelStrList:
            g = self.getGroupByLabelStr(l)
            if group_name == None:
                group_name = g
            elif group_name != g:
                return False
        return True
    
