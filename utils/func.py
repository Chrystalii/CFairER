import torch
cuda0 = torch.device('cuda:0')

def print_dict(dic):
    """print dictionary using specified format

    example: {"a": 1, "b": 2}
    output:
            "a": 1
            "b": 2
    """
    print("\n".join("{:10s}: {}".format(key, values) for key, values in dic.items()))

def interaction_dict(inter_mat):
    #return historical interaction list as dict: {'user_id': [item1_id, item2_id,...]}
    user_dict = dict()
    for u_id, i_id in zip(inter_mat.row, inter_mat.col):
        if u_id not in user_dict.keys():
            user_dict[u_id] = list()
        user_dict[u_id].append(i_id)
    return user_dict

