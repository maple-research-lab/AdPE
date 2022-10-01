


def add_weight_decay(model,lr,adv_lr,weight_decay=1e-5,adv_wd=1e-5, use_adv=True,skip_list=()):
    decay = []
    no_decay = []
    no_decay2 = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if "adv" in name and use_adv:
            no_decay2.append(param)
            print("add track of adv param:",name)
            continue
        if "adv" in name and not use_adv:
            print("skip track of adv param:",name)
            continue
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.,"lr":lr},
        {'params': decay, 'weight_decay': weight_decay,"lr":lr},
        {'params': no_decay2, 'weight_decay': -adv_wd,"lr":-adv_lr,'fix_lr':0},#use fix_lr indicate it's adv or not.
        ]
