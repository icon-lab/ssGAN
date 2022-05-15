def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'ssGAN':
        from .ssGAN_model import ssGANModel
        model = ssGANModel()        
    else:
        raise NotImplementedError('model [%s] not implemented.' % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
