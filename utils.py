def init_obj(module_name, module_args, module, *args):
    return getattr(module, module_name)(*args, **module_args)