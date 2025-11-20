import json 

class SimpleConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def __getitem__(self, key):
        return self.config[key] 

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])

        assert all([k not in module_args for k in kwargs]), "Cannot overwrite config file"
        module_args.update(kwargs)

        return getattr(module, module_name)(*args, **module_args)
    
    def get_nested(self, *keys):
        result = self.config
        for key in keys:
            result = result[key]
        return result
        


