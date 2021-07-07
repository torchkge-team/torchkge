import yaml
import sys
import logging


class ConfigDict(dict):
    __getattr__ = dict.__getitem__


def _make_config_dict(obj):
    if isinstance(obj, dict):
        return ConfigDict({k: _make_config_dict(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_make_config_dict(x) for x in obj]
    else:
        return obj


_config = None


def config():
    global _config
    if _config is None:
        # config_path = 'torchkge/config.yaml'
        config_path = 'config.yaml'
        for arg in sys.argv[1:]:
            if arg.startswith('--config='):
                config_path = arg[9:]
                break
        print('Reading config from ' + config_path)
        with open(config_path) as f:
            _config = _make_config_dict(yaml.load(f))
        overwrite_config_with_args()
    return _config


def path_set(path, val, sep='.', auto_convert=False):
    steps = path.split(sep)
    obj = _config
    for step in steps[:-1]:
        obj = obj[step]
    old_val = obj[steps[-1]]
    if not auto_convert:
        obj[steps[-1]] = val
    elif isinstance(old_val, bool):
        obj[steps[-1]] = val.lower() == 'true'
    elif isinstance(old_val, float):
        obj[steps[-1]] = float(val)
    elif isinstance(old_val, int):
        try:
            obj[steps[-1]] = int(val)
        except ValueError:
            obj[steps[-1]] = float(val)
    else:
        obj[steps[-1]] = val


def overwrite_config_with_args(args=None, sep='.'):
    if args is None:
        args = sys.argv[1:]
    for arg in args:
        if arg.startswith('--') and '=' in arg:
            path, val = arg[2:].split('=')
            if path != 'config':
                path_set(path, val, sep, auto_convert=True)


def _dump_config(obj, prefix):
    if isinstance(obj, dict):
        for k, v in obj.items():
            _dump_config(v, prefix + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _dump_config(v, prefix + (str(i),))
    else:
        if isinstance(obj, str):
            rep = obj
        else:
            rep = repr(obj)
        logging.debug('%s=%s', '.'.join(prefix), rep)


def dump_config():
    return _dump_config(_config, tuple())
