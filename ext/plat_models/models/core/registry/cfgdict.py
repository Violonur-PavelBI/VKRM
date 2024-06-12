from addict import Dict

# NOTE: hz, we need it?
class CfgDict(Dict):
    """A dictionary for config which has the same interface as python's built-
    in dictionary and can be used as a normal dictionary.
    The Config class would transform the nested fields (dictionary-like fields)
    in config file into ``ConfigDict``.
    """

    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super().__getattr__(name)
        except KeyError:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no " f"attribute '{name}'"
            )
        except Exception as e:
            raise e
        else:
            return value
