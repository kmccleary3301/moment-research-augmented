import json

def safe_serialize(obj, **kwargs) -> str:
    """
    Serialize an object, but if an element is not serializable, return a string representation of said element.
    """
    serialize_attempts = [
        lambda d: d.tolist(),
        lambda d: d.dict(),
        lambda d: d.to_dict(),
        # lambda d: d.__dict__,
    ]

    def default_callable(o):
        nonlocal serialize_attempts
        
        serialized, success = None, False
        
        for attempt in serialize_attempts:
            try:
                serialized = attempt(o)
                success = True
            except:
                pass
        
        if not success:
            return f"<<non-serializable: {type(o).__qualname__}>>"
            
        return serialized
        

    default = lambda o: default_callable(o)
    return json.dumps(obj, default=default, **kwargs)