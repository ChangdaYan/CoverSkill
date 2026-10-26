"""Separate actual stopping from the original LR/auxiliary scheduling horizon."""


def resolve_budget(recipe, smoke_steps=0):
    horizon = int(recipe["max_steps"])
    stop = int(recipe.get("stop_after_steps", horizon))
    if not 1 <= stop <= horizon:
        raise ValueError("stop_after_steps must be in [1,max_steps]")
    if not recipe.get("checkpoint_steps") or max(recipe["checkpoint_steps"]) != stop:
        raise ValueError("last saved checkpoint must be the actual endpoint")
    if any(not 1 <= int(x) <= stop for x in recipe["checkpoint_steps"]):
        raise ValueError("checkpoint beyond actual stopping step")
    return horizon, int(smoke_steps) if smoke_steps else stop
