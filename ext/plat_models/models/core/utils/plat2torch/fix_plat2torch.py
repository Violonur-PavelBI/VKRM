def fix_dense(layer, tensors):
    tensors[layer.get("input")] = tensors.get(layer.get("input")).view(
        tensors.get(layer.get("input")).size(0), -1
    )
    return tensors


FixPreviousLayers = {
    "dense": fix_dense,
}


def fix_lstm(layer, tensors):
    tensors["lstm"] = tensors.get(layer.get("output"))[0]
    return tensors


FixNextLayers = {"lstm": fix_lstm}
