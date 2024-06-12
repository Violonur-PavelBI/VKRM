import argparse
import yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path")
    args = parser.parse_args()
    config_path = args.config_path
    with open(config_path) as yamlfile:
        args = yaml.load(yamlfile, Loader=yaml.Loader)
    args["common"]["exp"]["debug"] = True
    args["common"]["exp"]["device"] = "cpu"
    # filter ps stages
    old_execute = args["execute"]
    new_execute = []
    skip = False
    for i, stage in enumerate(old_execute):
        stage: str
        if not stage.startswith("ps"):
            if skip:
                new_execute.append(old_execute[i - 1])
            skip = False
        if not skip:
            new_execute.append(old_execute[i])
        if i == 0:
            skip = True
    args["execute"] = new_execute
    with open(config_path, "w") as yamlfile:
        yaml.dump(args, yamlfile, default_flow_style=False)
