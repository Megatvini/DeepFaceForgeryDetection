import json


def write_json(data, file_path):
    with open(file_path, 'w') as out:
        json.dump(data, out, indent=4)


def copy_file(src_path, dest_path):
    with open(src_path) as src:
        with open(dest_path, 'w') as dest:
            dest.write(src.read())
