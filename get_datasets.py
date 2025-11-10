from ai import cdas
import json # for pretty output

datasets = cdas.get_datasets(
    'istp_public',
    idPattern='STA.*',
    labelPattern='.*STEREO.*'
)
print(json.dumps(datasets, indent=4))