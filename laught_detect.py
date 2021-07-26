import os
from keras.models import load_model
from laugh_segmenter import segment_laughs

path_to_fragments = "fragments/"
fragments = [os.path.join(path_to_fragments, name) for name in os.listdir(path_to_fragments) if name.endswith('.wav')]
path_to_model = "models/wight_laugh.h5"
model = load_model(path_to_model)

threshold = 0.83
min_laugh_length = 0.3

for fragment in fragments:
    laughs = segment_laughs(fragment, model, threshold, min_laugh_length)

    print(fragment)
    if not isinstance(laughs, str):
        for laugh in laughs:
            print(laugh)
        print()
    else:
        print(laughs)
        print()