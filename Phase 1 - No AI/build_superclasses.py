import nltk
import pandas as pd
from nltk.corpus import wordnet as wn

nltk.download("wordnet", quiet=True)

# first match wins, so specific groups come before broad ones
SUPERCLASSES = [
    ("dog", ["dog.n.01"]),
    ("other mammal", ["mammal.n.01"]),
    ("bird", ["bird.n.01"]),
    ("reptile/amphibian", ["reptile.n.01", "amphibian.n.03"]),
    ("fish", ["fish.n.01"]),
    ("invertebrate", ["invertebrate.n.01"]),
    ("food/plant", ["food.n.01", "food.n.02", "plant.n.02", "fungus.n.01"]),
    ("vehicle", ["vehicle.n.01", "craft.n.02", "conveyance.n.03"]),
    ("clothing", ["clothing.n.01", "footwear.n.02", "armor.n.01"]),
    ("container", ["container.n.01"]),
    ("furniture", ["furniture.n.01"]),
    ("structure", ["structure.n.01"]),
    ("device/instrument", ["device.n.01", "instrument.n.01", "equipment.n.01", "implement.n.01", "home_appliance.n.01"]),
    ("other artifact", ["artifact.n.01"]),
    ("nature/scene", ["natural_object.n.01", "geological_formation.n.01", "location.n.01"]),
]


def superclass(synset):
    ancestors = {s.name() for path in synset.hypernym_paths() for s in path}
    for name, roots in SUPERCLASSES:
        if ancestors & set(roots):
            return name
    return "other"


preds = pd.read_csv("results_card/predictions.csv")
classes = preds.assign(wnid=preds["image_id"].str.split("/").str[0])[["label", "wnid"]]
classes = classes.drop_duplicates().sort_values("label")
assert len(classes) == 1000

synsets = classes["wnid"].map(lambda w: wn.synset_from_pos_and_offset("n", int(w[1:])))
classes["name"] = synsets.map(lambda s: s.lemma_names()[0])
classes["superclass"] = synsets.map(superclass)
classes.to_csv("imagenet_superclasses.csv", index=False)
print(classes["superclass"].value_counts())
