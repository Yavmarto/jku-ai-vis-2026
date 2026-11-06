"""Build aiv/imagenet_hierarchy.csv: for each of the 1000 ImageNet classes, its readable name,
a coarse group, and its WordNet hypernym path. Run once; the CSV is committed so the
notebooks do not need nltk.

    pip install nltk && python -m nltk.downloader wordnet
    python phase1/ai-aided/make_hierarchy.py "Phase 1 - No AI/results_card/predictions.csv"

Groups are the first matching WordNet ancestor in GROUPS (order matters: dog before mammal,
vehicle before artifact). Labels are the index of the wnid in sorted order (checked).
"""
import sys
from pathlib import Path

import pandas as pd
from nltk.corpus import wordnet as wn

OUT = Path(__file__).resolve().parent / "aiv" / "imagenet_hierarchy.csv"

GROUPS = [  # (group name, WordNet synsets that define it)
    ("dog", ["dog.n.01", "wolf.n.01", "fox.n.01", "wild_dog.n.01", "hyena.n.01"]),
    ("cat", ["feline.n.01"]),
    ("other mammal", ["mammal.n.01"]),
    ("bird", ["bird.n.01"]),
    ("reptile/amphibian", ["reptile.n.01", "amphibian.n.03"]),
    ("fish", ["fish.n.01"]),
    ("invertebrate", ["invertebrate.n.01"]),
    ("food/plant", ["food.n.01", "food.n.02", "plant_part.n.01", "vascular_plant.n.01",
                    "fungus.n.01", "plant.n.02"]),
    ("vehicle", ["conveyance.n.03", "craft.n.02"]),
    ("clothing/covering", ["clothing.n.01", "covering.n.02", "headdress.n.01"]),
    ("structure/place", ["structure.n.01", "geological_formation.n.01", "area.n.05"]),
    ("device/instrument", ["device.n.01", "instrument.n.01", "equipment.n.01"]),
    ("container", ["container.n.01"]),
    ("other object", ["entity.n.01"]),
]


def main(pred_csv):
    d = pd.read_csv(pred_csv)
    wnid = d.image_id.str.split("/").str[0]
    lab = d.assign(wnid=wnid).groupby("label").wnid.first().sort_index()
    assert len(lab) == 1000 and list(lab) == sorted(lab), "labels must be sorted-wnid indices"
    rows = []
    for label, w in lab.items():
        s = wn.synset_from_pos_and_offset("n", int(w[1:]))
        paths = s.hypernym_paths()
        anc = {a.name() for p in paths for a in p}
        group = next(g for g, syns in GROUPS if anc & set(syns))
        path = max(paths, key=len)  # longest path: most specific chain
        rows.append((label, w, s.lemma_names()[0].replace("_", " "), group,
                     ">".join(a.name() for a in path)))
    out = pd.DataFrame(rows, columns=["label", "wnid", "name", "group", "path"])
    out.to_csv(OUT, index=False)
    print(out.group.value_counts().to_string())
    print("wrote", OUT)


if __name__ == "__main__":
    main(sys.argv[1])
