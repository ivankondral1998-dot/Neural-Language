#!/usr/bin/env python3
"""Prepare official UCI HAR files. No fitting, normalization or training."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

SOURCE = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
ARCHIVE_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
PREFIX = "UCI HAR Dataset/"
CLASSES = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(archive, dest, phase):
    outer = archive.read_bytes()
    if sha(outer) != ARCHIVE_SHA256:
        raise ValueError("unexpected UCI archive SHA-256")
    with ZipFile(io.BytesIO(outer)) as z:
        inner = z.read("UCI HAR Dataset.zip")
    manifest = {"source": SOURCE, "doi": "10.24432/C54S4K", "license": "CC BY 4.0",
                "outer_sha256": sha(outer), "inner_sha256": sha(inner), "members": {},
                "output": {}, "phase": phase, "classes": CLASSES}

    def member(z, name):
        data = z.read(PREFIX + name)
        manifest["members"][name] = sha(data)
        return data

    def write(name, lines):
        data = b"".join(line + b"\n" for line in lines)
        with (dest / name).open("xb") as f:
            f.write(data)
        manifest["output"][name] = sha(data)

    with ZipFile(io.BytesIO(inner)) as z:
        features = member(z, "features.txt").splitlines()
        labels = member(z, "activity_labels.txt").decode("ascii").splitlines()
        assert len(features) == 561
        assert tuple(line.split()[1] for line in labels) == CLASSES
        train_people = list(map(int, member(z, "train/subject_train.txt").split()))
        assert len(train_people) == 7352 and len(set(train_people)) == 21
        split, count = ("test", 2947) if phase == "test" else ("train", 7352)
        people = (list(map(int, member(z, "test/subject_test.txt").split()))
                  if split == "test" else train_people)
        if split == "test":
            assert len(set(people)) == 9 and not set(people) & set(train_people)
            assert set(people) | set(train_people) == set(range(1, 31))
        x = member(z, f"{split}/X_{split}.txt").splitlines()
        y = [int(label) - 1 for label in member(z, f"{split}/y_{split}.txt").split()]
    assert len(x) == len(y) == len(people) == count and set(y) == set(range(6))
    x = [line.split() for line in x]
    assert all(len(row) == 561 for row in x)

    dest.mkdir(parents=True, exist_ok=True)
    validation_people = set(sorted(set(train_people))[-3:]) if phase == "pilot" else set()
    manifest["subjects"] = sorted(set(people) - validation_people)
    manifest["validation_subjects"] = sorted(validation_people)
    if split == "train":
        indices = [i for i, person in enumerate(people) if person not in validation_people]
        write("har_train.txt", (b" ".join([str(y[i]).encode(), *x[i]]) for i in indices))
        manifest["samples"] = len(indices)
        if validation_people:
            indices = [i for i, person in enumerate(people) if person in validation_people]
            write("har_test.txt", (b" ".join(x[i]) for i in indices))
            write("har_labels.txt", (str(y[i]).encode() for i in indices))
            manifest["validation_samples"] = len(indices)
    else:
        write("har_test.txt", (b" ".join(row) for row in x))
        write("har_labels.txt", (str(label).encode() for label in y))
        manifest["samples"] = count
    with (dest / f"data_{phase}.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("phase", choices=("pilot", "train", "test"))
    args = parser.parse_args()
    prepare(args.archive, args.dest, args.phase)
