import pandas as pd
import random
import csv
from collections import OrderedDict

data = []
with open('captions.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        if row and len(row) >= 2:
            data.append({'image': row[0].strip(), 'caption': row[1].strip()})

df = pd.DataFrame(data)

all_captions_list = df['caption'].tolist()

ordered_groups = OrderedDict()
for _, row in df.iterrows():
    img = row['image']
    if img not in ordered_groups:
        ordered_groups[img] = []
    ordered_groups[img].append(row['caption'])

def extend_with_plurals(base_map):
    extended = dict(base_map)
    for k, v in list(base_map.items()):
        if not k.endswith('s'):
            extended[k + 's'] = v + 's'
    return extended

color_map = {
    'pink': 'blue', 'blue': 'pink', 'black': 'white', 'white': 'black', 'red': 'green', 'green': 'red',
    'yellow': 'purple', 'purple': 'yellow', 'orange': 'green', 'dark': 'bright', 'bright': 'dark'
}

number_map = {
    'one': 'three', 'two': 'four', 'three': 'one', 'four': 'two',
    'five': 'three', 'six': 'four', 'seven': 'five',
    'single': 'multiple', 'many': 'few', 'few': 'many',
    'first': 'last', 'last': 'first',
}

size_map = {
    'small': 'large', 'large': 'small', 'little': 'big', 'big': 'little',
    'tall': 'short', 'short': 'tall', 'long': 'short', 'tiny': 'huge',
    'huge': 'tiny', 'young': 'old', 'old': 'young',
}

action_map = {
    'running': 'sitting', 'sitting': 'running', 'jumping': 'standing', 'standing': 'jumping',
    'climbing': 'descending', 'descending': 'climbing', 'playing': 'sleeping', 'sleeping': 'playing',
    'fighting': 'hugging', 'hugging': 'fighting', 'walking': 'running', 'swimming': 'flying',
    'flying': 'swimming', 'eating': 'drinking', 'drinking': 'eating',
}

animal_map = {
    'dog': 'cat', 'cat': 'dog', 'horse': 'cow', 'cow': 'horse', 'bird': 'fish', 'fish': 'bird',
    'puppy': 'kitten'
}

object_map = {
    'ball': 'frisbee', 'frisbee': 'ball', 'toy': 'stick', 'stick': 'toy', 'hat': 'glasses', 'glasses': 'hat',
    'bench': 'chair', 'rope': 'net', 'fence': 'wall', 'tent': 'hut', 'bike': 'skateboard'
}

scene_map = {
    'beach': 'mountain', 'mountain': 'beach', 'snow': 'sand', 'sand': 'snow',
    'water': 'grass', 'grass': 'water', 'pool': 'field', 'field': 'pool',
    'ocean': 'forest', 'forest': 'ocean', 'lake': 'desert',
    'river': 'road', 'road': 'river',
}

all_maps = [color_map, number_map, size_map, action_map, animal_map, object_map, scene_map]

def generate_hard_negative(caption):
    words = caption.split()
    num_changes = random.randint(1, 2)
    changed_indices = set()
    total_changed = 0

    for _ in range(num_changes):
        made_change = False
        maps_order = list(all_maps)
        random.shuffle(maps_order)

        for sub_map in maps_order:
            indices = list(range(len(words)))
            random.shuffle(indices)
            for i in indices:
                if i in changed_indices:
                    continue
                lower = words[i].lower().strip('.,!?\'"():;')
                if lower in sub_map:
                    replacement = sub_map[lower]
                    original = words[i]
                    prefix = ''
                    suffix = ''
                    while original and not original[0].isalpha():
                        prefix += original[0]
                        original = original[1:]
                    while original and not original[-1].isalpha():
                        suffix = original[-1] + suffix
                        original = original[:-1]
                    if original and original[0].isupper():
                        replacement = replacement.capitalize()

                    words[i] = prefix + replacement + suffix
                    changed_indices.add(i)
                    total_changed += 1
                    made_change = True
                    break
            if made_change:
                break

    return ' '.join(words) if total_changed > 0 else None


output_data = []
stats = {'hard': 0, 'random': 0, 'failed_hard': 0}

for image, positives in ordered_groups.items():
    num_neg = len(positives)

    for cap in positives:
        output_data.append({'image': image, 'caption': cap, 'label': 1})

    generated_neg = set()
    attempts = 0

    while len(generated_neg) < num_neg and attempts < num_neg * 20:
        attempts += 1

        if random.random() < 0.85:
            base = random.choice(positives)
            neg = generate_hard_negative(base)
            if neg is None:
                stats['failed_hard'] += 1
                continue
            neg_type = 'hard'
        else:
            neg = random.choice(all_captions_list)
            neg_type = 'random'

        if neg not in positives and neg not in generated_neg:
            generated_neg.add(neg)
            stats[neg_type] += 1

    for neg in generated_neg:
        output_data.append({'image': image, 'caption': neg, 'label': 0})

dataset_df = pd.DataFrame(output_data)
dataset_df.to_csv('dataset_with_negatives.csv', index=False, encoding='utf-8')

print(f"Gotowe! Dataset ma {len(dataset_df)} wierszy")
print(dataset_df['label'].value_counts())
print(f"\nStatystyki negatywow:")
print(f"  Trudne: {stats['hard']}")
print(f"  Losowe: {stats['random']}")
print(f"  Nieudane proby trudnych: {stats['failed_hard']}")