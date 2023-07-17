import sys
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
from ast import literal_eval
sys.path.append("/Users/claire/Desktop/Stuff-/codes/dissertation/cxr/rgrg/src/full_model")
sys.path.append("/Users/claire/Desktop/Stuff-/codes/dissertation/cxr/rgrg/src")
# import sys
sys.path.append("/Users/claire/Desktop/Stuff-/codes/dissertation/cxr/rgrg/")

from full_model.train_full_model import *
from full_model.generate_reports_for_images import *

def get_datasets():
    PERCENTAGE_OF_TRAIN_SET_TO_USE = 0.1
    PERCENTAGE_OF_VAL_SET_TO_USE = 0.1
    usecols = [
        "mimic_image_file_path",
        "bbox_coordinates",
        "bbox_labels",
        "bbox_phrases",
        "bbox_phrase_exists",
        "bbox_is_abnormal",
        'bbox_abnormalities',
    ]

    # all of the columns below are stored as strings in the csv_file
    # however, as they are actually lists, we apply the literal_eval func to convert them to lists
    converters = {
        "bbox_coordinates": literal_eval,
        "bbox_labels": literal_eval,
        "bbox_phrases": literal_eval,
        "bbox_phrase_exists": literal_eval,
        "bbox_is_abnormal": literal_eval,
        "bbox_abnormalities": literal_eval,
    }

    datasets_as_dfs = {}
    datasets_as_dfs["train"] = pd.read_csv(os.path.join(path_full_dataset, "train_ab.csv"), usecols=usecols, converters=converters)
    datasets_as_dfs["test"] = pd.read_csv(os.path.join(path_full_dataset, "test_ab.csv"), usecols=usecols, converters=converters)

    total_num_samples_train = len(datasets_as_dfs["train"])
    total_num_samples_val = len(datasets_as_dfs["test"])

    # compute new number of samples for both train and val
    new_num_samples_train = int(PERCENTAGE_OF_TRAIN_SET_TO_USE * total_num_samples_train)
    new_num_samples_val = int(PERCENTAGE_OF_VAL_SET_TO_USE * total_num_samples_val)


    from datasets import Dataset
    # limit the datasets to those new numbers
    datasets_as_dfs["train"] = datasets_as_dfs["train"][:new_num_samples_train]
    datasets_as_dfs["test"] = datasets_as_dfs["test"][:new_num_samples_val]

    raw_train_dataset = Dataset.from_pandas(datasets_as_dfs["train"])
    raw_test_dataset = Dataset.from_pandas(datasets_as_dfs["test"])

    return raw_train_dataset, raw_test_dataset

import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms, log):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms
        self.log = log

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
        # get the image_path for potential logging in except block
        image_path = self.tokenized_dataset[index]["mimic_image_file_path"]

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        try:
            bbox_coordinates = self.tokenized_dataset[index]["bbox_coordinates"]  # List[List[int]]
            bbox_labels = self.tokenized_dataset[index]["bbox_labels"]  # List[int]
            input_ids = self.tokenized_dataset[index]["input_ids"]  # List[List[int]]
            attention_mask = self.tokenized_dataset[index]["attention_mask"]  # List[List[int]]
            bbox_phrase_exists = self.tokenized_dataset[index]["bbox_phrase_exists"]  # List[bool]
            bbox_is_abnormal = self.tokenized_dataset[index]["bbox_is_abnormal"]  # List[bool]
            bbox_abnormalities = self.tokenized_dataset[index]["bbox_abnormalities"]  # List[List[int]]
            

            if self.dataset_name != "train":
                # we only need the reference phrases during evaluation when computing scores for metrics
                bbox_phrases = self.tokenized_dataset[index]["bbox_phrases"]  # List[str]

                # same for the reference_report
                reference_report = self.tokenized_dataset[index]["reference_report"]  # str

            # cv2.imread by default loads an image with 3 channels
            # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,0]
            image = cv2.resize(image, (512, 512))


            # apply transformations to image, bbox_coordinates and bbox_labels
            transformed = self.transforms(image=image, bboxes=bbox_coordinates, class_labels=bbox_labels)

            transformed_image = transformed["image"]

            transformed_bbox_coordinates = transformed["bboxes"]
            transformed_bbox_labels = transformed["class_labels"]

            transformed_bbox_coordinates = [[x * 2 for x in bbox] for bbox in transformed_bbox_coordinates]
            sample = {
                "image": transformed_image,
                "bbox_coordinates": torch.tensor(transformed_bbox_coordinates, dtype=torch.float),
                "bbox_labels": torch.tensor(transformed_bbox_labels, dtype=torch.int64),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "bbox_phrase_exists": torch.tensor(bbox_phrase_exists, dtype=torch.bool),
                "bbox_is_abnormal": torch.tensor(bbox_is_abnormal, dtype=torch.bool),
                "bbox_abnormalities": torch.tensor(bbox_abnormalities, dtype=torch.int64),
            }

            if self.dataset_name != "train":
                sample["bbox_phrases"] = bbox_phrases
                sample["reference_report"] = reference_report

        except Exception as e:
            self.log.error(f"__getitem__ failed for: {image_path}")
            self.log.error(f"Reason: {e}")
            return None

        return sample

def get_tokenized_datasets(tokenizer, raw_train_dataset):
    def tokenize_function(example):
        phrases = example["bbox_phrases"]  # List[str]
        bos_token = "<|endoftext|>"  # note: in the GPT2 tokenizer, bos_token = eos_token = "<|endoftext|>"
        eos_token = "<|endoftext|>"

        phrases_with_special_tokens = [bos_token + phrase + eos_token for phrase in phrases]

        # the tokenizer will return input_ids of type List[List[int]] and attention_mask of type List[List[int]]
        return tokenizer(phrases_with_special_tokens, truncation=True, max_length=1024)

    tokenized_train_dataset = raw_train_dataset.map(tokenize_function)

    # tokenized datasets will consist of the columns
    #   - mimic_image_file_path (str)
    #   - bbox_coordinates (List[List[int]])
    #   - bbox_labels (List[int])
    #   - bbox_phrases (List[str])
    #   - input_ids (List[List[int]])
    #   - attention_mask (List[List[int]])
    #   - bbox_phrase_exists (List[bool])
    #   - bbox_is_abnormal (List[bool])
    #
    #   val dataset will have additional column:
    #   - reference_report (str)

    return tokenized_train_dataset

import cv2
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_name: str, tokenized_dataset, transforms, log):
        
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenized_dataset = tokenized_dataset
        self.transforms = transforms
        self.log = log
    

    def __len__(self):
        return len(self.tokenized_dataset)

    def __getitem__(self, index):
   
        # get the image_path for potential logging in except block
        image_path = self.tokenized_dataset[index]["mimic_image_file_path"]

        # if something in __get__item fails, then return None
        # collate_fn in dataloader filters out None values
        bbox_coordinates = self.tokenized_dataset[index]["bbox_coordinates"]  # List[List[int]]
        bbox_labels = self.tokenized_dataset[index]["bbox_labels"]  # List[int]
        input_ids = self.tokenized_dataset[index]["input_ids"]  # List[List[int]]
        attention_mask = self.tokenized_dataset[index]["attention_mask"]  # List[List[int]]
        bbox_phrase_exists = self.tokenized_dataset[index]["bbox_phrase_exists"]  # List[bool]
        bbox_is_abnormal = self.tokenized_dataset[index]["bbox_is_abnormal"]  # List[bool]
        bbox_abnormalities = self.tokenized_dataset[index]["bbox_abnormalities"]  # List[List[int]]


        # if self.dataset_name != "train":
        #     # we only need the reference phrases during evaluation when computing scores for metrics
        #     bbox_phrases = self.tokenized_dataset[index]["bbox_phrases"]  # List[str]

        #     # same for the reference_report
        #     reference_report = self.tokenized_dataset[index]["reference_report"]  # str

        # cv2.imread by default loads an image with 3 channels
        # since we have grayscale images, we only have 1 channel and thus use cv2.IMREAD_UNCHANGED to read in the 1 channel
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)[:,:,0]
        image = cv2.resize(image, (512, 512))


        # apply transformations to image, bbox_coordinates and bbox_labels
        transformed = self.transforms(image=image, bboxes=bbox_coordinates, class_labels=bbox_labels)

        transformed_image = transformed["image"]

        transformed_bbox_coordinates = transformed["bboxes"]
        transformed_bbox_labels = transformed["class_labels"]
     
        transformed_bbox_coordinates = [[x * 2 for x in bbox] for bbox in transformed_bbox_coordinates]
        sample = {
            "image": transformed_image,
            "bbox_coordinates": torch.tensor(transformed_bbox_coordinates, dtype=torch.float),
            "bbox_labels": torch.tensor(transformed_bbox_labels, dtype=torch.int64),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "bbox_phrase_exists": torch.tensor(bbox_phrase_exists, dtype=torch.bool),
            "bbox_is_abnormal": torch.tensor(bbox_is_abnormal, dtype=torch.bool),
            "bbox_abnormalities": bbox_abnormalities,
        }
    
        # if self.dataset_name != "train":
        #     sample["bbox_phrases"] = bbox_phrases
        #     sample["reference_report"] = reference_report



        return sample

raw_train_dataset, raw_test_dataset = get_datasets()

tokenizer = get_tokenizer()
# tokenize the raw datasets
tokenized_train_dataset= get_tokenized_datasets(tokenizer, raw_train_dataset)
train_transforms = get_transforms("train")
train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms, log)
tokenized_val_dataset = get_tokenized_datasets(tokenizer, raw_test_dataset)
val_transforms = get_transforms("val")
val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, log)

checkpoint_path = "/Users/claire/Desktop/Stuff-/codes/dissertation/data/checkpoints/full_model_checkpoint_val_loss_19.793_overall_steps_155252.pt"
model = get_model(checkpoint_path)

train_dataset_complete = CustomDataset("train", tokenized_train_dataset, train_transforms, log)
val_dataset_complete = CustomDataset("val", tokenized_val_dataset, val_transforms, log)

def initial_features(dataset, model):
    def one_hot(lst):
        lst = [0 if x not in lst else 1 for x in range(41)]
        return lst
    features = []
    one_hot_abs= []
    for sample in dataset:
        print(1)
        image = sample['image']
        image = image.unsqueeze(0)
        _, _, feature,_=  model.object_detector(image)
        features.append(feature)
        bbox_abnormalities = sample['bbox_abnormalities']
        one_hot_ab = torch.Tensor([one_hot(x) for x in bbox_abnormalities])
        one_hot_abs.append(one_hot_ab)
    return features, one_hot_abs

# save the features and one_hot_abs

val_features, val_one_hot_abs = initial_features(val_dataset_complete, model)
train_features, train_one_hot_abs = initial_features(train_dataset_complete, model)
# save the features and one_hot_abs
torch.save(train_features, '/Users/claire/Desktop/Stuff-/codes/dissertation/data/train_features.pt')
torch.save(train_one_hot_abs, '/Users/claire/Desktop/Stuff-/codes/dissertation/data/train_one_hot_abs.pt')
torch.save(val_features, '/Users/claire/Desktop/Stuff-/codes/dissertation/data/val_features.pt')
torch.save(val_one_hot_abs, '/Users/claire/Desktop/Stuff-/codes/dissertation/data/val_one_hot_abs.pt')

# load the features and one_hot_abs
train_features = torch.load('/Users/claire/Desktop/Stuff-/codes/dissertation/data/train_features.pt')
train_one_hot_abs = torch.load('/Users/claire/Desktop/Stuff-/codes/dissertation/data/train_one_hot_abs.pt')
val_features = torch.load('/Users/claire/Desktop/Stuff-/codes/dissertation/data/val_features.pt')
val_one_hot_abs = torch.load('/Users/claire/Desktop/Stuff-/codes/dissertation/data/val_one_hot_abs.pt')