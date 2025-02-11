from typing import List
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTFeatureExtractor, AutoTokenizer

mean = torch.tensor([0.485, 0.456, 0.406])[:,None,None]
std = torch.tensor([0.229, 0.224, 0.225])[:,None,None]

class Dataset4M(Dataset):
    def __init__(
            self, 
            task,
            data, 
            modalities,
            image_size,
            img_model_name,
            split='train',
            with_diagnoses=True,
            root=Path('./data'),
            max_length=128
        ):
        self.task = task
        self.data = data[data.split==split]
        if len(modalities) == 1 and modalities[0] == 'ecg':
            self.data = self.data.dropna(subset=['last_ecg_id'])
        self.root = root
        self.split = split
        self.modalities = modalities
        self.with_diagnoses = with_diagnoses

        self.tokenizer_img = ViTFeatureExtractor.from_pretrained(img_model_name) # 'google/vit-base-patch16-224')
        self.cxr_train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=45, scale=(.85, 1.15), shear=0, translate=(0.15, 0.15))
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.text_tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.max_length = max_length
        self.empty_ecg_path = self.root / 'ecgs' / 'empty_ecg.png'

        self.phenotyping_columns = [
            'Acute and unspecified renal failure',
            'Acute cerebrovascular disease',
            'Acute myocardial infarction',
            'Cardiac dysrhythmias',
            'Chronic kidney disease',
            'Chronic obstructive pulmonary disease and bronchiectasis',
            'Complications of surgical procedures or medical care',
            'Conduction disorders',
            'Congestive heart failure; nonhypertensive',
            'Coronary atherosclerosis and other heart disease',
            'Diabetes mellitus with complications',
            'Diabetes mellitus without complication',
            'Disorders of lipid metabolism',
            'Essential hypertension',
            'Fluid and electrolyte disorders',
            'Gastrointestinal hemorrhage',
            'Hypertension with complications and secondary hypertension',
            'Other liver diseases',
            'Other lower respiratory disease',
            'Other upper respiratory disease',
            'Pleurisy; pneumothorax; pulmonary collapse',
            'Pneumonia (except that caused by tuberculosis or sexually transmitted disease)',
            'Respiratory failure; insufficiency; arrest (adult)',
            'Septicemia (except in labor)',
            'Shock'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        if self.task == 'phenotyping':
            label = sample[self.phenotyping_columns].astype(float).tolist()
        else:
            label = [sample.y_true]
        stay_id = sample.stay_id
        
        text = sample.demographics_text
        if self.with_diagnoses:
            text += ' ' + sample[f'icd_text_{self.task}']
        if type(sample.ecg_text) == str:
            text += ' ' + sample.ecg_text
        if type(sample.cxr_text) == str:
            text += ' ' + sample.cxr_text
        if type(sample.med_text) == str:
            text += ' ' + sample.med_text
        
        data = {'y': torch.tensor(label, dtype=torch.float32)}
        if 'lab' in self.modalities:
            lab_img_file = self.root / 'labs2' / self.split / f'{stay_id}.png'
            lab_img = Image.open(lab_img_file).convert('RGB')
            # lab_img = self.tokenizer_img(lab_img)['pixel_values'][0]
            lab_img = self.img_transform(lab_img)
            data['lab'] = lab_img
        
        if 'ecg' in self.modalities:
            ecg_img_file = self.root / 'ecgs2' / self.split / f'{stay_id}.png'
            if not ecg_img_file.exists():
                ecg_img_file = self.empty_ecg_path
            
            ecg_img = Image.open(ecg_img_file).convert('RGB')
            # ecg_img = self.tokenizer_img(ecg_img)['pixel_values'][0]
            ecg_img = self.img_transform(ecg_img)
            # if type(sample.ecg_text) == str:
            #     text += ' ' + sample.ecg_text
            data['ecg'] = ecg_img

        if 'cxr' in self.modalities:
            cxr_img_file = self.root / 'cxrs' / self.split / f'{stay_id}.jpg'
            cxr_img = Image.open(cxr_img_file).convert('RGB')
            # if self.split == 'train':
            #     cxr_img = self.cxr_train_transform(cxr_img)
            # cxr_img = self.tokenizer_img(cxr_img)['pixel_values'][0]
            cxr_img = self.img_transform(cxr_img)
            # if type(sample.cxr_text) == str:
            #     text += ' ' + sample.cxr_text
            data['cxr'] = cxr_img
        
        if 'med' in self.modalities:
            med_img_file = self.root / 'meds2' / self.split / f'{stay_id}.png'
            med_img = Image.open(med_img_file).convert('RGB')
            # med_img = self.tokenizer_img(med_img)['pixel_values'][0]
            med_img = self.img_transform(med_img)
            # if type(sample.med_text) == str:
            #     text += ' ' + sample.med_text
            data['med'] = med_img

        # print(text)
        text_inputs = self.text_tokenizer(
            text, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt', 
            truncation=True
        )
        data['input_ids'] = text_inputs['input_ids'].squeeze()
        data['attention_mask'] = text_inputs['attention_mask'].squeeze()

        return data

    def _text(self, idx):
        sample = self.data.iloc[idx]
        text = sample.demographics_text2
        # text += ' ' + sample.icd_text
        # if 'ecg' in self.modalities and type(sample.ecg_text) == str:
        #     text += ' ' + sample.ecg_text
        # if 'cxr' in self.modalities and type(sample.cxr_text) == str:
        #     text += ' ' + sample.cxr_text
        # if 'med' in self.modalities and type(sample.med_text) == str:
        #     text += ' ' + sample.med_text
        return text
    
def dataloaders(
        task: str,
        modalities: List[str],
        image_size: int = 224,
        img_model_name: str = 'microsoft/swin-base-patch4-window7-224-in22k',
        batch_size: int = 16,
        root: str = './data',
        with_diagnoses: bool = True
        # cache_dir: str = './cache'
) -> List[DataLoader]:
    
    root = Path(root)
    df = pd.read_csv(root / 'meta_pheno.csv', index_col=0)

    trainset = Dataset4M(
        task=task,
        data=df,
        split='train',
        modalities=modalities,
        image_size=image_size,
        img_model_name=img_model_name,
        root=root,
        with_diagnoses=with_diagnoses,
        max_length=512
    )
    valset = Dataset4M(
        task=task,
        data=df,
        split='val',
        modalities=modalities,
        image_size=image_size,
        img_model_name=img_model_name,
        root=root,
        with_diagnoses=with_diagnoses,
        max_length=512
    )
    testset = Dataset4M(
        task=task,
        data=df,
        split='test',
        modalities=modalities,
        image_size=image_size,
        img_model_name=img_model_name,
        root=root,
        with_diagnoses=with_diagnoses,
        max_length=512
    )

    # #####################
    # import random
    # def worker_init_fn(worker_id):
    #     seed = 42 + worker_id
    #     np.random.seed(seed)
    #     random.seed(seed)
        
    # g = torch.Generator()
    # g.manual_seed(42)
    # #####################

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)#, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    valloader = DataLoader(valset, batch_size=batch_size)#, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    testloader = DataLoader(testset, batch_size=batch_size)#, num_workers=0, worker_init_fn=worker_init_fn, generator=g)
    return trainloader, valloader, testloader