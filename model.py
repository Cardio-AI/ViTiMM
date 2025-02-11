import torch
import torch.nn as nn
import timm
from transformers import ViTModel, AutoModel, SwinModel

class MultiModalTransformer(nn.Module):
    def __init__(self, task, img_model_name, img_modalities, with_text):
        super().__init__()
        
        if 'vit' in img_model_name:
            feature_extractor = ViTModel
            self.vision_embed_dim = 768
        elif img_model_name == 'microsoft/swin-base-patch4-window7-224-in22k':
            feature_extractor = SwinModel
            self.vision_embed_dim = 1024
        elif img_model_name == 'microsoft/swin-large-patch4-window12-384-in22k':
            feature_extractor = SwinModel
            self.vision_embed_dim = 1536

        self.img_models = nn.ModuleDict({
            mod: feature_extractor.from_pretrained(img_model_name, output_attentions=True)
            for mod in img_modalities
        })

        self.with_text = with_text
        self.text_model = AutoModel.from_pretrained('roberta-base', output_attentions=True)

        self.text_embed_dim = 768
        self.projection_dim = 512
        projection_in_dim = self.vision_embed_dim*len(img_modalities)
        if with_text:
            projection_in_dim += self.text_embed_dim
        self.num_classes = 25 if task == 'phenotyping' else 1
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(projection_in_dim, self.projection_dim, bias=False)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.projection_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

    def forward(self, data, return_attention=False):
        image_embeds, attentions = [], {}
        for mod, vit in self.img_models.items():
            vision_outputs = vit(data[mod])
            # image_embeds_modality = vision_outputs[0][:,0,:]
            image_embeds_modality = vision_outputs[1]
            image_embeds.append(image_embeds_modality)
            if return_attention:
                attentions[mod] = [a.detach() for a in vision_outputs.attentions]
        vl_embeds = torch.cat(image_embeds, dim=1)
        
        if self.with_text:
            text_outputs = self.text_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
            text_embeds = text_outputs[0][:,0,:]
            vl_embeds = torch.cat([vl_embeds, text_embeds], dim=1)
            if return_attention:
                attentions['text'] = [a.detach() for a in text_outputs.attentions]
        
        # vl_embeds = torch.concat([*image_embeds, text_embeds], dim=-1)
        pooled_output = self.dropout(vl_embeds)
        pooled_output = self.projection(vl_embeds)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # classification logits
        logits = self.classifier(pooled_output)
        if return_attention:
            return logits, attentions
        else:
            return logits

    
class MultiModalConv(nn.Module):
    def __init__(self, task, img_modalities, with_text, img_model_name=None):
        super().__init__()

        self.img_models = nn.ModuleDict()
        for mod in img_modalities:
            feature_extractor = timm.create_model('convnext_base', pretrained=True) # 'convnext_base'
            feature_extractor.head = nn.Identity()
            self.img_models[mod] = feature_extractor

        self.with_text = with_text
        self.text_model = AutoModel.from_pretrained('roberta-base', output_attentions=True)
        
        self.vision_embed_dim = 1024
        self.text_embed_dim = 768
        self.projection_dim = 512
        projection_in_dim = self.vision_embed_dim*len(img_modalities)
        if with_text:
            projection_in_dim += self.text_embed_dim
        self.num_classes = 25 if task == 'phenotyping' else 1
        self.dropout = nn.Dropout(0.1)
        self.projection = nn.Linear(projection_in_dim, self.projection_dim, bias=False)
        self.activation = nn.Tanh()
        self.classifier = nn.Linear(self.projection_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
    
    def forward(self, data, return_attention=False):
        image_embeds, attentions = [], {}
        for mod, cnn in self.img_models.items():
            vision_outputs = cnn(data[mod])
            # image_embeds_modality = vision_outputs[0][:,0,:]
            image_embeds_modality = vision_outputs.mean(dim=(2,3))
            image_embeds.append(image_embeds_modality)
        vl_embeds = torch.cat(image_embeds, dim=1)
        
        if self.with_text:
            text_outputs = self.text_model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
            text_embeds = text_outputs[0][:,0,:]
            vl_embeds = torch.cat([vl_embeds, text_embeds], dim=1)
        
        # vl_embeds = torch.concat([*image_embeds, text_embeds], dim=-1)
        pooled_output = self.dropout(vl_embeds)
        pooled_output = self.projection(vl_embeds)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # classification logits
        logits = self.classifier(pooled_output)
        return logits
