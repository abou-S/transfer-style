import torch
import torch.nn as nn
import torchvision.models as models
from utils import device, content_loss, style_loss

# Définir les couches VGG à utiliser pour le contenu et le style
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Fonction pour obtenir le modèle VGG19 avec les couches nécessaires
def get_style_model_and_losses(content_img, style_img, content_layers=content_layers_default, style_layers=style_layers_default):
    # Charger le modèle VGG19 pré-entraîné
    vgg = models.vgg19(pretrained=True).features.to(device).eval()
    
    # Normalisation pour les images d'entrée
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    
    # Dictionnaires pour stocker les pertes et les sorties
    content_losses = []
    style_losses = []
    
    # Créer un modèle séquentiel avec la normalisation comme première couche
    model = nn.Sequential(normalization)
    
    # Compteur pour nommer les couches de convolution
    conv_count = 0
    
    # Parcourir les couches du modèle VGG
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            name = f'conv_{conv_count}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{conv_count}'
            # Remplacer les ReLU en place par des ReLU hors place
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{conv_count}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{conv_count}'
        else:
            raise RuntimeError(f'Layer non reconnu: {layer.__class__.__name__}')
        
        # Ajouter la couche au modèle séquentiel
        model.add_module(name, layer)
        
        # Ajouter les pertes de contenu si nécessaire
        if name in content_layers:
            # Ajouter une couche de perte de contenu après cette couche
            target = model(content_img).detach()
            content_loss_layer = ContentLoss(target)
            model.add_module(f'content_loss_{conv_count}', content_loss_layer)
            content_losses.append(content_loss_layer)
        
        # Ajouter les pertes de style si nécessaire
        if name in style_layers:
            # Ajouter une couche de perte de style après cette couche
            target_feature = model(style_img).detach()
            style_loss_layer = StyleLoss(target_feature)
            model.add_module(f'style_loss_{conv_count}', style_loss_layer)
            style_losses.append(style_loss_layer)
    
    # Supprimer toutes les couches après la dernière couche de perte de contenu ou de style
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    
    model = model[:(i + 1)]
    
    return model, style_losses, content_losses

# Classe pour la normalisation des images
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# Classe pour la perte de contenu
class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target
        self.loss = 0

    def forward(self, input):
        self.loss = content_loss(input, self.target)
        return input

# Classe pour la perte de style
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = target_feature
        self.loss = 0

    def forward(self, input):
        self.loss = style_loss(input, self.target)
        return input