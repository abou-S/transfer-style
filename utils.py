import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Taille d'image par défaut
IMG_SIZE = 512 if torch.cuda.is_available() else 256

# Appareil à utiliser (GPU si disponible, sinon CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations pour prétraiter les images
loader = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # Redimensionner l'image
    transforms.ToTensor(),  # Convertir en tenseur
    transforms.Lambda(lambda x: x.mul(255))  # Multiplier par 255
])

unloader = transforms.ToPILImage()  # Reconvertir en image PIL

# Fonction pour charger une image
def load_image(image_path, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    # Si une forme est spécifiée, redimensionner l'image
    if shape is not None:
        image = image.resize(shape, Image.LANCZOS)
    
    # Ajouter une dimension de lot et déplacer vers l'appareil approprié
    image = loader(image).unsqueeze(0).to(device)
    
    return image

# Fonction pour afficher une image
def show_image(tensor, title=None):
    # Cloner le tenseur pour éviter de modifier l'original
    image = tensor.cpu().clone()
    # Supprimer la dimension de lot
    image = image.squeeze(0)
    # Convertir en image PIL
    image = unloader(image.div(255))
    
    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# Fonction pour enregistrer une image
def save_image(tensor, path):
    # Cloner le tenseur pour éviter de modifier l'original
    image = tensor.cpu().clone()
    # Supprimer la dimension de lot
    image = image.squeeze(0)
    # Convertir en image PIL
    image = unloader(image.div(255))
    # Enregistrer l'image
    image.save(path)

# Fonction pour calculer la perte de contenu
def content_loss(pred, target):
    return F.mse_loss(pred, target)

# Fonction pour calculer la matrice de Gram
def gram_matrix(input):
    batch_size, n_feature_maps, h, w = input.size()
    # Aplatir les dimensions spatiales
    features = input.view(batch_size, n_feature_maps, h * w)
    # Calculer le produit matriciel
    G = torch.bmm(features, features.transpose(1, 2))
    # Normaliser par le nombre d'éléments
    return G.div(n_feature_maps * h * w)

# Fonction pour calculer la perte de style
def style_loss(pred, target):
    G_pred = gram_matrix(pred)
    G_target = gram_matrix(target)
    return F.mse_loss(G_pred, G_target)