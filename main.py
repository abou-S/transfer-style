import torch
import torch.optim as optim
import argparse
import os
from model import get_style_model_and_losses
from utils import load_image, show_image, save_image, device

def style_transfer(content_img_path, style_img_path, output_path, num_steps=300, style_weight=1000000, content_weight=1):
    # Charger les images de contenu et de style
    content_img = load_image(content_img_path)
    style_img = load_image(style_img_path)
    
    # Créer une image d'entrée (initialisée avec l'image de contenu)
    input_img = content_img.clone()
    
    # Obtenir le modèle et les pertes
    model, style_losses, content_losses = get_style_model_and_losses(content_img, style_img)
    
    # Optimiseur LBFGS
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    
    print('Optimisation en cours...')
    run = [0]
    
    # Fonction de fermeture pour l'optimiseur
    def closure():
        # Corriger les valeurs de l'image d'entrée
        input_img.data.clamp_(0, 255)
        
        # Réinitialiser les gradients
        optimizer.zero_grad()
        
        # Passer l'image d'entrée à travers le modèle
        model(input_img)
        
        # Calculer les pertes
        style_score = 0
        content_score = 0
        
        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss
        
        # Pondérer les pertes
        style_score *= style_weight
        content_score *= content_weight
        
        # Calculer la perte totale
        loss = style_score + content_score
        loss.backward()
        
        run[0] += 1
        if run[0] % 50 == 0:
            print(f"Itération {run[0]}:")
            print(f'Perte de style : {style_score.item():.4f}')
            print(f'Perte de contenu : {content_score.item():.4f}')
            print(f'Perte totale : {loss.item():.4f}\n')
        
        return loss
    
    # Exécuter l'optimisation
    for i in range(num_steps):
        optimizer.step(closure)
    
    # Corriger les valeurs finales de l'image d'entrée
    input_img.data.clamp_(0, 255)
    
    # Enregistrer l'image résultante
    save_image(input_img, output_path)
    print(f"Image de transfert de style enregistrée sous '{output_path}'")

def main():
    # Analyser les arguments de ligne de commande
    parser = argparse.ArgumentParser(description='Transfert de style avec PyTorch')
    parser.add_argument('--content', type=str, required=True, help='Chemin vers l\'image de contenu')
    parser.add_argument('--style', type=str, required=True, help='Chemin vers l\'image de style')
    parser.add_argument('--output', type=str, default='output.jpg', help='Chemin pour l\'image de sortie')
    parser.add_argument('--steps', type=int, default=300, help='Nombre d\'étapes d\'optimisation')
    parser.add_argument('--style-weight', type=float, default=1000000, help='Poids pour la perte de style')
    parser.add_argument('--content-weight', type=float, default=1, help='Poids pour la perte de contenu')
    args = parser.parse_args()
    
    # Vérifier si les fichiers d'entrée existent
    if not os.path.isfile(args.content):
        print(f"Erreur: Le fichier d'image de contenu '{args.content}' n'existe pas.")
        return
    if not os.path.isfile(args.style):
        print(f"Erreur: Le fichier d'image de style '{args.style}' n'existe pas.")
        return
    
    # Exécuter le transfert de style
    style_transfer(
        args.content,
        args.style,
        args.output,
        num_steps=args.steps,
        style_weight=args.style_weight,
        content_weight=args.content_weight
    )

if __name__ == '__main__':
    main()