#!/usr/bin/env python3

import os
import argparse

import numpy as np
import pandas as pd
from keras.datasets import cifar10
from keras.models import load_model
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import differential_evolution
from scipy.stats import truncnorm
from IPython.display import clear_output
import pickle

# Custom Networks
from networks.lecun_net import LecunNet
from networks.pure_cnn import PureCnn
from networks.network_in_network import NetworkInNetwork
from networks.resnet import ResNet
from networks.densenet import DenseNet
from networks.wide_resnet import WideResNet
from networks.capsnet import CapsNet

# Helper functions
from helper import plot_image, plot_images, evaluate_models, visualize_attack, attack_stats

model_defs = { 
    'lecun_net': LecunNet,
    'pure_cnn': PureCnn,
    'net_in_net': NetworkInNetwork,
    'resnet': ResNet,
    'densenet': DenseNet,
    'wide_resnet': WideResNet,
    'capsnet': CapsNet
}

parser = argparse.ArgumentParser(description="Attack models on Cifar10")
parser.add_argument('--model', nargs='+', choices=model_defs.keys(), default=model_defs.keys())
parser.add_argument('--pixels', nargs='+', default=[1,2,3], type=int)
parser.add_argument('--maxiter', default=30, type=int)
parser.add_argument('--popsize', default=30, type=int)
parser.add_argument('--samples', default=100, type=int)
parser.add_argument('--targeted', action='store_true')
parser.add_argument('--savedir', default='networks/results/results.pkl')
parser.add_argument('--show_image', action='store_true')
parser.add_argument('--verbose', action='store_true')

args = parser.parse_args()

# Load data and model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
models = [model_defs[m](load_weights=True) for m in args.model]

def perturb_image(x, img):
    # Copy the image to keep the original unchanged
    img = np.copy(img) 
    
    # Split into an array of 5-tuples (perturbation pixels)
    # Make sure to floor the members of x as int types
    pixels = np.split(x.astype(int), len(x) // 5)
    
    # At each pixel's x,y position, assign its rgb value
    for pixel in pixels:
        x_pos, y_pos, *rgb = pixel
        img[x_pos, y_pos] = rgb
    return img

def predict_class(x, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    img_perturbed = perturb_image(x, img)
    prediction = model.predict_one(img_perturbed)[target_class]
    # This function should always be minimized, so return its complement if needed
    return prediction if minimize else 1 - prediction

def attack_success(x, img, target_class, model, targeted_attack=False, verbose=False):
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_image = perturb_image(x, x_test[img])

    confidence = model.predict_one(attack_image)
    predicted_class = np.argmax(confidence)
    
    # If the prediction is what we want (misclassification or 
    # targeted classification), return True
    if (verbose):
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
        (not targeted_attack and predicted_class != target_class)):
        return True

def attack(img, model, target=None, pixel_count=1, 
           maxiter=30, popsize=30, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else y_test[img,0]
    
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0,32), (0,32), (0,256), (0,256), (0,256)] * pixel_count
    
    # Format the predict/callback functions for the differential evolution algorithm
    predict_fn = lambda x: predict_class(
        x, x_test[img], target_class, model, target is None)
    callback_fn = lambda x, convergence: attack_success(
        x, img, target_class, model, targeted_attack, verbose)
    
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=max(1, popsize // pixel_count),
        recombination=1, atol=-1, callback=callback_fn)

    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, x_test[img])
    prior_probs = model.predict_one(x_test[img])
    predicted_probs = model.predict_one(attack_image)
    predicted_class = np.argmax(predicted_probs)
    actual_class = y_test[img,0]
    success = predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    if args.show_image:
        plot_image(attack_image, actual_class, class_names, predicted_class)

    return [model.name, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs, predicted_probs, attack_image]

def attack_all(models, samples=100, pixels=(1,2,3), targeted=False, verbose=False):
    results = []
    for model in models:
        model_results = []
        valid_imgs = correct_imgs[correct_imgs.name == model.name].img
        img_samples = np.random.choice(valid_imgs, samples)
        for pixel_count in pixels:
            for i,img in enumerate(img_samples):
                print(model.name, '- image', img, '-', i+1, '/', len(img_samples))
                targets = [None] if not targeted else range(10)
                
                for target in targets:
                    if (targeted):
                        print('Attacking with target', class_names[target])
                        if (target == y_test[img,0]):
                            continue
                    result = attack(img, model, target, pixel_count, maxiter=args.maxiter, popsize=args.popsize, verbose=verbose)
                    model_results.append(result)
        results += model_results
    return results

network_stats, correct_imgs = evaluate_models(models, x_test, y_test)
correct_imgs = pd.DataFrame(correct_imgs, columns=['name', 'img', 'label', 'confidence', 'pred'])
network_stats = pd.DataFrame(network_stats, columns=['name', 'accuracy', 'param_count'])

print('Starting attack')

results = attack_all(models, samples=args.samples, targeted=args.targeted, pixels=args.pixels, verbose=args.verbose)

print('Saving to', args.savedir)
with open(args.savedir, 'wb') as file:
        pickle.dump(results, file)
