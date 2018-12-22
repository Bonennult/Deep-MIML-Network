# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 13:14:10 2018

@author: Ljy
"""
from dataset.NMF import transformer_nmf

def nmf_fixed(Audio_stft, Bases):
    # Audio_stft: stft for audio sourse 2401*402,V
    # Bases: Trained Bases for detected objects,2401*J ,W
    # Return H
    J_num = Bases.shape[1]
    NMF_model = transformer_nmf.TransformerNMF(input_matrix=Audio_stft, num_components=J_num,
                                               template_dictionary=Bases,distance_measure='kl_divergence',
                                               should_update_template=False,seed=2018, should_do_epsilon=False,
                                                     max_num_iterations=50)
    Activation_matrix = NMF_model.activation_matrix
    return Activation_matrix
    