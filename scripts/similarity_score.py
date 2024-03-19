import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import matplotlib.pyplot as plt


def cross_entropy(label_cols, fraction_data, question, question_weight, top_N):
    """Take a vector of pred_frac across all questions for a single galaxy,
    calculate the CE Loss for this galaxy against all others.
    For the specified question, assign a heavier weight to make CE Loss more
    sensitive to this question.
    Sum the CE Loss over all questions
    """
    num_galaxies = fraction_data.shape[0]
    num_questions = len(label_cols)
    f_pred = fraction_data.filter(regex='pred|id_str')
    f_obs = fraction_data.filter(regex='fraction|id_str')
    # specific_question_s, specific_question_e = question_name_dict[question]

    # Initialize the CE Loss matrix
    ce_loss = np.zeros((num_galaxies, num_questions))
    # ce_loss_dict = np.zeros((num_galaxies, len(question_name_dict)))

    # Initialize maximum ce loss list
    max_ce_j = []
    max_ce_j_ques = []

    # Iterate over all galaxies
    for i in range(num_galaxies):
        # Get the predicted fractions for the i-th galaxy
        pred_frac = f_pred.iloc[i,1:]
        true_frac = f_obs.iloc[i,:-1]


        # Iterate over all questions
        for j in range(num_questions):
            # Get the true label for the i-th galaxy and j-th question
            true_frac_j = true_frac.iloc[j]

            # Get the predicted fraction for the i-th galaxy and j-th question
            pred_frac_j = pred_frac.iloc[j]

            # Compute the cross-entropy loss for the i-th galaxy and j-th question
            if pred_frac_j == 0:
               ce = 0
            else:
               ce = - true_frac_j * np.log(pred_frac_j) - (1 - true_frac_j) * np.log(1 - pred_frac_j)

            # Add a heavier weight to the specified question
            if j in question:
                ce *= question_weight
                # print(ce)

            # Add the CE Loss to the matrix
            ce_loss[i][j] = ce

        # Sum over all dictionary questions
        # for key, q in zip(question_name_dict, range(len(question_name_dict))):
        #     sum_range = list(question_name_dict[key])
        #     ce_loss_dict[i][q] = np.sum(ce_loss[i][sum_range[0]:sum_range[1]])


        #Find maximum for this row
        max_ce_j.append(max(ce_loss[i]))
        max_ques_id = np.where(ce_loss[i]==max_ce_j[i])[0].tolist()
        # print(max_ques_id)
        max_ce_j_ques.append(label_cols[max_ques_id[0]])



    #Find maximum for entire dataset
    # galaxy_w_max_ce = fraction_data.iloc[max_ce_j.index(max(max_ce_j)),:]
    # galaxy_w_max_ce_ques = max_ce_j_ques[max_ce_j.index(max(max_ce_j))]
    #top N max ce
    galaxy_indices = sorted(range(len(max_ce_j)), key=lambda i: max_ce_j[i], reverse=True)[:top_N]
    top_N_galaxies = fraction_data.iloc[galaxy_indices,:]
    top_N_ques = [max_ce_j_ques[i] for i in galaxy_indices]


    # Return the CE Loss matrix
    return ce_loss, top_N_galaxies, top_N_ques

def get_top_k_most_similar_CE(weight, specific_question, given_index, is_pred, df, k):
  """
  specific_question: it has to be one of the keys of question_name_dict.
  """

  specific_question_i = df.columns.get_loc(specific_question + "_fraction")

  given_img_arr = df.iloc[given_index].values.squeeze()
  all_img_arr = df.values

  if is_pred:
    specific_question_i += 34
    ce_error = (-1) * np.sum(given_img_arr[34:specific_question_i] * np.log(all_img_arr[:, 34:specific_question_i].astype(float)+ 1e-10) + (1 - given_img_arr[34:specific_question_i]) * np.log(1 - all_img_arr[:, 34:specific_question_i].astype(float)+ 1e-10), axis=1)
    ce_error += (-1) * weight * np.sum(given_img_arr[specific_question_i - 34: specific_question_i + 1 -34] * np.log(all_img_arr[:, specific_question_i: specific_question_i + 1].astype(float) + 1e-10) + (1 - given_img_arr[specific_question_i -34: specific_question_i + 1 -34]) * np.log(1 - all_img_arr[:, specific_question_i: specific_question_i + 1].astype(float) + 1e-10), axis=1)
    ce_error += (-1) * np.sum(given_img_arr[specific_question_i + 1:68] * np.log(all_img_arr[:, specific_question_i + 1:68].astype(float)+ 1e-10) + (1 - given_img_arr[specific_question_i + 1:68]) * np.log(1 - all_img_arr[:, specific_question_i + 1:68].astype(float) + 1e-10), axis=1)
  else:
    # actual
    ce_error = (-1) * np.sum(given_img_arr[:specific_question_i] * np.log(all_img_arr[:, :specific_question_i].astype(float) + 1e-10) + (1 - given_img_arr[:specific_question_i]) * np.log(1 - all_img_arr[:, :specific_question_i].astype(float) + 1e-10), axis=1)
    ce_error += (-1) * weight * np.sum(given_img_arr[specific_question_i + 34: specific_question_i + 1 + 34] * np.log(all_img_arr[:, specific_question_i: specific_question_i + 1].astype(float)+ 1e-10) + (1 - given_img_arr[specific_question_i + 34: specific_question_i + 1 + 34]) * np.log(1 - all_img_arr[:, specific_question_i: specific_question_i + 1].astype(float) + 1e-10), axis=1)
    ce_error += (-1) * np.sum(given_img_arr[specific_question_i + 1:34] * np.log(all_img_arr[:, specific_question_i + 1:34].astype(float) + 1e-10) + (1 - given_img_arr[specific_question_i + 1:34]) * np.log(1 - all_img_arr[:, specific_question_i + 1:34].astype(float) + 1e-10), axis=1)

  sorted_indices = np.argsort(ce_error)
  top_k_indices = sorted_indices[0: k]

  return all_img_arr[top_k_indices, 68]

def similar_galaxies_id(original_lst, question, df):
    pred_lst = []
    actual_lst = []
    for i in range(len(original_lst)):
        pred_lst.append(get_top_k_most_similar_CE(5, question[i], original_lst[i], True, df, 1)[0])
        actual_lst.append(get_top_k_most_similar_CE(5, question[i], original_lst[i], False, df, 1)[0])