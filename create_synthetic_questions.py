import pandas as pd
import argparse
import nltk
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from datetime import datetime

# %% a pronoun modifier
def pronoun_modifier(text):
    if 'my' in text:
        text = text.replace(' my ', ' your ')

    if 'My' in text:
        text = text.replace('My ', 'Your ')

    if 'I' in text:
        text = text.replace('I ', 'You ')

        if 'am' in text:
            text = text.replace('am ', 'are ')

    if 'me.' in text:
        text = text.replace(' me.', ' you.')

    if ' me ' in text:
        text = text.replace(' me ', ' you ')

    return text

# %% Evaluation
def evaluation_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xIsc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 1:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + sc + '?')
        rfa_dr.append('How' + ' ' + sc + ' ' + I + ' ' + x.lower() + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'how' + ' ' + sc + ' ' + x.lower() + ' ' + I + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'how' + ' ' + sc + ' ' + x.lower() + ' ' + I + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'how' + ' ' + sc + ' ' + x.lower() + ' ' + I + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_dec


# %% Importance
def importance_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xI-important-sc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 2:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + 'important ' + sc + '?')
        rfa_dr.append('How important' + ' ' + I + ' ' + x.lower() + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'how important' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'how important' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' important ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'how important' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' important ' + sc + '?')

        return rfa_int_dec


# %% Feelings
def feelings_rfa(statement, form_request, pre_rfa_ls):
    if any(word in statement.split() for word in ['is', 'are', 'am', 'was', 'were']):
        return evaluation_rfa(statement, form_request, pre_rfa_ls)

    # elif any(word in statement for word in ['make', 'makes', 'made']):

    else:
        statement = pronoun_modifier(statement)
        text = tokenizer.tokenize(statement)
        pos_result = nltk.pos_tag(text)

        # find the verb and its location
        for idx, (word, pos) in enumerate(pos_result):
            if pos.startswith('V'):
                break
        # represent sentence structure as xIsc
        I = word
        x = ' '.join(text[0:idx])
        sc = ' '.join(text[idx + 1:])

        # direct request
        if form_request == 'dr':
            rfa_dr = []
            rfa_dr.append('Do' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_dr.append('How much do' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

            return rfa_dr

        # imperative-interrogative
        elif form_request == 'imp_int':
            rfa_imp_int = []
            for pre_rfa in pre_rfa_ls:
                rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I  + ' ' + sc + '.')
                rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
                rfa_imp_int.append(pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

            return rfa_imp_int

        # interrogative-interrogative
        elif form_request == 'int_int':
            rfa_int_int = []
            for pre_rfa in pre_rfa_ls:
                rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
                rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
                rfa_int_int.append(pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

            return rfa_int_int

        # declarative-interrogative
        elif form_request == 'dec_int':
            rfa_dec_int = []
            for pre_rfa in pre_rfa_ls:
                rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
                rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
                rfa_dec_int.append(pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

            return rfa_dec_int

        # interrogative-declarative
        elif form_request == 'int_dec':
            rfa_int_dec = []
            for pre_rfa in pre_rfa_ls:
                rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

            return rfa_int_dec


# %% Cognitive judgments
def judgments_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xIsc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 1:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + sc + '?')
        rfa_dr.append('What' + ' ' + I + ' ' + x.lower() + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'what' + ' ' + x.lower() + ' ' + I + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'what' + ' ' + x.lower() + ' ' + I + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'what' + x.lower() + ' ' + I + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_dec


# %% Causal relationships
def causal_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if word in ['has', 'have']:
            break
    # represent sentence structure as x-have/has-made-sc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 2:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + 'made' + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' made ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' made ' + sc + '?')

        return rfa_int_dec


# %% Similarity relationship
def similarity_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xI-sim_word-sc
    I = word
    x = ' '.join(text[0:idx])
    sim_word = text[idx + 1]
    sc = ' '.join(text[idx + 2:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + sim_word + ' ' + sc + '?')
        rfa_dr.append('How ' + sim_word + ' ' + I + ' ' + x.lower() + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'how ' + sim_word + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'how ' + sim_word + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'how ' + sim_word + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sim_word + ' ' + sc + '?')

        return rfa_int_dec


# %% Preference
def preference_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xI-sim_word-sc
    I = word
    x = ' '.join(text[0:idx])
    pref_word = text[idx + 1]
    sc = ' '.join(text[idx + 2:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + pref_word + ' ' + sc + '?')
        rfa_dr.append('How much ' + I + ' ' + x.lower() + ' ' + pref_word + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')
            rfa_imp_int.append(
                pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '?')
            rfa_int_int.append(
                pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')
            rfa_dec_int.append(
                pre_rfa + ' ' + 'how much' + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + pref_word + ' ' + sc + '?')

        return rfa_int_dec


# %% Norms
def norms_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if word == 'should':
            break
    # represent sentence structure as xIsc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 1:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_dec


# %% Policies
def policies_rfa(statement, form_request, pre_rfa_ls):
    return (norms_rfa(statement, form_request, pre_rfa_ls))


# %% Rights (are allowed to)
def rights_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if pos.startswith('V'):
            break
    # represent sentence structure as xI-sim_word-sc
    I = word
    x = ' '.join(text[0:idx])
    rights_word = text[idx + 1]
    sc = ' '.join(text[idx + 2:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + rights_word + ' ' + sc + '?')
        rfa_dr.append('To what extent ' + I + ' ' + x.lower() + ' ' + rights_word + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(
                pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')
            rfa_imp_int.append(
                pre_rfa + ' ' + 'to what extent' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(
                pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '?')
            rfa_int_int.append(
                pre_rfa + ' ' + 'to what extent' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(
                pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')
            rfa_dec_int.append(
                pre_rfa + ' ' + 'to what extent' + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + rights_word + ' ' + sc + '?')

        return rfa_int_dec


# %% Action tendencies (will...)
def action_rfa(statement, form_request, pre_rfa_ls):
    statement = pronoun_modifier(statement)
    text = tokenizer.tokenize(statement)
    pos_result = nltk.pos_tag(text)

    # find the verb and its location
    for idx, (word, pos) in enumerate(pos_result):
        if word == 'will':
            break
    # represent sentence structure as xIsc
    I = word
    x = ' '.join(text[0:idx])
    sc = ' '.join(text[idx + 1:])

    # direct request
    if form_request == 'dr':
        rfa_dr = []
        rfa_dr.append(I.capitalize() + ' ' + x.lower() + ' ' + sc + '?')

        return rfa_dr

    # imperative-interrogative
    elif form_request == 'imp_int':
        rfa_imp_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_imp_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_imp_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_imp_int

    # interrogative-interrogative
    elif form_request == 'int_int':
        rfa_int_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')
            rfa_int_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_int

    # declarative-interrogative
    elif form_request == 'dec_int':
        rfa_dec_int = []
        for pre_rfa in pre_rfa_ls:
            rfa_dec_int.append(pre_rfa + ' ' + 'whether' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')
            rfa_dec_int.append(pre_rfa + ' ' + 'if' + ' ' + x.lower() + ' ' + I + ' ' + sc + '.')

        return rfa_dec_int

    # interrogative-declarative
    elif form_request == 'int_dec':
        rfa_int_dec = []
        for pre_rfa in pre_rfa_ls:
            rfa_int_dec.append(pre_rfa + ' ' + x.lower() + ' ' + I + ' ' + sc + '?')

        return rfa_int_dec


# %% Expectation of future events (will)
def expectation_rfa(statement, form_request, pre_rfa_ls):
    return (action_rfa(statement, form_request, pre_rfa_ls))


# %% Evaluative belief (verb not be)
def belief_rfa(statement, form_request, pre_rfa_ls):
    return (feelings_rfa(statement, form_request, pre_rfa_ls))






# a function to generate synthetic questions in different formats, based on the type of basic concept
# %% here it goes: basic_concept, group_id
def synthetic_generator(datafile, form_request_ls):
    n_statement = len(datafile)

    pre_rfa_imp_int = ['Tell me',
                       'Specify',
                       'Please tell me',
                       'Please specify']

    pre_rfa_int_int = ['Will you tell me',
                       'Can you tell me',
                       'Can you please tell me',
                       'Could you tell me',
                       'Could you please tell me',
                       'Would you tell me',
                       'Would you please tell me',
                       'Would you like to tell me',
                       'would you mind telling me',
                       'Would you be so kind as to tell me']

    pre_rfa_dec_int = ['I ask you',
                       'I would like to ask you']

    pre_rfa_int_dec = ['Do you think that',
                       'Do you believe that',
                       'Do you agree or disagree that']

    rfa_dict = {}
    for i in range(n_statement):
        basic_concept = datafile['basic_concept'][i]
        statement = datafile['statement'][i]
        rfa_dict[i + 1] = {}

        if basic_concept == 'evaluation':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = evaluation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = evaluation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = evaluation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = evaluation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = evaluation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)
        elif basic_concept == 'importance':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = importance_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = importance_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = importance_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = importance_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = importance_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'feelings':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = feelings_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = feelings_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = feelings_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = feelings_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = feelings_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'cognitive judgment':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = judgments_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = judgments_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = judgments_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = judgments_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = judgments_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'causal relationship':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = causal_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = causal_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = causal_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = causal_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = causal_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'similarity':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = similarity_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = similarity_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = similarity_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = similarity_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = similarity_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'preferences':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = preference_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = preference_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = preference_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = preference_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = preference_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'norms':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = norms_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = norms_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = norms_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = norms_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = norms_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'policies':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = policies_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = policies_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = policies_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = policies_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = policies_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'rights':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = rights_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = rights_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = rights_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = rights_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = rights_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'action tendencies':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = action_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = action_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = action_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = action_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = action_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'expectation':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = expectation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = expectation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = expectation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = expectation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = expectation_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

        elif basic_concept == 'belief':
            for form in form_request_ls:
                if form == 'dr':
                    rfa_dict[i + 1][form] = belief_rfa(statement, form_request=form,
                                                           pre_rfa_ls=None)
                elif form == 'imp_int':
                    rfa_dict[i + 1][form] = belief_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_imp_int)
                elif form == 'int_int':
                    rfa_dict[i + 1][form] = belief_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_int)
                elif form == 'dec_int':
                    rfa_dict[i + 1][form] = belief_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_dec_int)
                elif form == 'int_dec':
                    rfa_dict[i + 1][form] = belief_rfa(statement, form_request=form,
                                                           pre_rfa_ls=pre_rfa_int_dec)

    return rfa_dict


# %% Generate synthetic questions
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datafile", type=str, default='./data/synthetic/Synthetic_Questions_Controlled_20210611.xlsx')

    args = parser.parse_args()
    datafile = args.datafile
    controlled_questions_df = pd.read_excel(datafile)

    form_ls = ['dr', 'imp_int', 'int_int', 'dec_int', 'int_dec']

    # pre texts
    pre_rfa_imp_int = ['Tell me',
                       'Specify',
                       'Please tell me',
                       'Please specify']

    pre_rfa_int_int = ['Will you tell me',
                       'Can you tell me',
                       'Can you please tell me',
                       'Could you tell me',
                       'Could you please tell me',
                       'Would you tell me',
                       'Would you please tell me',
                       'Would you like to tell me',
                       'would you mind telling me',
                       'Would you be so kind as to tell me']

    pre_rfa_dec_int = ['I ask you',
                       'I would like to ask you']

    pre_rfa_int_dec = ['Do you think that',
                       'Do you believe that',
                       'Do you agree or disagree that']

    # the form of requests
    direct_instruction = ['Indicate your opinion about',
                          'Give your opinion about',
                          'Show your opinion about',
                          'Evaluate',
                          'Provide your opinion about']

    direct_request = ['reverse',
                      'How']

    imperative_interrogative = ['Tell me,',
                                'Specify,',
                                'Please tell me,',
                                'Please specify,']

    interrogative_interrogative = ['Will you tell me',
                                   'Can you tell me',
                                   'Can you please tell me',
                                   'Could you tell me',
                                   'Could you please tell me',
                                   'Would you tell me',
                                   'Would you please tell me',
                                   'Would you like to tell me',
                                   'would you mind telling me',
                                   'Would you be so kind as to tell me']

    declarative_interrogative = ['I ask you',
                                 'I would like to ask you']

    interrogative_declarative = ['Do you think that',
                                 'Do you believe that',
                                 'Do you agree or disagree that']

    synthetic_rfas = synthetic_generator(datafile=controlled_questions_df,
                                         form_request_ls=form_ls)

    synthetic_rfas = pd.DataFrame.from_dict(synthetic_rfas, orient='index')
    synthetic_rfas['question_id'] = synthetic_rfas.index

    synthetic_rfas_long = pd.melt(synthetic_rfas,
                                  id_vars=['question_id'],
                                  var_name='form_request',
                                  value_name='rfa')

    now = datetime.now()
    now = now.strftime("%Y%m%d%H%M%S")
    save_path = './data/synthetic/Synthetic_Questions_Controlled_Variants_' + now + '.xlsx'


    synthetic_rfas_long.explode('rfa').reset_index(drop=True).to_excel(
        save_path,
        index=True,
        index_label='row_id')

if __name__ == '__main__':
    main()