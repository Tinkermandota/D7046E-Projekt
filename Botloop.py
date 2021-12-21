import random
import torch
import numpy as np
from spellchecker import SpellChecker
from Bertfjärt import predict_sentiment, model, tokenizer
import re
import time
spell = SpellChecker()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('tut6-model.pt'))  # load in ze model
model = model.to(device)


def spellCheck(text):
    text = text.split(" ")
    misspelled = spell.unknown(text)
    for word in misspelled:
        print(
            f'Sorry, I don\'t recognize the word "{word}". Did you mean "{spell.candidates(word)}"?')
        time.sleep(1)
        new_word = input("Please type the correct word\n")
        text[text.index(word)] = new_word  # WORD?
    return text


def listToString(s):
    # initialize an empty string
    str1 = " "
    # return string
    return (str1.join(s))


yes = ["yes", "yep", "fine", "okay", "okey", "ok", "affirmative", "yeah", "ye", "yeboi", "absolutely", "agreed", "all right", "alright", "assuredly", "aye", "beyond a doubt", "by all means", "certainly",
       "definitely", "gladly", "indubitably", "assuredly", "naturally", "of course", "ofcourse", "positively", "precisely", "sure", "surely", "undoubtedly", "unquestionably", "very well", "willingly"]
no = ["no", "nay", "nix", "never", "not", "negative", "nah", "nope"]


def participation_analysis(text):
    yesno = [False, False]
    for each in yes:
        if each in text:
            yesno[0] = True
    for each in no:
        if each in text:
            yesno[1] = True
    return yesno


intro = ["Hello!", "Good day to you!", "Welcome friend!",
         "Nice! A visitor!", "I'm awake! At last!", "Hello! Welcome!"]
participation_request = ["Would you like to review a movie?", "I am really good at predicting if your movie review is good or bad. Would you like to give it a try?", "If you write a movie review, I will predict if it is good or bad. Sounds fun?",
                         "Please think of a movie you have seen. You will write a review for it, and then I will predict if that review is good or bad. Sounds good?", "As you might know, I am a chatbot that can predict if your movie review is good or bad. Would you like to try it?"]
affirmative = ["Cool!", "Nice!", "Let's go!", "Here we go!",
               "I'm ready!", "Fun! Let's go!", "Cool beans!", "Very nice!", "Great!"]
goodbye = ["Well, if you change your mind, I'm right here...", "Another time then!", "OK cool, see you later then!", "Alright, another time then.",
           "OK, I understand. Another time, friend!", "Unfortunate, another time I guess...", "Ah OK, I understand. Until next time!"]
title_request = ["What is the movie called? Please write the title.", "What movie would you like to review? Please write the title.", "What is the movie you want to review called?",
                 "What is the title of the movie you would like to review?", "Please write the title of the movie you want to review.", "Please write the title of the movie you are reviewing."]
review_request = ["Write a review please!", "Please type your review here!", "Please write a review:", "Now, please write a review:",
                  "Now, please write what you thought of the movie!", "Great! Now write the review...", "Write your review please:"]
review_again = ["Would you like to review another movie?", "Would you like to give it another try?", "Would you like me to predict another review?", "This is fun! Want to go again?",
                "Would you like to try it again?", "Do you want another go?", "Is it OK if we go again?", "Let's go again! Are you in?", "Let's review another movie! Are you in?"]
positive = ["Cool! That was a positive review, you seem to like <TITLE>", "Nice, you seem to have enjoyed <TITLE> quite a bit. That's a good review.", "A good review for a good movie, right? You seem to like that movie.",
            "That's a positive review. <TITLE> must be a pretty good movie.", "I'm glad you liked the movie <TITLE>. Maybe worth checking out?", "Nice review, <TITLE> sounds like a really good movie.", "From your review, that does sound like a pretty good movie, not gonna lie."]
confused = ["Hmm, that's a tough one. Either you weren't too fond of it, or you didn't write a good enough review.", "It is really hard to determine if it is a good or bad review when you are that vague.", "You don't seem to know if you like it or not.",
            "I can't really discern what you thought about the movie.", "You have to be a bit more clear in your review. It is hard to know what you thought of the movie.", "That's too vague for me, sorry.", "Hmm, it's too hard to tell from that review.", "Neutral review? To be honest, I can't tell..."]
negative = ["Ouch, that sounds like a pretty bad movie to be honest. But, not all movies have to be good!", "Sorry to hear that. I take that as a pretty bad review.", "Oh, that definitely sounds like a pretty bad movie.", "From your review, I would say that <TITLE> is a pretty bad movie...",
            "OK, better stay away from <TITLE> then. That was a rather harsh review.", "Yeah alright, from that review it is quite clear that you didn't like the movie <TITLE>...", "Oh, you don't seem to like <TITLE> at all. Better stay away from that one then..."]
repeat_responses = ["Please give it another try!", "You get another shot for the same movie, please try and be more specific!", "Please write the review again. For my sake, please try to be more precise this time!",
                    "Please write the review again. I'm sure I will get it this time!", "Let's give the same movie another shot. Please be slightly more on point time, so I can properly analyze your review!", "You have to write the review again to give me a chance to guess it. Let's go!"]
participation_confused = ["Please be clear"]


print(random.choice(intro))
yesno = participation_analysis(
    input(random.choice(participation_request) + "\n"))
prompt_review = True

while prompt_review:

    if yesno[0] == False and yesno[1] == False:
        yesno = participation_analysis(
            input(random.choice(participation_confused) + "\n"))

    elif yesno[0] and yesno[1]:
        yesno = participation_analysis(
            input(random.choice(participation_confused) + "\n"))

    elif yesno[0]:

        print(random.choice(affirmative))
        movietitle = input(random.choice(title_request) + "\n")
        review = input(random.choice(review_request) + "\n")
        review = review.replace(movietitle + " ", "")
        review = review.replace(" " + movietitle, "")
        repeat = True

        while repeat:

            repeat = False
            # Implement some functionality to remove the movie title in the review
            review = spellCheck(review)
            # Implement some functionality to input the review into the trained network
            result = predict_sentiment(model, tokenizer, listToString(review))
            print(f'predict {result}')

            if result > 0.4 and result < 0.6:
                repeat = True
                print(random.choice(confused))
                review = input(random.choice(repeat_responses) + "\n")

            elif result > 0.6:
                print(random.choice(positive).replace("<TITLE>", movietitle))

            else:
                print(random.choice(negative).replace("<TITLE>", movietitle))

        yesno = participation_analysis(
            input(random.choice(participation_again) + "\n"))
    else:
        prompt_review = False
print(random.choice(goodbye))
