import random
import numpy as np

intro = ["Hello!","Good day to you!","Welcome friend!","Nice! A visitor!","I'm awake! At last!"]
participation_request = ["Would you like to review a movie?","I am really good at predicting if your movie review is good or bad. Would you like to give it a try?","If you write a movie review, I will predict if it is good or bad. Sounds fun?","Please think of a movie you have seen. You will write a review for it, and then I will predict if that review is good or bad. Sounds good?","As you might know, I am a chatbot that can predict if your movie review is good or bad. Would you like to try it?"]
affirmative = ["Cool!","Nice!","Let's go!","Here we go!","I'm ready!"]
goodbye = ["Well, if you change your mind, I'm right...","Another time then!","OK cool, see you later then!","Alright, another time then.","OK, I understand. Another time, friend!"]
title_request = ["What is the movie called? Please write the title.","What movie would you like to review? Please write the title.","What is the movie you want to review called?","What is the title of the movie you would like to review?","Please write the title of the movie you want to review."]
review_request = ["Write a review please!","Please type your review here!","Please write a review:","Now, please write a review:","Now, please write what you thought of the movie!"]
confused = ["Hmm, that's a tough one. Either you weren't too fond of it, or you didn't write a good enough review.","It is really hard to determine if it is a good or bad review when you are that vague.","You don't seem to know if you like it or not.","I can't really discern what you thought about the movie.","You have to be a bit more clear in your review. It is hard to know what you thought of the movie."]
repeat_responses = ["Please give it another try!","You get another shot for the same movie, please try and be more specific!","Please write the review again. For my sake, please try to be more precise this time!","Please write the review again. I'm sure I will get it this time!","Let's give the same movie another shot. Please be slightly more on point time, so I can properly analyze your review!"]
positive = ["Cool! That was a positive review, you seem to like <TITLE>","Nice, you seem to have enjoyed <TITLE> quite a bit. That's a good review.","A good review for a good movie, right? You seem to like that movie.","That's a positive review. <TITLE> must be a pretty good movie.","I'm glad you liked the movie <TITLE>. Maybe worth checking out?"]
negative = ["Ouch, that sounds like a pretty bad movie to be honest. But, not all movie have to be good!","Sorry to hear that. I take that as a pretty bad review.","Oh, that definitely sounds like a pretty bad movie.","From your review, I would say that <TITLE> is a pretty bad movie...","OK, better stay away from <TITLE> then. That was a rather harsh review."]
review_again = ["Would you like to review another movie?","Would you like to give it another try?","Would you like me to predict another review?","This is fun! Want to go again?","Would you like to try it again?"]

print(random.choice(intro))
yesorno = input(random.choice(participation_request) + "\n")
# Implement some functionality that actually analyzes the response
yesorno = True
while yesorno:
    print(random.choice(affirmative))
    movietitle = input(random.choice(title_request) + "\n")
    review = input(random.choice(review_request) + "\n")
    repeat = True
    while repeat:
        repeat = False
        # Implement some functionality to remove the movie title in the review
        # Implement some functionality to input the review into the trained network
        result = np.random.rand()
        if result > 0.4 and result < 0.6:
            repeat = True
            print(random.choice(confused))
            review = input(random.choice(repeat_responses) + "\n")
        elif result > 0.6:
            print(random.choice(positive))
        else:
            print(random.choice(negative))
    
    yesorno = input(random.choice(review_again) + "\n")
    # Implement some functionality that actually analyzes the response
    yesorno = True
print(random.choice(goodbye))