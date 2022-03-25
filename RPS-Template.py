import random
import numpy as np
import cv2
from keras.models import load_model
model = load_model('keras_model.h5')
cap = cv2.VideoCapture(0)
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
import time


def game_outcome(user_choice):

    rps_list = [0, 1, 2] # 0-rock 1-paper 2-scissors
    com_chosen = random.choice(rps_list)

    usr_chosen_predict = np.amax(user_choice[0])
    usr_chosen = np.where(user_choice[0] == usr_chosen_predict)[0][0]


    #0-com wins 1-user wins -1-draw 
    if com_chosen == 0 and usr_chosen == 1:
        print("user wins")
        return(1, "User Wins: Computer chose rock, User chose paper")
    elif com_chosen == 0 and usr_chosen == 0:
        print("it's a draw")
        return(-1, "It's a draw: Computer chose rock, User chose rock")
    elif com_chosen == 0 and usr_chosen == 2:
        print("computer wins")
        return(0, "Computer Wins: Computer chose rock, User chose scissors")
    elif com_chosen == 1 and usr_chosen == 0:
        print("computer wins")
        return(0, "Computer Wins: Computer chose paper, User chose rock")
    elif com_chosen == 1 and usr_chosen == 1:
        print("it's a draw")
        return(-1, "It's a draw: Computer chose paper, User chose paper")
    elif com_chosen == 1 and usr_chosen == 2:
        print("user wins")
        return(1, "User Wins: Computer paper rock, User chose scissors")
    elif com_chosen == 2 and usr_chosen == 0:
        print("user wins")
        return(1, "User Wins: Computer chose scissors, User chose rock")
    elif com_chosen == 2 and usr_chosen == 1:
        print("computer wins")
        return(0, "Computer Wins: Computer chose scissors, User chose paper")
    elif com_chosen == 2 and usr_chosen == 2:
        print("it's a draw")
        return(-1, "It's a draw: Computer chose scissors, User chose scissors")
    elif usr_chosen == 3:
        print("computer wins")
        return(0, "Computer wins by default: User did nothing")
    else:
        return(print("invalid"))

flag = 1
user_victories = 0
computer_victories = 0
outcome_text="First Round"

while True: 

    if cv2.waitKey(33) == ord('c') and (computer_victories == 3 or user_victories == 3):
        user_victories = 0
        computer_victories = 0

    if flag == 1:
        initial_time = time.time()
        flag = 0

    final_time = time.time()

    time_diff = final_time-initial_time



    ret, frame = cap.read()
    resized_frame = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
    image_np = np.array(resized_frame)
    normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image
    data[0] = normalized_image

    time_text = str(3-int(time_diff))

    if time_diff < 4 and computer_victories < 3 and user_victories < 3:
        frame = cv2.putText(frame, time_text, (600,450), cv2.FONT_HERSHEY_SIMPLEX, 5, (255,255,255), 5, cv2.LINE_AA)
    elif computer_victories == 3 or user_victories == 3:
        frame = cv2.putText(frame, "Hit C to restart game", (120,450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,255,255), 4, cv2.LINE_AA)

    if flag == 2:
        print(outcome_text)
        frame = cv2.putText(frame, outcome_text, (50,700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        if time_diff>6:
            flag = 1
    frame = cv2.putText(frame, "Computer Score", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "= "+str(computer_victories), (350,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "User Score", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    frame = cv2.putText(frame, "= "+str(user_victories), (350,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    
    
    cv2.imshow("Text", frame)

    
    if time_diff > 4 and flag == 0 and computer_victories < 3 and user_victories < 3:
        prediction = model.predict(data)
        outcome, outcome_text = game_outcome(prediction)
        flag = 2

        if outcome == 0:
            computer_victories += 1
        elif outcome == 1:
            user_victories += 1




    # Press q to close the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
cv2.waitKey(1)