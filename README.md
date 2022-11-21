# Amadeus
## Amadeus is a multi-functional discord bot made by me using python and discord.py.  
I got inspired to make this bot after the music bot 'rhythm' which I used to play music with in my discord server got shut down.
During this time I had been searching for ideas for the final project so I thought why not make a discord bot using the knowledge I gained from CS50? and this is how Amadeus was born!
Along the way, I added some extra features to Amadeus so it can do much more than stream music!
Some useful functions like translation, QRcode generator, and anime search are also available in this bot to help the user.
I've deployed this bot using Daki, a free bot hosting service (https://daki.cc/). The source code is available on GitHub at (https://github.com/Om-Alve/Amadeus).  
I'll now tell you the use of each file in the source code:
* main.py: This file contains all the commands and contains all the primary code which runs the bot. 
* random_responses.py: This file contains the functions which control the chatbot feature of the bot.
* bots.json: This JSON file contains various keywords to search in the input message and their respective response. 
* requirements.txt contains a list of all the required dependencies to run this project.
* pic1.jpg: This is an image file that is a part of the function which welcomes a newly joined member.

***

* The dependencies to install are listed in the requirements.txt file

Amadeus can send welcome messages to the server whenever a new member joins the server along with a beautiful image! 

###  *Commands can be made by sending* 

> $(command name)

*Amadeus can also respond to normal messages* 
* The responses and keywords to search in the input message are stored in a json file called bot.json
* For now it can only respond accurately to certain messages however I'm planning on making it AI based later on when I learn about Artificial Intelligence
* The random_responses.py file then returns the responses according to the input given by the user
* However this feature is limited to only a certain channel so as not to interfere with the conversations of the users

### *The commands of Amadeus can be widely categorized into three groups based on genre:*

### ***Utility*** :

#### Here are some commands which can help you out 

>$qrcode (text or link)

* Converts given text or link to a QR code (This uses the API- https://goqr.me/api/ )

>$translate (language code) (text to translate)

* Translates given text to the language required by the user. (Uses the google translate library in python)

>$anime (search keywords)

* Gives information about the searched anime (Uses the Anilist library)

> $kick (@member) (reason(optional))

* Kicks the mentioned member from the server (the author must have the permissions required to kick people) 

> $ban (@member) (reason(optional))

* Bans the mentioned member from the server (the author must have the permissions required to ban people) 

> $avatar (@member(optional))

* Gets the avatar of the mentioned user (if no member is mentioned it gets the avatar of the author)

***

### ***Music*** :
#### The bot can stream music in vc by using the youtubedl and ffmpeg library  
#### You can also queue songs
> $join

* Joins the vc the user is connected to  
> $leave  

* Leaves the vc  

> $play (search keywords or name of the song or url)  

* Plays the searched song or queues the song if there's another song playing     

>$pause

* Pauses the song

>$resume

* Resumes paused song  

>$queue

* Shows current queue  

>$clear 

* Clears the current queue

***

### ***Fun*** :

#### Here are some interesting commands to have some fun while chatting on discord

>$inspire  

* Tells a random quote (Uses a quotes library)

>$meme

* Gives a random meme from r/memes (Uses praw to get memes from the r/memes reddit server)

>$joke

* Tells a joke (Uses a jokes api to get programming jokes *API - https://sv443.net/jokeapi/v2/*)

>$fact

* Tells a random fact (Uses a facts api to get random facts *API -https://api-ninjas.com/api/facts*)

> $roll (range(int))

* Gives a random number in the given range

>$toss

* Simulates a coin toss  

>$compliment

* Gives you a compliment to make your day (Uses a compliment api to get random compliments *API - https://complimentr.com/*)

>$bored

* Gives you a fun activity to do if your bored (Uses an api which gives random activities *API - http://www.boredapi.com/*)

***
### Help

* To get an overview if all the commands you can use the following function

> $help

* If you need any help for any specific command you can use the following command

> $help (command_name)

***

#### Doing this project really helped me learn a lot of  new things which will surely be of great use to me in the future! It also helped improve the fundamentals of python. Developing Amadeus has been a great experience and I'll continue working on it and make improvement in it as I get acquainted with new technologies.

#### I give my thanks to all the people who have helped me develop this bot especially @MeltingDiamond.
