import os
import openai
import json

openai_llms = ["gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-3.5-turbo-0125"]
# JSON for the locate scene function, allowing the API to call it
LOCATE_SCENE = {
            "type": "function",
            "function": {
                "name": "locate_scene",
                "description": "Labels first sentence of the scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene_id": {
                            "type": "integer",
                            "description": "sentence ID of first sentence in scene. If -1, then no scene was found.",
                        }
                    },
                    "required" : ['scene_id']
                },
            },
            "tool_choice": "required"
        }

LOCATE_SCENE_EXPLANATION = {
            "type": "function",
            "function": {
                "name": "locate_scene",
                "description": "Labels first sentence of the scene.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scene_id": {
                            "type": "integer",
                            "description": "sentence ID of first sentence in scene. If -1, then no scene was found.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "A 1-2 sentence explanation for the choice of scene_id",
                        }
                    },
                    "required" : ['scene_id', 'explanation']
                },
            },
            "tool_choice": "required"
        }


"""This is a face of the bot, standardized to use any LLM so long as the wrapper is properly implemented"""

class Bot:
    def __init__(self, 
                 init_message="You're a bot designed to determine if there is a scene transition in a group of consecutive sentences. \
The user will input a list of sentences labeled with the sentence ID's. \
Call the \'locate_scene\' function, passing in the sentence ID of first sentence of the scene, or -1 if no scene \
transition was found. A scene transition is defined as a significant change in location, characters, action, or time. \
Limit your text response to describing your logic in 1 sentence. Do not assume that 0 is a transition, as you do not know the context.", 
                model="gpt-4o-2024-05-13"):
        
        self.init_message = init_message
        self.st_memory = []
        self.load_credentials()
        self.inject_message(init_message, "system")

        self.llm = model
        self.API_calls = 0
        self.prompt_tokens_total = 0
        self.response_tokens_total = 0

    def wipe_st(self): #leaves only the init message behind
        self.st_memory = []
        self.seen_memories = []
        self.inject_message(self.init_message, "system")

    #determine if/where there is a scene in a list of sentences
    def check_scene(self, sentences):

        #Format the sentences into an easy to digest format for the LLM
        sentences_with_ids = [[i, sentence] for i, sentence in enumerate(sentences)]
        formatted_string = "\n".join([f"{entry[0]}: {entry[1]}" for entry in sentences_with_ids])

        text_response, tool_response = self.respond_to_input(formatted_string, tools=[LOCATE_SCENE])

        args = self.extract_argument(tool_response[0], ['scene_id'])
        scene_id = args['scene_id']

        self.locate_scene(scene_id)

        self.wipe_st()

        return self.scene_id, self.scene_found
    
      #used by the bot to inform user of where scene transition is
    def locate_scene(self, scene_id):
        self.scene_id = scene_id
        if scene_id == -1: #no scene transition
            self.scene_found = False
        else:
            self.scene_found = True

    
    ### The next 3 functions may seem uncessarily abstract from the task. That is because they are 
    ### from a framework that allows for easy use of LLM's in a large number of situations, such as this one. 

    #Injects input message and generate response into st memory. Useful for conversations
    def respond_to_input(self, message, role="user", tools=[]):
        self.inject_message(message, role) #add message to st memory
        text_response, tool_response = self.generate_response(tools)
        self.inject_message(text_response, "assistant") #remember what it said (only text output)
        if tools:
            return text_response, tool_response #only return tool response if there were available tools
        return text_response
        
    #Respond to current short term memory. Useful for more precise control
    def generate_response(self, tools=[]):
        return self.use_llm(self.st_memory, tools)

    #injects a message directly into the bots short term memory without prompting a response
    def inject_message(self, message, role="system"):
        formatted = {"role": f'{role}', "content": f'{message}'}
        self.st_memory.append(formatted)

    def load_credentials(self):
        if os.path.exists("APIkey.py"):
            try:
                import APIkey
            except ImportError:
                print("Failed to import APIkey.py. Manually input key and org ID for openAI API.")
        else:
            openai.api_key = input("Enter API Key: ")
            openai.organization = input("Enter Organization ID: ")

    #Wrapper function, allows for other LLM's to be used, local or otherwise
    def use_llm(self, st_memory, tools=[]):

        if self.llm in openai_llms:
            if len(tools) > 0:
                response = openai.ChatCompletion.create(
                model=self.llm,
                messages=st_memory,
                tools=tools)
            else:
                response = openai.ChatCompletion.create(
                model=self.llm,
                messages=st_memory)
            response_text = response["choices"][0]["message"]["content"]
            tool_calls = response["choices"][0]["message"]["tool_calls"]
            self.count_response(response)
        else:
            response = "Error: llm is invalid"

        
        return response_text, tool_calls
    
    def count_response(self, response): #useful for determining cost of your bot
        self.prompt_tokens_total += response["usage"]["prompt_tokens"]
        self.response_tokens_total += response["usage"]["completion_tokens"]
        self.API_calls += 1

    def extract_argument(self, json_str, arg_names):
        """
        Extracts the specified arguments from the given JSON string representing an OpenAI function call.

        Parameters:
        json_str (str): JSON string of the OpenAI function call.
        arg_names (list): List of argument names to extract.

        Returns:
        dict: Dictionary with argument names as keys and their values. If an argument is not found, its value will be None.
        """
        try:
            # Parse the JSON string into a dictionary
            openai_function_call = json.loads(str(json_str))
            
            # Extract the 'arguments' field which is a string
            arguments_str = openai_function_call['function']['arguments']
            
            # Parse the 'arguments' string into a dictionary
            arguments = json.loads(arguments_str)
            
            # Initialize the result dictionary
            result = {}
            
            # Extract the specified arguments
            for arg in arg_names:
                if arg in arguments:
                    result[arg] = arguments[arg]
                else:
                    print(f"Error: Argument '{arg}' not found")
                    result[arg] = -1
            
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            for arg in arg_names:
                if arg in arguments:
                    result[arg] = None
            return result
        except KeyError as e:
            print(f"Error: Missing key in JSON - {e}")
            for arg in arg_names:
                if arg in arguments:
                    result[arg] = None
            return result



if __name__ == "__main__":
    bot = Bot("You're a bot designed to determine if there is a scene transition in a group of consecutive sentences. \
    The user will input a list of sentences labeled with the sentence ID's. \
    Call the \'locate_scene\' function, passing in the sentence ID of first sentence of the scene, or -1 if no scene \
    transition was found. A scene transition is defined as a significant change in location, characters, action, or time. \
    Limit your text response to describing your logic in 1 sentence.")

    transition_clear = [
        "As the lights of Tom's house faded in the rear-view mirror, a melancholy settled over Gatsby.",
        "The chapter of that night had closed, but the story of his longing was far from finished.",
        "Now, as we turn the page, the sun rises over West Egg, casting a golden glow on Gatsby's mansion.",
        "The laughter and music of the previous night's revelry have faded into a hushed silence, broken only by the gentle lapping of the waves against the shore.",
        "In the calm of the early morning, Gatsby stands alone on his dock, staring out at the green light that burns unwaveringly in the distance."
    ]

    transition_moderate = [
        "The music had filled the dimly lit room with jazz notes that spiraled up into the smoky ceiling, where conversations tangled with laughter and clinking glasses.",
        "Sal watched as Dean danced between the tables, his movements echoing the frenetic energy of the saxophone.",
        "After a while, the night deepened and the last song trailed off into a quiet hum.",
        "They stepped out into the cool air of the early morning, the empty streets a stark contrast to the warmth of the club.",
        "The city had calmed, its bustling daytime façade replaced by the serene stillness of deserted avenues.",
        "As they walked, the first hints of dawn began to etch the skyline, the night's escapades slowly receding like a dream upon waking."
    ]


    transition_subtle = [
        "Scout watched as Atticus put away his glasses, closing the case with a soft click.",
        "He had been reading in his chair by the window since lunch, the light slowly dimming as the day wore on.",
        "Without a word, he stood and walked to the porch, signaling the end of his respite.",
        "Scout followed, the wooden floorboards creaking under their feet.",
        "Outside, the street was quiet, the Radley house looming silently across the way.",
        "As they sat watching the sunset, the mood shifted gently from the calm introspection of the indoor light to the thoughtful stillness of the twilight hour."
    ]

    non_transition = [
        "Mr. Rochester continued to talk, his voice low and reflective as he recounted tales of his travels.",
        "Jane listened intently, her eyes never straying from his face, captivated by his narratives.",
        "The firelight flickered across the room, casting shadows that danced upon the walls.",
        "Every so often, Mr. Rochester would pause, searching for the right word, his brow furrowing in thought.",
        "Jane waited patiently, her hands folded neatly in her lap, her mind racing with questions about the stories he shared.",
        "He smiled wryly as he remembered a particularly amusing incident, his eyes lighting up with the joy of the memory."
    ]

    falling = [
        "Aiming for a section of the wall near a balcony window, I jump."
        ,"Three floors pass before my glove makes contact with the metal wall and slows me down."
        ,"At the nearest balcony, I catch the railing and wrap my leg over to pull myself to my back on the rough cement floor, the comfortable feel of which the upper floors lack."
        ,"With a smile on my face, I attempt in vain to hold in my laughter."
        ,"I am invincible."
        ,"With my arm in a sling and my right hand still red and sore from the heat of the glove in my back pocket, I curse my terrible luck on my way out of the General Unit Security office for landing in sight of a security patrol; although with the instatement of the new Commissioner of the colony, and by extension the appointment of the Commissioner's chosen Chief of Security, they were too busy to deal with a reckless student, so in reality I am rather fortunate."
        ,"I was even able to conceal my glove well enough to prevent it from being confiscated again."
        ,"Up the outer stairway to the Academy, I pause by the railing to breathe in the salty ocean air and look at the morning sun over the rolling blue expanse."
        ,"This scene alone is enough for me to envy those who live on the lower floors of the spire."

    ]



    print(bot.check_scene(transition_clear))
    print(bot.check_scene(transition_moderate))
    print(bot.check_scene(transition_subtle))
    print(bot.check_scene(non_transition))
    print(bot.check_scene(falling))

    print(f"API Calls: {bot.API_calls}")
    print(f"Prompt Tokens Total: {bot.prompt_tokens_total}")
    print(f"Response Tokens Total: {bot.response_tokens_total}")
