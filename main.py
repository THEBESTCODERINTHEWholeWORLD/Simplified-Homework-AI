    
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import cv2
from PIL import Image
import easyocr

#templates for AI
confidence_template = """
You are a strict classifier of a student's understanding.

conversation history:
{context}

latest student answer:
{question}

Decide the SINGLE best label:

- Respond with EXACTLY "YES" if:
The student clearly understands the topic and has consistently answered correctly.

- Respond with EXACTLY "CORRECT" if:
The student's latest answer to the last question is correct, but you are not sure they fully mastered the topic yet.

- Respond with EXACTLY "NO" if:
The student's latest answer is incorrect OR shows confusion.

Respond with ONE WORD ONLY: YES, CORRECT, or NO.
"""


classifier_template = """
You are a strict classifier.

Decide whether the following text is likely an actual homework problem
or assignment description from school (math, science, history, etc.).

Text:
{text}

If it IS likely homework, respond with EXACTLY:
HOMEWORK

If it is NOT likely homework (small talk, instructions to the bot, random Qs, etc.),
respond with EXACTLY:
NOT_HOMEWORK

Respond with ONE WORD ONLY.
"""

rearrange_question_template = """
You are a strict homework rearranger

Rearrange the problem if it seems like it isn't worded correctly, and it looks like a homework.

problem:
{problem}

When you are rearranging, keep the core meaning of the problem exactly the same as it was.
Respond with the rearranged problem.
"""

tutor_template = """
You are a patient AI tutor helping a student with their homework.

Conversation so far:
{context}

The student's latest message:
{question}

Your current task mode is: {mode}

If mode == "get_topic":
- Infer what SUBJECT (e.g., arithmetic, algebra, physics) and specific TOPIC or UNIT (e.g., solving linear equations, subtraction with small numbers) the homework problem is about.
- In 1–2 sentences, say something like: "This looks like [subject], specifically [topic]."
- In 1 sentence, politely ask the student to confirm if that subject/topic is correct.
- For now, do not give any information on how to solve the problem.
- Do NOT output any of these words by themselves: HOMEWORK, NOT_HOMEWORK, classify_homework, get_topic, explain, ask_question, feedback, ask_additional_questions.

If mode == "explain":
- Briefly explain ONE key concept they need next for THIS problem.
- For example, if they need a formula, give that formula and explain what the symbols mean.
- Use 3–6 sentences.
- Do NOT ask any questions in this mode.
- Do NOT include a question mark ("?") anywhere in your response.
- Based on the recent chat context, avoid repeating nearly identical sentences.

If mode == "ask_question":
- Ask ONE short checkpoint question to test their understanding of the concept you just explained.
- The question should be answerable in a sentence or a small calculation.
- Do NOT include the answer.

If mode == "feedback":
- Assume the student's latest message is their answer to your last checkpoint question.
- In 1–3 sentences, say if they are correct or not and briefly correct any misunderstanding.
- In 1–2 more sentences, suggest what they should try next (for example: one more practice question, or moving to the next step of the original problem).

If mode == "ask_additional_questions":
- Ask ONE new practice question that is similar in difficulty and topic to the original homework problem.
- The user's input should be the answer to that question.
- Do NOT give the answer.
- Do NOT ask more than one question at a time.

Respond with ONLY what belongs to this mode, nothing else.
"""

#model of AI 
model = OllamaLLM(model="gemma3:4b")

#prompts
classifier_prompt = ChatPromptTemplate.from_template(classifier_template)
classifier_chain = classifier_prompt | model

tutor_prompt = ChatPromptTemplate.from_template(tutor_template)
tutor_chain = tutor_prompt | model

prompt = ChatPromptTemplate.from_template(confidence_template)
confidence_chain = prompt | model

rearrange_prompt = ChatPromptTemplate.from_template(rearrange_question_template)
rearrange_chain = rearrange_prompt | model

#take a photo
def take_photo():
    def capture_when_ready(filename="photo.jpg"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Press SPACE to capture an image, or ESC to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            cv2.imshow("Press SPACE to capture", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Exiting without saving.")
                cap.release()
                cv2.destroyAllWindows()
                return False
            elif key == 32:  # SPACE key
                cv2.imwrite(filename, frame)
                print(f"✅ Image saved as {filename}")
                break

        cap.release()
        cv2.destroyAllWindows()
        return True

    def extract_text_from_image(image_path):
        """
        Extracts text from an image using pytesseract.
        """
        reader = easyocr.Reader(['en'])
        try:
            # Open the image using Pillow
            img = Image.open(image_path)

            # Use pytesseract to extract text
            text = reader.readtext(img, detail =0)
            return text
        
        except FileNotFoundError:
            return f"Image file not found at: {image_path}"
        except Exception as e:
            return f"An error occurred: {e}"

    filename = 'download.jpeg'
    if capture_when_ready(filename):
        extracted_text = extract_text_from_image(filename)
        return extracted_text
    
    else:
        return "Capture Cancelled"

#functions
def check_first_input(): #make sure the first input is the homework- add a parameter
    while True:
        sigma_text = take_photo() #random variable name- sorry
        user_text = " ".join(sigma_text)
        
        AI_text = rearrange_chain.invoke({
            "problem": user_text
        })

        AI_problem = getattr(AI_text, "content", str(AI_text))
        print(AI_problem)

        result = classifier_chain.invoke({
            "text": AI_problem,
        })

        label = getattr(result, "content", str(result)).strip().upper()

        if label == "HOMEWORK":
            print("\nAI: That looks like a homework problem. I'll help you with this one.\n")
            return AI_problem
        else:
            print("\nAI: That doesn't look like a homework problem or assignment description.")
            print("Please paste the exact text of the question from your homework.\n")

def understand_topic(context, lastanswer: str) -> str: 
    result = confidence_chain.invoke({
        "context": context,
        "question": lastanswer
    })
    label = getattr(result, "content", str(result)).strip().upper()
    return label


def handleconversation(): 
    #additional questions correct counter
    questions_correct = 0
    #additional questions wrong in a row
    questions_wrong = 0

    context = ""
    mode = "get_topic"

    print("Welcome to Simplified Homework AI! Type 'exit' to quit.")

    homework = check_first_input()
    context += f"\nHomework problem: {homework}"

    mode = "get_topic"

    result = tutor_chain.invoke({
        "context": context,
        "question": homework,
        "mode": mode
    })

    topic_text = getattr(result, "content", str(result))
    print("\nBot (getting topic):\n", topic_text, "\n")

    mode = "explain"
    print("Now you can respond and ask questions about this problem. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        elif mode == "explain": #explain how to solve the problem
            result = tutor_chain.invoke({
                "context": context,
                "question": user_input,
                "mode": "explain"
            })
            text = getattr(result, "content", str(result))

            print("\nBot (explanation):\n", text, "\n")

            context += f"\nUser: {user_input}\nAI: {text}"
            mode = "ask_question"  

        elif mode == "ask_question": #ask question based on their understanding
            result = tutor_chain.invoke({
                "context": context,
                "question": user_input,
                "mode": "ask_question"
            })
            text = getattr(result, "content", str(result))

            print("\nBot (checkpoint question):\n", text, "\n")

            context += f"\nUser: {user_input}\nAI: {text}"
            mode = "feedback"
        
        elif mode == "feedback": #assume the latest message is an answer attempt
            result = tutor_chain.invoke({
                "context": context,
                "question": user_input,
                "mode": "feedback"
            })
            text = getattr(result, "content", str(result))

            print("\nBot (feedback):\n", text, "\n")

            context += f"\nUser: {user_input}\nAI: {text}"
            
            label = understand_topic(context, user_input)

            if label == "YES":
                # Declare mastery
                print("\nBot: It looks like you really understand this topic! 🎉\n")
                break

            elif label == "CORRECT":
                questions_correct += 1
                questions_wrong = 0
                if questions_correct >= 2:
                    print("\nBot: You’ve answered several questions correctly. You’ve mastered this concept! 🎓\n")
                    break
                else:
                    # Ask another similar practice question
                    mode = "ask_additional_questions"

            else:  # NO
                questions_wrong += 1
                questions_correct = 0
                if questions_wrong >= 2:
                    print("\nBot: It seems like this is still tricky. Let’s review the concept again.\n")
                    mode = "explain"
                else:
                    # Give another explanation and then ask again
                    mode = "explain"

        elif mode == "ask_additional_questions":
            result = tutor_chain.invoke({
               "context": context,
                "question": user_input,
                "mode": "ask_additional_questions" 
            })
            text = getattr(result, "content", str(result))
            print("\nBot (ask_additional_question):\n", text, "\n")
            context += f"\nUser: {user_input}\nAI: {text}"
            
            mode = "feedback"
            
if __name__ == "__main__":
    handleconversation()