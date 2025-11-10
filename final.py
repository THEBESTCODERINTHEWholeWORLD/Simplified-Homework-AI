import streamlit as st
from PIL import Image
import easyocr

from main import (
    classifier_chain,
    tutor_chain,
    rearrange_chain,
    understand_topic,
)
#main code, other code was the prototype
# Initialize EasyOCR once
reader = easyocr.Reader(["en"])

def extract_text_from_image_file(file_obj) -> str:
    img = Image.open(file_obj)
    text_list = reader.readtext(img, detail=0)
    return " ".join(text_list)


def classify_and_clean_homework(raw_text: str):
    # Rearrange the problem
    rearranged = rearrange_chain.invoke({"problem": raw_text})
    problem = getattr(rearranged, "content", str(rearranged))

    # Classify as HOMEWORK / NOT_HOMEWORK
    result = classifier_chain.invoke({"text": problem})
    label = getattr(result, "content", str(result)).strip().upper()

    if label == "HOMEWORK":
        return problem
    return None


def start_session(homework_text: str):
    context = f"\nHomework problem: {homework_text}"
    mode = "get_topic"
    questions_correct = 0
    questions_wrong = 0

    # Get topic / subject
    result = tutor_chain.invoke({
        "context": context,
        "question": homework_text,
        "mode": mode
    })
    bot_msg = getattr(result, "content", str(result))

    # Next mode will be explain
    mode = "explain"

    # Save to Streamlit session state
    st.session_state.context = context
    st.session_state.mode = mode
    st.session_state.questions_correct = questions_correct
    st.session_state.questions_wrong = questions_wrong
    st.session_state.homework = homework_text

    return bot_msg


def next_turn(user_input: str) -> str:
    context = st.session_state.context
    mode = st.session_state.mode
    qc = st.session_state.questions_correct
    qw = st.session_state.questions_wrong

    if mode == "explain":
        result = tutor_chain.invoke({
            "context": context,
            "question": user_input,
            "mode": "explain"
        })
        text = getattr(result, "content", str(result))
        context += f"\nUser: {user_input}\nAI: {text}"
        mode = "ask_question"

    elif mode == "ask_question":
        result = tutor_chain.invoke({
            "context": context,
            "question": user_input,
            "mode": "ask_question"
        })
        text = getattr(result, "content", str(result))
        context += f"\nUser: {user_input}\nAI: {text}"
        mode = "feedback"

    elif mode == "feedback":
        result = tutor_chain.invoke({
            "context": context,
            "question": user_input,
            "mode": "feedback"
        })
        text = getattr(result, "content", str(result))
        context += f"\nUser: {user_input}\nAI: {text}"

        label = understand_topic(context, user_input)

        if label == "YES":
            text += "\n\nIt looks like you really understand this topic!"
            mode = "done"

        elif label == "CORRECT":
            qc += 1
            qw = 0
            if qc >= 2:
                text += "\n\nYou’ve answered several questions correctly. You’ve mastered this concept! 🎓"
                mode = "done"
            else:
                mode = "ask_additional_questions"

        else:  # "NO"
            qw += 1
            qc = 0
            if qw >= 2:
                text += "\n\nIt seems like this is still tricky. Let’s review the concept again."
                mode = "explain"
            else:
                mode = "explain"

    elif mode == "ask_additional_questions":
        result = tutor_chain.invoke({
            "context": context,
            "question": user_input,
            "mode": "ask_additional_questions"
        })
        text = getattr(result, "content", str(result))
        context += f"\nUser: {user_input}\nAI: {text}"
        mode = "feedback"

    else:
        text = "This session is finished. Please start a new one."
        mode = "done"

    # Update state
    st.session_state.context = context
    st.session_state.mode = mode
    st.session_state.questions_correct = qc
    st.session_state.questions_wrong = qw

    return text

st.title("Simplified Homework AI")

# Initialize chat history if not present
if "chat" not in st.session_state:
    st.session_state.chat = []
if "mode" not in st.session_state:
    st.session_state.mode = None

st.header("1. Provide your homework problem")

# Option A: Paste text
st.subheader("A. Paste your homework text")

problem_text = st.text_area(
    "Paste the exact homework question here:",
    placeholder="e.g., A ball is thrown upward with initial velocity...",
)

if st.button("Start Tutor from Text"):
    if not problem_text.strip():
        st.warning("Please paste a homework problem first.")
    else:
        homework = classify_and_clean_homework(problem_text)
        if homework is None:
            st.error("This doesn't look like a homework problem. Try again with a real assignment.")
        else:
            first_bot = start_session(homework)
            st.session_state.chat = []
            st.session_state.chat.append(("bot", "Homework detected: " + homework))
            st.session_state.chat.append(("bot", first_bot))

# Option B: Upload photo
st.subheader("B. Upload a photo of your homework")

uploaded_image = st.file_uploader(
    "Choose an image file",
    type=["png", "jpg", "jpeg"],
    help="Take a picture with your phone or laptop and upload it here.",
)

if st.button("Start Tutor from Uploaded Photo"):
    if uploaded_image is None:
        st.warning("Please upload an image first.")
    else:
        raw_text = extract_text_from_image_file(uploaded_image)
        homework = classify_and_clean_homework(raw_text)
        if homework is None:
            st.error("This doesn't look like a homework problem. Try a clearer image.")
        else:
            first_bot = start_session(homework)
            st.session_state.chat = []
            st.session_state.chat.append(("bot", "Homework detected from photo: " + homework))
            st.session_state.chat.append(("bot", first_bot))

# Option C: Use webcam (Streamlit camera)
st.subheader("C. Take a picture with your webcam")

camera_image = st.camera_input("Take a homework photo with your webcam")

if st.button("Start Tutor from Camera Photo"):
    if camera_image is None:
        st.warning("Please capture a photo first (use the 'Take Photo' button in the camera box).")
    else:
        raw_text = extract_text_from_image_file(camera_image)
        homework = classify_and_clean_homework(raw_text)
        if homework is None:
            st.error("This doesn't look like a homework problem. Try taking a clearer picture.")
        else:
            first_bot = start_session(homework)
            st.session_state.chat = []
            st.session_state.chat.append(("bot", "Homework detected from camera: " + homework))
            st.session_state.chat.append(("bot", first_bot))

st.header("2. Chat with the tutor")

# Show chat history
for speaker, text in st.session_state.chat:
    if speaker == "user":
        st.markdown(f"**You:** {text}")
    else:
        st.markdown(f"**Bot:** {text}")


def send_message():
    user_msg = st.session_state.user_input

    if st.session_state.mode is None:
        st.session_state.chat.append(
            ("bot", " Start the tutor first using text, upload, or camera above.")
        )
        return

    if not user_msg.strip():
        return

    # Add user message to chat
    st.session_state.chat.append(("user", user_msg))

    # Get bot reply using your existing state machine
    bot_reply = next_turn(user_msg)
    st.session_state.chat.append(("bot", bot_reply))

    # Clear input field (allowed inside callback)
    st.session_state.user_input = ""


# Text input bound to session state
st.text_input("Your message:", key="user_input")

# Button that triggers the callback
st.button("Send Message", on_click=send_message)
