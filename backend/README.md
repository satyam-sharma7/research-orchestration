# Backend Developer Guide

Welcome to the backend of the Research Orchestration platform!

If you are new to standard backend development, this document will serve as your tour guide. We use **FastAPI**, a modern and fast Python web framework. Recently, this backend was completely refactored using **Domain-Driven Design (DDD)**. 

Don't let the jargon intimidate you! "Domain-Driven Design" just means that instead of putting all our code in one giant file, we organize our folders based on *what the code actually does*. 

---

## 📂 The File Structure (and why it's built this way)

Here is a map of the `backend/` directory:

```text
backend/
├── main.py                  <-- The "Front Door"
├── api/                     <-- The "Receptionists"
│   ├── auth_routes.py       
│   └── chat_routes.py       
├── core/                    <-- The "Rulebook & Toolbelt"
│   ├── config.py            
│   ├── deps.py              
│   └── security.py          
├── db/                      <-- The "Filing Cabinets"
│   ├── database.py          
│   └── models.py            
├── memory/                  <-- The "AI Brain Storage"
│   ├── episodic.py
│   ├── long_term.py         
│   └── short_term.py        
└── schemas/                 <-- The "Bouncers"
    ├── auth.py
    └── chat.py
```

Let's break down what each of these folders does in plain English.

### 1. `main.py` (The Front Door)
This is the entry point of the app. It's incredibly small on purpose. Its only job is to create the FastAPI `app` object, set up some basic security (CORS), and connect the different "Receptionists" (`api/` routes) to the app. 
- You start the whole server by running: `fastapi dev backend/main.py`.

### 2. `api/` (The Receptionists)
When you type a URL in your browser or frontend (like `http://localhost:8000/api/auth/login`), this folder determines what Python function handles it. We split our routes so the file doesn't get thousands of lines long.
- `auth_routes.py`: Handles creating accounts and logging in.
- `chat_routes.py`: Handles everything related to talking to the AI.

### 3. `schemas/` (The Bouncers)
Notice that passing data via the internet is messy. We use **Pydantic** in the `schemas/` folder to create strict rules for what data is allowed *in* and *out*.
- When a user tries to log in, `schemas/auth.py` checks that they definitely provided an `email` (string) and a `password` (string). If they didn't, FastAPI automatically bounces them back an error. We also use schemas to format the data going *out* (like stripping out the user password before sending a profile to the frontend).

### 4. `core/` (The Rulebook & Toolbelt)
This folder holds logic that applies to the *entire* application:
- `config.py`: Reads your hidden `.env` file just once and holds those values (API keys, URLs). If you see `os.getenv` a million times in an app, it slows things down. We do it here centrally.
- `security.py`: Logic for changing raw passwords into gibberish (hashing) and creating/reading JWT tokens (the digital wristbands that keep users logged in).
- `deps.py`: Stands for "Dependencies." FastApi uses these heavily! For example, `get_current_user_id` takes a user's web request, looks for their JWT Token, and extracts who they are before letting them access their chats.

### 5. `db/` (The Filing Cabinets)
This connects to your **PostgreSQL** database (via `SQLAlchemy`). 
- `models.py`: While `schemas/` defines what data looks like over the *internet*, `models.py` defines what the tables look like literally stored on the *hard drive* in the SQL database. (e.g., the `users` table, the `sessions` table).

### 6. `memory/` (The AI Brain Storage)
This handles where the AI keeps both the recent conversation and lifelong facts it learns about a user.
- `short_term.py`: Connects to **Redis**. This is like the AI's short-term working memory. It stores the last 20 messages of a chat session so the API is extremely fast. Redis is stored in RAM, so it's lightning quick but disappears if turned off.
- `long_term.py`: Connects to **Pinecone** (a Vector Database). This is the AI's *Semantic Memory*. If a user says "I am allergic to peanuts", we don't want the AI to forget that just because they started a new chat session. We store that fact here forever.
- `episodic.py`: Connects to **Pinecone**. This is the AI's *Episodic Memory*. While Semantic Memory remembers *facts*, Episodic Memory remembers *events*. For example: "Yesterday, the AI successfully debugged the auth system and fixed a security bug." It records the agent's actions and outcomes so it doesn't repeat past mistakes.

---

## 🌊 The Flow: What happens when a user sends a message?

Let's trace exactly how these files work together when a frontend user types "Hello AI" and hits send.

**1. The Request Arrives**
- The frontend shoots a POST request to `/api/chat/sessions/123/messages`. 
- `main.py` catches it and hands it to `api/chat_routes.py`.

**2. The Bouncers Check the Data**
- FastAPI checks `schemas/chat.py` to ensure the user actually provided a message and a session ID. 

**3. The Toolbelt verifies the User**
- Before running the route, FastAPI runs a Dependency (`Depends(get_current_user_id)`) from `core/deps.py`. This reads the user's hidden JWT Token and mathematically proves they are logged in using rules from `core/security.py`.

**4. The Filing Cabinet verifies Ownership**
- Inside `api/chat_routes.py`, it makes a quick call to Postgres via `db/database.py` and `db/models.py`. It asks: "Does chat session 123 actually belong to this specific user?" If not, it kicks them out. 

**5. Fetching the AI's Memories**
- The backend asks `memory/short_term.py` for the last 20 messages of this chat from Redis so the AI understands context. (If Redis is empty, it quietly asks Postgres for the history, saves it in Redis for next time, and returns it).

**6. Calling the LLM (Large Language Model)**
- In `api/chat_routes.py`, the backend loops through standard asynchronous HTTP requests calling an external LLM server (like OpenAI).
- **Because it runs via `StreamingResponse`:** It doesn't wait for the whole answer to finish generating. As the LLM spits out one word at a time, FastAPI immediately streams that single word down to the frontend! This is how you get that cool "typing" effect on modern AI apps.

**7. Storing the New Memories**
- Once the AI finishes streaming its answer, the backend turns around and silently saves the exact conversation transcript back into Postgres (`db/models.py`) and bumps the Redis cache (`memory/short_term.py`). 

Done! And because this architecture is entirely asynchronous (`async def`, `await`, `AsyncSession`), your single server can handle hundreds of users chatting at the exact same time without them waiting in line behind each other. 

---

## 💡 How do Semantic and Episodic Memory get used?

In `memory/long_term.py` and `memory/episodic.py`, we created "tool factories": `create_memory_tools(user_id)` and `create_episodic_tools(user_id)`.

When you are ready to build a complex LangGraph or LangChain Agent, you will call those factories. They will generate actual Python functions (Save Fact, Search Past Events, etc.) completely boxed up for an AI to access. 

The AI will read the descriptions we mapped out. When you say "Remember that my dog is named Pluto", the AI will decide completely on its own to pause the UI, call the "Save Fact" function to push the vector to Pinecone, and then respond to you. When you say "Have we ever run into this Redis bug before?", it will call "Search Past Events" to read its episodic history before answering.

Welcome to modern backend AI development!