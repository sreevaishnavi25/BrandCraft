#importing the tools we need
import io    #helps us handle data like images in memory instead of saving to a file.
import uvicorn  ## The server 'engine' that runs our app so people can visit it.
from fastapi import FastAPI, HTTPException  # FastAPI is the skeleton of our web app; HTTPException handles errors.
from fastapi.responses import StreamingResponse  # Used to send an actual image back to a user's browser.
from fastapi.middleware.cors import CORSMiddleware  # Allows our website (frontend) to talk to this code (backend).
from pydantic import BaseModel  # A tool to define what a 'packet' of data from a user should look like.
from groq import Groq  # The tool to talk to the "Groq" AI (for text generation).
from PIL import Image, ImageDraw, ImageFont  # A library for creating and editing images (adding watermarks).
from huggingface_hub import InferenceClient  # The tool to talk to "Hugging Face" (for logo making and classifying text).

# 🔑 API KEYS ALSO CALLED AS SECRET KEYS
## These are like passwords. They tell the AI services who we are and that we have permission to use them.
GROQ_API_KEY = "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"
HF_TOKEN = "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"

app = FastAPI()

# Allow the frontend to talk to the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 #Carries our requests to AI companies.
groq_client = Groq(api_key=GROQ_API_KEY)
hf_client = InferenceClient(token=HF_TOKEN)

class QuizData(BaseModel):
    answers: str

# ANALYZES PERSONALITY
@app.post("/quiz-personality")
async def analyze_personality(quiz: QuizData):
    try:
        #List of Categories
        labels = ["Luxury", "Eco-Friendly", "Tech-Forward", "Playful", "Minimalist"]
        
        #Used for categorizing the text.
        result = hf_client.zero_shot_classification(
            quiz.answers, 
            candidate_labels=labels, 
            model="facebook/bart-large-mnli"
        )
        return {"top_personality": result[0]['label']}
    except Exception:
        return {"top_personality": "Modern"}

# GENERATE LOGO
@app.get("/generate-logo")
async def generate_logo(brand_name: str):
    prompt = f"Professional minimalist vector logo for {brand_name}, white background, high quality, centered"
    try:
        # Generate image
        image = hf_client.text_to_image(prompt, model="stabilityai/stable-diffusion-xl-base-1.0")
        
        # Process image 
        img = image.convert("RGBA")
        width, height = img.size
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Simple text watermark
        draw.text((width - 160, height - 40), "BrandCraft AI", fill=(150, 150, 150, 180))
        
        #It is used to combine the logo and the watermark
        final_img = Image.alpha_composite(img, overlay).convert("RGB")

        #coverts it into bytes
        img_io = io.BytesIO()
        final_img.save(img_io, 'PNG')
        img_io.seek(0)

        #shown to the user
        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

#Generate Brand Text
@app.get("/generate-brand")
def generate_brand(description: str, personality: str = "Modern", type: str = "name"):
    if type == "name":
        prompt = f"Create 3 catchy names and slogans for a {personality} brand: {description}. Be brief."
    else:
        prompt = f"Write a premium product description for a {personality} brand specializing in: {description}."
        
    chat = groq_client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return {"result": chat.choices[0].message.content}

#AI Chatbot
@app.get("/branding-chat")
async def branding_chat(message: str):
    try:
        # 1. We call the Groq AI
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are MuseAI, a professional brand consultant."},
                {"role": "user", "content": message}
            ],
            model="llama-3.3-70b-versatile",
        )
        
        # GET the AI"s response text.
        ai_response = chat.choices[0].message.content
        
        # 3. CRITICAL: This MUST be 'response' to match your index.html
        return {"response": ai_response}
        
    except Exception as e:
        print(f"Chat Error: {e}")
        return {"response": "I'm having trouble connecting to my brain right now."}

#Multilingual 
@app.get("/localize")
async def localize(text: str, language: str):
    system_prompt = (
        f"You are a professional translation engine.\n"
        f"Translate the given text strictly into {language}.\n"
        f"Output ONLY the translated text.\n"
        f"Do not add explanations, examples, or extra words."
    )

    try:
        chat = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            max_tokens=500
        )

        return {"result": chat.choices[0].message.content.strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
#starting the engine
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)