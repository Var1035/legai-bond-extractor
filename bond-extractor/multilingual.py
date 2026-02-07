from mistralai import Mistral

def normalize_query(query: str, client: Mistral) -> str:
    """
    Convert any User Input (Telugu/Hindi/Hinglish/Teluglish) to clear English.
    """
    try:
        sys_msg = "You are a language translator and normalizer. Your task is to convert the User's input into clear, professional English. If the input is already in English, return it exactly as is. Do not answer the question. Just translate/normalize it."
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": query}
            ]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Normalization Error: {e}")
        return query # Fallback to original

def translate_answer(answer: str, target_lang: str, client: Mistral) -> str:
    """
    Translate the English answer to the target language (Telugu/Hindi).
    """
    if target_lang not in ["te", "hi"]:
        return answer
    
    lang_map = {"te": "Telugu", "hi": "Hindi"}
    lang_name = lang_map.get(target_lang, "English")
    
    try:
        sys_msg = f"""You are an expert legal translator. Translate the following English legal answer into PURE {lang_name}.
Rules:
1. Maintain the original formatting (Markdown, bold, italics, bullets).
2. Do not use English words in the output script (use {lang_name} script only).
3. Keep legal terms accurate.
4. Do not add any introductory text like 'Here is the translation'. Just the translated content."""
        
        resp = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": answer}
            ]
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"Translation Error: {e}")
        return answer # Fallback to English
