import config

def _call_llm(prompt: str) -> str:
    provider  = config.ACTIVE_PROVIDER
    model_cfg = config.MODEL_OPTIONS[provider]

    if provider == "gemini":
        import google.generativeai as genai
        genai.configure(api_key=model_cfg["api_key"])
        return genai.GenerativeModel(model_cfg["chat_model"]).generate_content(prompt).text.strip()

    elif provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=model_cfg["api_key"])
        msg = client.messages.create(
            model=model_cfg["chat_model"], max_tokens=256,
            messages=[{"role": "user", "content": prompt}])
        return msg.content[0].text.strip()

    elif provider == "llama":
        from groq import Groq
        client = Groq(api_key=model_cfg["api_key"])
        resp = client.chat.completions.create(
            model=model_cfg["chat_model"], max_tokens=256,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=model_cfg["api_key"])
        resp = client.chat.completions.create(
            model=model_cfg["chat_model"], max_tokens=256,
            messages=[{"role": "user", "content": prompt}])
        return resp.choices[0].message.content.strip()

    return ""

def expand_topic(user_topic: str) -> list:
    prompt = (f"Give 5 academic search query variants for: {user_topic}\n"
              f"Return only 5 short queries, one per line.")
    try:
        text  = _call_llm(prompt)
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        return list(dict.fromkeys([user_topic] + lines[:5]))
    except Exception as e:
        print(f"  [topic_expander] failed: {e}")
        return [user_topic]
