"""
AI Question Generator (Groq Powered)
=====================================
Responsibility: Accept clean text, call Llama 3/4 via Groq,
parse and validate the response.

It knows NOTHING about files, URLs, or extraction.
That is the job of text_extractor.py.
"""

import os
import re
import json
import time
from dotenv import load_dotenv
load_dotenv()

# Detect cloud environment
IS_RENDER = os.environ.get("RENDER", "false").lower() == "true"

class AIQuestionGenerator:

    def __init__(self, api_key: str = None):
        self.groq_key = os.environ.get("GROQ_API_KEY") or api_key
        if not self.groq_key:
            print("[AIGen] CRITICAL: GROQ_API_KEY not found in environment!")
        else:
            masked_key = self.groq_key[:6] + "..." + self.groq_key[-4:]
            print(f"[AIGen] Initialized with key: {masked_key}")

    def generate_questions(
        self,
        source: str,
        source_type: str,
        hardness: str = "medium",
        num_questions: int = 5,
        question_type: str = "objective",
    ) -> list:
        from app.utils.text_extractor import extract_text
        extracted = extract_text(source, source_type)

        text   = extracted.get("text", "")
        method = extracted.get("method", "unknown")
        frames = extracted.get("frames", [])

        print(f"[AIGen] Extraction complete. Method={method}, TextLen={len(text)}, Frames={len(frames)}")

        if not text and not frames:
            raise ValueError(f"Could not extract any context: {extracted.get('error', 'Unknown error')}")

        # ── STAGE 1: The Transformer Distiller (Learning Phase) ──
        print("[AIGen] Stage 1: Distilling knowledge...")
        knowledge_doc = self._distill_knowledge(text, frames, method)

        if not knowledge_doc or len(knowledge_doc) < 50:
            raise ValueError("The Transformer could not find enough educational content to learn from.")

        print(f"[AIGen] Stage 1 Complete: {len(knowledge_doc)} chars of clean knowledge.")

        # ── STAGE 2: The Examiner (Generation Phase) ──
        print(f"[AIGen] Stage 2: Generating {num_questions} {question_type} questions...")
        prompt = self._build_prompt(hardness, num_questions, question_type)
        return self._generate_from_knowledge(knowledge_doc, prompt, num_questions, question_type)

    def _distill_knowledge(self, text, frames, method):
        """Stage 1: Uses a Transformer to 'learn' the topic and scrub the noise."""
        from groq import Groq

        # Groq client with an explicit HTTP timeout to prevent silent hangs on Render
        client = Groq(
            api_key=self.groq_key,
            timeout=90.0,   # 90-second hard timeout per request
            max_retries=2,  # Auto-retry on transient API errors
        )

        if method == "video_frames" and frames:
            # Stage 1: Vision Learning (Llama 4 Natively Multimodal Models)
            vision_models = [
                "meta-llama/llama-4-scout-17b-16e-instruct",
                "meta-llama/llama-4-maverick-17b-128e-instruct",
            ]

            for v_model in vision_models:
                try:
                    print(f"[AIGen] Vision Learning via: {v_model}")
                    message_content = [{
                        "type": "text",
                        "text": "Observe these video frames and write a detailed TECHNICAL SUMMARY of the concepts shown. Include any code, slides, or diagrams. Output ONLY the technical text."
                    }]
                    for frame in frames[:5]:
                        message_content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame['inline_data']['data']}"}
                        })

                    completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": message_content}],
                        model=v_model,
                        temperature=0,
                        max_tokens=2000,
                    )
                    return completion.choices[0].message.content
                except Exception as ve:
                    print(f"[AIGen] Vision model {v_model} failed: {ve}")
                    continue

            raise ValueError("All Groq Vision / Llama 4 models are currently busy. Please try again in 60 seconds.")

        # ── Text Learning (Llama 3.3 Transformer) ──
        system_msg = "You are a Technical Knowledge Distiller. Your job is to READ the source text and EXTRACT a clean fact-sheet. "

        if method == "youtube_metadata_scrape":
            system_msg += (
                "NOTE: Only video metadata is available. "
                "Use the VIDEO TITLE to perform a 'Knowledge Synthesis': "
                "Write a professional 500-word technical overview of the computer science topic "
                "mentioned in the title. IGNORE all ads, links, or promotional content."
            )
        else:
            system_msg += (
                "DELETE all sponsors, mentorships, links, and speaker self-promotions. "
                "Output a structured 'Technical Fact Sheet' based on the educational content only."
            )

        completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user",   "content": f"SOURCE CONTENT:\n{text[:15000]}"}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=2000,
        )
        return completion.choices[0].message.content

    def _generate_from_knowledge(self, knowledge_doc, prompt, num, q_type):
        """Stage 2: Generate questions from the clean Knowledge Document."""
        from groq import Groq

        client = Groq(
            api_key=self.groq_key,
            timeout=90.0,
            max_retries=2,
        )

        # For large question counts, cap per-call at 20 to stay within token limits
        # and then make multiple calls if needed
        BATCH_SIZE = 20
        if num <= BATCH_SIZE:
            return self._single_generation(client, knowledge_doc, prompt, num, q_type)
        else:
            # Multi-batch generation for 21-50 questions
            all_questions = []
            remaining = num
            batch_num = 1
            while remaining > 0:
                batch = min(remaining, BATCH_SIZE)
                print(f"[AIGen] Batch {batch_num}: Generating {batch} questions (Difficulty: {hardness})...")
                batch_prompt = self._build_prompt(hardness, batch, q_type)
                questions = self._single_generation(client, knowledge_doc, batch_prompt, batch, q_type)
                all_questions.extend(questions)
                remaining -= batch
                batch_num += 1
                if remaining > 0:
                    time.sleep(1)  # Brief pause between batches to avoid rate limits
            return all_questions[:num]

    def _single_generation(self, client, knowledge_doc, prompt, num, q_type):
        """Single Groq API call for question generation."""
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a University Professor and Exam Paper Setter. "
                        "Use the provided KNOWLEDGE DOCUMENT to create academic questions. "
                        "STRICT RULE: Questions must be 100% academic. "
                        "Never mention people, starter kits, or ads."
                    )
                },
                {
                    "role": "user",
                    "content": f"KNOWLEDGE DOCUMENT:\n\"\"\"\n{knowledge_doc}\n\"\"\"\n\n{prompt}"
                }
            ],
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        raw_res = chat_completion.choices[0].message.content
        return self._parse_and_validate(raw_res, q_type)[:num]

    # ─────────────────────────────────────────────────────────────
    # Internal: JSON Parsing + Validation
    # ─────────────────────────────────────────────────────────────

    def _parse_and_validate(self, raw_text: str, question_type: str) -> list:
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as e:
            print(f"[AIGen] JSON parse error: {e}")
            return []

        if isinstance(data, list):
            raw_qs = data
        elif isinstance(data, dict):
            raw_qs = data.get("questions", [])
            if not raw_qs and "question_text" in data:
                raw_qs = [data]
        else:
            return []

        final = []
        for q in raw_qs:
            if not isinstance(q, dict):
                continue
            question_text = (q.get("question_text") or "").strip()
            if not question_text:
                continue

            normalised = {
                "question_text": question_text,
                "question_type": "multiple_choice" if question_type == "objective" else "subjective",
                "marks":         1,
                "model_answer":  q.get("model_answer", ""),
            }

            if question_type == "objective":
                options = q.get("options", [])
                if len(options) < 4:
                    continue
                ca = q.get("correct_answer")
                if isinstance(ca, int) and 0 <= ca < len(options):
                    ca = options[ca]
                if not ca or not isinstance(ca, str):
                    continue
                normalised["options"]        = options[:4]
                normalised["correct_answer"] = ca.strip()
            else:
                normalised["model_answer"] = q.get("model_answer", "")

            final.append(normalised)

        return final

    def _build_prompt(self, hardness: str, num: int, q_type: str) -> str:
        # Map difficulty to instructions
        diff_info = {
            "easy":   "Questions should be foundational, simple, and direct. Focus on key definitions and basic concepts.",
            "medium": "Questions should be balanced. Include some direct and some conceptual/application-based questions.",
            "hard":   "Questions should be advanced, complex, and high-difficulty. Focus on critical thinking, deep theory, and edge cases."
        }.get(hardness.lower(), "Questions should be balanced and academic.")

        if q_type == "objective":
            format_block = """
Each question must have:
  - "question_text": rigorous, specific question based ONLY on provided source
  - "options": exactly 4 distinct, meaningful choices
  - "correct_answer": EXACT text of one option
  - "model_answer": brief logical explanation based on the content
"""
        else:
            format_block = """
Each question must have:
  - "question_text": open-ended academic question requiring detail
  - "model_answer": a detailed ideal answer derived strictly from context
"""
        return f"""
You are a University Exam Paper Setter. 

DIFFICULTY LEVEL: {hardness.upper()}
INSTRUCTION: {diff_info}

OBJECTIVE: Generate exactly {num} questions based strictly on the Knowledge Document.

STRICT RULES:
1. ONLY use information from the source content.
2. If content is empty or insufficient, return: {{"questions": []}}
3. DO NOT use general knowledge. DO NOT hallucinate.
4. Accuracy is paramount.

JSON FORMAT (return ONLY this JSON, nothing else):
{{
  "questions": [
    {{
      "question_text": "...",
      "options": ["...", "...", "...", "..."],
      "correct_answer": "...",
      "model_answer": "..."
    }}
  ]
}}
{format_block}
"""
