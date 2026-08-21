
import os
import re
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(
    title="API Corporativa Recebimentos Bahia - O Boticário",
    version="1.0.0"
)

# Permitir CORS para o front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar cliente oficial do Gemini
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "SUA_CHAVE_GEMINI_AQUI")
client = genai.Client(api_key=GEMINI_API_KEY)

# Validação do Domínio Corporativo Boticário
ALLOWED_DOMAIN = "grupoboticario.com.br"

def validar_usuario_corporativo(x_user_email: Optional[str] = Header(None)):
    if not x_user_email or not x_user_email.endswith(f"@{ALLOWED_DOMAIN}"):
        raise HTTPException(
            status_code=403, 
            detail="Acesso negado. Apenas funcionários com e-mail @grupoboticario.com.br têm acesso."
        )
    return x_user_email

class ChatRequest(BaseModel):
    pergunta: str
    loja_selecionada: Optional[str] = "HUB"

@app.get("/")
def read_root():
    return {"status": "online", "sistema": "Recebimentos Bahia - O Boticário"}

@app.post("/api/v1/analisar-nota-ocr")
async def analisar_nota_ocr(
    file: UploadFile = File(...),
    user_email: str = Depends(validar_usuario_corporativo)
):
    """
    Recebe um arquivo (PDF, JPG, PNG, XML) e utiliza o Gemini para realizar o OCR
    e extração inteligente de dados contábeis da Nota Fiscal.
    """
    try:
        content = await file.read()
        
        prompt = """
        Você é o auditor contábil corporativo do Grupo Boticário responsável pelo Estado da Bahia.
        Analise esta nota fiscal e extraia os dados estritamente em formato JSON com as chaves:
        - numero_nota: (String, ex: "000123456/1")
        - cnpj_emitente: (String, ex: "02.096.748/0002-46")
        - nome_fornecedor: (String)
        - valor_total: (Number)
        - data_emissao: (String, formato YYYY-MM-DD)
        - loja_destino: (String: uma de ["SIMOES_FILHO", "PITUBA", "CAJAZEIRAS", "CALCADA", "LAURO_FREITAS"])
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=content,
                    mime_type=file.content_type,
                ),
                prompt
            ]
        )
        return {"sucesso": True, "resposta_gemini": response.text, "operador": user_email}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat-gemini")
async def chat_gemini(
    request: ChatRequest,
    user_email: str = Depends(validar_usuario_corporativo)
):
    """
    Atende dúvidas operacionais e de SLA exclusivamente com o modelo Gemini.
    """
    prompt_sistema = f"""
    Você é o assistente virtual exclusivo de operações do Grupo Boticário para a regional Bahia.
    Lojas sob escopo: SIMÕES FILHO (V084), PITUBA (V078), CAJAZEIRAS (V070), CALÇADA (V074), LAURO DE FREITAS (V072).
    Aba ativa pelo usuário: {request.loja_selecionada}.
    
    Responda objetivamente à dúvida do colaborador: {request.pergunta}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt_sistema
    )
    return {"resposta": response.text}
