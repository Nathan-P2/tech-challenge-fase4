# Monitoramento Multimodal de Pacientes

Tech Challenge da Fase 4 da Pós Tech IA para Devs (FIAP). O sistema acompanha pacientes internados com dados de **vídeo, áudio e texto** e com séries de **sinais vitais, prescrições e mobilidade**. Ele detecta sinais precoces de risco, junta as evidências das várias modalidades e envia alertas à equipe médica com a conduta recomendada.

O relatório técnico está em [`docs/Relatorio_Tecnico_Fase4.pdf`](docs/Relatorio_Tecnico_Fase4.pdf).

## Fluxo multimodal

```
 Vídeo (fisioterapia/cirurgia) ─► YOLOv8-pose / OpenPose ─► ângulos, tremor, equilíbrio, queda, zonas ─┐
 Áudio (consultas) ─► Azure Speech to Text ─► Azure Text Analytics (Health) ─► termos críticos/sentimento ┤
                  └► biomarcadores vocais (librosa) ─► fadiga, dif. respiratória, disartria ─────────────┤
 Sinais vitais (1/min) ─► NEWS2 + z-score robusto + CUSUM sazonal + Isolation Forest (em fluxo) ────────┼─► Alertas ─► Fusão multimodal ─► Equipe
 Prescrições ─► regras farmacológicas (dose máxima, erro de vírgula, duplicidade, interação, escalonamento) ┤   (dedup.)   (regras clínicas +    (console, JSONL,
 Mobilidade ─► queda, imobilidade, agitação noturna, declínio ───────────────────────────────────────────┘                escore de risco)     webhook, painel)
```

| Modalidade | Modelos e técnicas | Saída |
|---|---|---|
| Vídeo | YOLOv8n-pose (equivalente ao OpenPose, mesmo esqueleto COCO; também há backend OpenPose via OpenCV DNN), YOLOv8n para objetos, subtração de fundo, análise espectral (FFT) | relatório automático de desvios e vídeo anotado |
| Áudio | Azure Speech to Text (fallback: Whisper local), pYIN (F0), jitter, shimmer, HNR, núcleos silábicos e pausas | indicadores vocais explicáveis em relação à linha de base do paciente |
| Texto | Azure Text Analytics for Health e análise de sentimento (fallback: léxico clínico pt-BR com negação) | termos críticos e sentimento |
| Sinais vitais | NEWS2, z-score com mediana/MAD, CUSUM com referência de 24 h, Isolation Forest, filtro de artefato | alertas em tempo real |
| Prescrições | base farmacológica e z-score de log-dose | erros de dose, duplicidades, interações e escalonamentos |
| Mobilidade | regras com linha de base do próprio paciente | queda, imobilidade, delirium e declínio |
| Fusão | regras de correlação clínica e escore noisy-OR com decaimento temporal | alerta fundido com conduta e risco por paciente |

## Instalação

Requer Python 3.11 a 3.13 e `ffmpeg`. Os dados de áudio sintéticos usam o comando `say` do macOS, mas as gravações já geradas estão em `data/`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[whisper,dashboard,dev]'
cp .env.example .env   # opcional: preencha as chaves do Azure
```

## Executar

Pipeline completo: gera os dados, analisa todas as modalidades, faz a fusão e produz o painel e os relatórios. Leva cerca de 8 minutos num Apple M4.

```bash
hospital-monitor run --transcriber whisper --physionet
```

Opções úteis: `--rapido` pula a comparação de métodos de sinais vitais, `--pose-backend gt` dispensa o YOLO e `--transcriber referencia` dispensa a transcrição.

Painel da equipe médica, com relógio do plantão, feed de alertas, paciente, vídeo, áudio, métricas e upload de novos arquivos:

```bash
streamlit run app/dashboard.py
```

Feed de alertas em tempo real, com 72 h reproduzidas em cerca de 1 minuto:

```bash
hospital-monitor stream --speed 3600 --min alta
```

Analisar arquivos avulsos:

```bash
hospital-monitor audio data/audio/P-101_fadiga_respiratoria.wav --baseline data/audio/P-101_baseline.wav
hospital-monitor video data/video/sessao_fisioterapia.mp4 --referencia data/video/sala_vazia.png
```

Regerar o relatório técnico a partir de `output/summary.json`. O HTML vira PDF pelo Chrome headless:

```bash
python scripts/build_report.py
```

Testes, sem rede e sem modelos pesados:

```bash
pytest -q
```

## Integração com Azure

| Serviço | Uso | Variáveis |
|---|---|---|
| Azure AI Speech, Speech to Text | Transcrição contínua em pt-BR com tempo por palavra | `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` |
| Azure AI Language, Text Analytics | Sentimento com opinion mining e frases-chave | `AZURE_LANGUAGE_KEY`, `AZURE_LANGUAGE_ENDPOINT` |
| Azure AI Language, Text Analytics for Health | Sintomas e medicamentos com negação (assertion) | as mesmas |
| Webhook (Teams ou Logic Apps) | Entrega de alertas de gravidade ALTA ou CRÍTICA | `ALERT_WEBHOOK_URL` |

Sem credenciais, cada serviço tem um substituto local com o mesmo formato de saída. Por isso o projeto roda inteiro offline, e o motor usado aparece em cada resultado.

## Saídas (`output/`)

- `summary.json`: todas as métricas;
- `alertas.jsonl`: trilha de auditoria dos alertas entregues;
- `passagem_de_plantao.md`: pacientes ordenados por risco, com alertas fundidos e condutas;
- `video/<sessão>/`: `video_report.md`, `video_report.json` e `video_anotado.mp4`;
- `audio/audio_report.json`: transcrições, biomarcadores, indicadores e avaliação;
- `figuras/*.png`: gráficos usados no relatório;
- `physionet/`: validação com dados reais (PhysioNet Challenge 2012).

## Dados

O projeto não usa dados reais de pacientes. Os dados sintéticos foram gerados com gabarito conhecido, para medir recall, falsos alarmes e latência:

- **Vídeo**: um paciente 2D renderizado faz abdução de ombros e agachamentos. Foram injetados assimetria, tremor de 6 Hz, desequilíbrio, queda e imobilidade. Há também uma sessão de controle sem anomalias.
- **Áudio**: 4 vozes pt-BR do macOS × 4 condições (admissão, controle, fadiga com dispneia e disartria). As alterações acústicas foram produzidas por processamento de sinal.
- **Sinais vitais, mobilidade e prescrições**: cenário com 6 pacientes por 72 h e uma coorte de avaliação com 30 pacientes, com eventos e artefatos aleatórios.
- **PhysioNet Challenge 2012** (dados reais de UTI, licença ODC-BY): validação do sinal prognóstico dos detectores em 1000 pacientes.
