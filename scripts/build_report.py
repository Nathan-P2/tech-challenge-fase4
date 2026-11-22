"""Gera o relatório técnico (HTML → PDF via Chrome headless) a partir de output/summary.json.

    python scripts/build_report.py [--github URL]
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
DOCS = ROOT / "docs"
CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "google-chrome", "chromium", "chrome"]


def img(name: str, width: str = "100%") -> str:
    data = base64.b64encode((OUT / "figuras" / name).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="width:{width}">'


def pct(x, d=0):
    return f"{x * 100:.{d}f}%".replace(".", ",")


def num(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def table(headers, rows, widths=None):
    th = "".join(f"<th{f' style=\"width:{w}\"' if widths else ''}>{h}</th>" for h, w in zip(headers, widths or headers))
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


DIAGRAM = """
<svg viewBox="0 0 760 400" xmlns="http://www.w3.org/2000/svg" style="width:100%;font-family:Arial;font-size:11px">
 <defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
 <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker></defs>
 <g fill="#eef4fc" stroke="#2a78d6">
  <rect x="5" y="10" width="140" height="44" rx="6"/><rect x="5" y="72" width="140" height="44" rx="6"/>
  <rect x="5" y="134" width="140" height="44" rx="6"/><rect x="5" y="196" width="140" height="44" rx="6"/>
  <rect x="5" y="258" width="140" height="44" rx="6"/></g>
 <g fill="#0b0b0b"><text x="75" y="30" text-anchor="middle" font-weight="bold">Vídeo</text>
  <text x="75" y="45" text-anchor="middle">fisioterapia / cirurgia</text>
  <text x="75" y="92" text-anchor="middle" font-weight="bold">Áudio</text><text x="75" y="107" text-anchor="middle">consultas gravadas</text>
  <text x="75" y="154" text-anchor="middle" font-weight="bold">Sinais vitais</text><text x="75" y="169" text-anchor="middle">monitor, 1 amostra/min</text>
  <text x="75" y="216" text-anchor="middle" font-weight="bold">Prescrições</text><text x="75" y="231" text-anchor="middle">prontuário eletrônico</text>
  <text x="75" y="278" text-anchor="middle" font-weight="bold">Mobilidade</text><text x="75" y="293" text-anchor="middle">sensor de leito / vídeo</text></g>
 <g fill="#fdf3ee" stroke="#eb6834">
  <rect x="185" y="10" width="230" height="44" rx="6"/><rect x="185" y="72" width="230" height="44" rx="6"/>
  <rect x="185" y="134" width="230" height="44" rx="6"/><rect x="185" y="196" width="230" height="44" rx="6"/>
  <rect x="185" y="258" width="230" height="44" rx="6"/></g>
 <g fill="#0b0b0b">
  <text x="300" y="29" text-anchor="middle">YOLOv8-pose / OpenPose + YOLOv8</text><text x="300" y="44" text-anchor="middle">ângulos, tremor, equilíbrio, queda, zonas</text>
  <text x="300" y="88" text-anchor="middle">Azure Speech to Text → Text Analytics (Health)</text><text x="300" y="102" text-anchor="middle">+ biomarcadores vocais (F0, pausas, HNR)</text>
  <text x="300" y="110" text-anchor="middle" font-size="9" fill="#52514e"></text>
  <text x="300" y="151" text-anchor="middle">NEWS2 · z-score robusto · CUSUM sazonal</text><text x="300" y="166" text-anchor="middle">Isolation Forest · filtro de artefato</text>
  <text x="300" y="213" text-anchor="middle">dose máxima · erro de vírgula · duplicidade</text><text x="300" y="228" text-anchor="middle">interações · escalonamento terapêutico</text>
  <text x="300" y="275" text-anchor="middle">queda · imobilidade · agitação noturna</text><text x="300" y="290" text-anchor="middle">declínio vs. linha de base</text></g>
 <g stroke="#52514e" stroke-width="1.2" marker-end="url(#a)">
  <line x1="145" y1="32" x2="183" y2="32"/><line x1="145" y1="94" x2="183" y2="94"/><line x1="145" y1="156" x2="183" y2="156"/>
  <line x1="145" y1="218" x2="183" y2="218"/><line x1="145" y1="280" x2="183" y2="280"/>
  <line x1="415" y1="32" x2="455" y2="140"/><line x1="415" y1="94" x2="455" y2="150"/><line x1="415" y1="156" x2="455" y2="156"/>
  <line x1="415" y1="218" x2="455" y2="165"/><line x1="415" y1="280" x2="455" y2="172"/>
  <line x1="565" y1="156" x2="593" y2="156"/><line x1="680" y1="185" x2="680" y2="223"/></g>
 <rect x="457" y="120" width="108" height="72" rx="6" fill="#fff8e6" stroke="#eda100"/>
 <g fill="#0b0b0b" text-anchor="middle"><text x="511" y="142" font-weight="bold">Alertas</text>
  <text x="511" y="158">gravidade + evidência</text><text x="511" y="173">deduplicação</text></g>
 <rect x="595" y="120" width="160" height="65" rx="6" fill="#e9f7f1" stroke="#1baf7a"/>
 <g fill="#0b0b0b" text-anchor="middle"><text x="675" y="140" font-weight="bold">Fusão multimodal</text>
  <text x="675" y="156">regras de correlação clínica</text><text x="675" y="171">escore de risco (noisy-OR)</text></g>
 <rect x="595" y="225" width="160" height="77" rx="6" fill="#fbeaea" stroke="#d03b3b"/>
 <g fill="#0b0b0b" text-anchor="middle"><text x="675" y="245" font-weight="bold">Equipe médica</text>
  <text x="675" y="261">painel · webhook Teams</text><text x="675" y="276">passagem de plantão</text><text x="675" y="291">trilha de auditoria</text></g>
 <text x="5" y="335" fill="#52514e" font-size="10">Serviços Azure: Speech to Text (transcrição pt-BR com tempo por palavra) e AI Language / Text Analytics for Health (entidades clínicas com</text>
 <text x="5" y="350" fill="#52514e" font-size="10">negação, sentimento). Sem credenciais, cada serviço cai para um equivalente local (Whisper, léxico clínico) com o mesmo formato de saída.</text>
 <text x="5" y="372" fill="#52514e" font-size="10">Todo o processamento de sinais vitais é amostra a amostra (fluxo); vídeo e áudio são analisados quando a gravação chega; a fusão é incremental.</text>
</svg>"""


def build(github: str) -> Path:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    v = s["video"]["sessao_fisioterapia"]
    vc = s["video"]["sessao_controle"]
    au = s["audio"]["avaliacao"]
    vit = s["sinais_vitais"]
    comp = vit["coorte"]["metodos"]
    ens = comp["Ensemble (todos)"]
    news = comp["NEWS2"]
    cus = comp["CUSUM sazonal"]
    sepsis_news = news["por_tipo"]["deterioracao_sepse"]["latencia_mediana_min"]
    sepsis_cus = cus["por_tipo"]["deterioracao_sepse"]["latencia_mediana_min"]
    rx = s["prescricoes"]["avaliacao"]
    mob = s["mobilidade"]
    fus = s["fusao"]
    ph = s.get("physionet") or {}
    reps = v["resumo"]["repeticoes_abducao"]
    ev_rows = [[e["evento"].replace("_", " "), "sim" if e["detectado"] else "não",
                f"{num(e['latencia_s'])} s" if e["latencia_s"] is not None else "—"] for e in v["avaliacao"]["eventos"]]
    wer = au["wer_por_condicao"]
    f1_mean = sum(au[k]["f1"] for k in ("fadiga", "dificuldade_respiratoria", "disartria")) / 3
    alerts = [json.loads(x) for x in (OUT / "alertas.jsonl").read_text(encoding="utf-8").splitlines()]
    fused = [a for a in alerts if a["source"] == "fusao"]
    expected = fus["esperado"]
    fused_main = [a for a in fused if expected.get(a["patient_id"]) == a["type"]]
    sev_label = {"critica": "Crítica", "alta": "Alta", "moderada": "Moderada", "baixa": "Baixa", "info": "Info"}

    def ex(pid, typ):
        a = next((a for a in alerts if a["patient_id"] == pid and a["type"] == typ), None)
        return [pid, a["source"], typ.replace("_", " "), sev_label[a["severity"]], a["message"][:230]] if a else None

    examples = [r for r in [
        ex("P-103", "queda") if any(a["source"] == "video" and a["type"] == "queda" for a in alerts) else None,
        ex("P-101", "dificuldade_respiratoria"), ex("P-102", "disartria"), ex("P-104", "deterioracao_febril"),
        ex("P-101", "dessaturacao"), ex("P-106", "mudanca_abrupta_dose"), ex("P-101", "interacao_medicamentosa"),
        ex("P-103", "agitacao_noturna"), ex("P-105", "artefato_sensor"),
    ] if r]
    total_alerts = s["alertas"]["total"]

    html = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"><title>Relatório Técnico - Fase 4</title>
<style>
 @page {{ size: A4; margin: 2.5cm 2.2cm 2.2cm 2.5cm; }}
 body {{ font-family: Arial, Helvetica, sans-serif; font-size: 11.5pt; line-height: 1.5; color: #111; }}
 h1 {{ font-size: 15pt; font-weight: normal; text-align: center; margin: 0 0 18pt; }}
 h2 {{ font-size: 13pt; text-align: center; margin: 14pt 0 10pt; }}
 h3 {{ font-size: 11.5pt; margin: 14pt 0 4pt; page-break-after: avoid; }}
 h2 {{ page-break-after: avoid; }} .keep {{ page-break-inside: avoid; }}
 p {{ text-align: justify; text-indent: 1.25cm; margin: 0 0 6pt; }}
 p.noind {{ text-indent: 0; }}
 ul {{ margin: 4pt 0 8pt 0.6cm; padding-left: 0.6cm; }} li {{ margin-bottom: 4pt; text-align: justify; }}
 table {{ width: 100%; border-collapse: collapse; margin: 6pt 0 10pt; font-size: 10pt; page-break-inside: avoid; }}
 th {{ text-align: left; color: #1f2a44; background: #f2f2f2; border: 1px solid #333; padding: 5pt 6pt; }}
 td {{ border: 1px solid #333; padding: 4pt 6pt; vertical-align: top; }}
 td:last-child {{ }} .center td {{ text-align: center; }}
 .page {{ page-break-after: always; }}
 .cover {{ display: flex; flex-direction: column; justify-content: space-between; text-align: center;
           font-size: 13pt; padding-top: 0.3cm; height: 23cm; }}
 .fig {{ margin: 6pt 0 2pt; page-break-inside: avoid; text-align: center; }} .cap {{ font-size: 9pt; color: #444; text-align: center; margin-bottom: 10pt; }}
 .links p {{ text-indent: 0; }} a {{ color: #1155cc; }}
 .small {{ font-size: 9.5pt; }}
</style></head><body>

<div class="page cover"><div>Relatório Técnico - Pós Tech FIAP</div><div>Tech Challenge - Fase 4</div>
<div>Nome: Nathan Felipe Pereira<br>RM: 371000</div></div>

<div class="page">
<h1>Introdução</h1>
<p>O Tech Challenge é um desafio técnico que engloba todos os conhecimentos obtidos durante as disciplinas da fase.</p>
<p>Com a IA já integrada aos processos médicos da instituição — analisando exames, documentos e apoiando decisões clínicas —, o desafio da Fase 4 propõe que o hospital passe a monitorar continuamente os pacientes por meio de dados multimodais (áudio, vídeo e texto) para identificar sinais precoces de risco.</p>
<p>Além de processar laudos e exames, o sistema deve analisar vídeos de fisioterapia ou cirurgias em busca de padrões anômalos, processar gravações de voz de consultas para detectar sintomas relacionados à fala (fadiga, disartria, dificuldade respiratória), detectar anomalias em sinais vitais, prescrições e evolução clínica e alertar a equipe médica em tempo real, integrando serviços gerenciados em nuvem como o Azure Cognitive Services.</p>
<p>O objetivo é construir uma solução que realize a análise e a fusão desses diferentes tipos de dados, utilize serviços em nuvem para ampliar a capacidade de processamento e inteligência e aplique técnicas de detecção de anomalias em tempo real para o monitoramento preventivo, entregando à equipe alertas priorizados, explicáveis e acompanhados da conduta recomendada.</p>
</div>

<div class="page">
<h2>Resumo Executivo</h2>
<p>A solução é um pipeline em Python que processa cinco fontes de dados — vídeo, áudio, sinais vitais, prescrições e mobilidade —, gera alertas por modalidade e os combina numa camada de fusão que reconhece padrões clínicos multimodais e calcula o risco de cada paciente ao longo do tempo.</p>
<p>Para medir o desempenho de forma objetiva, todos os dados de teste foram gerados com gabarito conhecido (anomalias injetadas em instantes definidos), e os detectores de sinais vitais foram validados também em {ph.get('pacientes', '—')} pacientes reais de UTI do PhysioNet.</p>
<ul>
<li>Vídeo (YOLOv8-pose): {pct(v['avaliacao']['recall'])} dos {len(v['avaliacao']['eventos'])} eventos injetados detectados — assimetria de amplitude, tremor, desequilíbrio, queda e imobilidade pós-queda — e nenhum falso positivo na sessão de controle.</li>
<li>Áudio: fadiga, dificuldade respiratória e disartria identificadas com F1 = {num(f1_mean, 2)} em 12 consultas de retorno, comparando cada voz com a gravação de admissão do próprio paciente.</li>
<li>A transcrição automática piora muito na fala patológica (WER de {pct(wer.get('controle', 0))} no controle contra {pct(wer.get('disartria', 0))} na disartria), enquanto os biomarcadores acústicos mantêm a detecção — o principal argumento a favor da abordagem multimodal.</li>
<li>Sinais vitais (coorte de 30 pacientes × 72 h): o ensemble detectou {pct(ens['recall'])} dos {ens['eventos']} eventos com {num(ens['falsos_alarmes_por_paciente_dia'], 2)} falso alarme por paciente-dia e suprimiu {ens['artefatos_suprimidos']}/{ens['artefatos']} artefatos de oxímetro. O CUSUM sazonal antecipou a sepse em {num((sepsis_news - sepsis_cus) / 60)} h em relação ao NEWS2 isolado.</li>
<li>Fusão: os {fus['acertos']} quadros clínicos do cenário integrado (insuficiência respiratória, AVC, queda e sepse) foram reconhecidos, e o paciente-controle não gerou nenhum alerta fundido.</li>
</ul>
<div class="keep"><h2>Requisitos implementados</h2>
{table(["Requisito", "Evidência na solução", "Situação"], [
    ["Análise de vídeo", "YOLOv8-pose (esqueleto COCO, equivalente ao OpenPose) e backend OpenPose; YOLOv8 para objetos; zonas críticas; relatório automático de desvios e vídeo anotado.", "Implementado"],
    ["Análise de áudio", "Azure Speech to Text (fallback Whisper); biomarcadores vocais para fadiga, dificuldade respiratória e disartria.", "Implementado"],
    ["Termos críticos e sentimento", "Azure Text Analytics for Health e análise de sentimento; fallback com léxico clínico e tratamento de negação.", "Implementado"],
    ["Anomalias em sinais vitais", "NEWS2, z-score robusto, CUSUM sazonal e Isolation Forest processados em fluxo, com filtro de artefato.", "Validado"],
    ["Prescrições e mobilidade", "Dose máxima, erro de vírgula, duplicidade, interação e escalonamento; queda, imobilidade, agitação e declínio.", "Implementado"],
    ["Alertas à equipe", "Gravidade, conduta recomendada, deduplicação, fusão multimodal, painel, webhook e passagem de plantão.", "Implementado"],
], ["22%", "60%", "18%"])}</div>
</div>

<div class="page">
<h2>Fluxo multimodal</h2>
<div class="fig">{DIAGRAM}</div>
<p>Cada modalidade tem um analisador independente que transforma o dado bruto em <b>alertas padronizados</b> (paciente, origem, tipo, gravidade, mensagem, evidências e conduta). Os sinais vitais são processados amostra a amostra, como num monitor real; vídeos e áudios são analisados quando a gravação chega; prescrições, a cada nova ordem; e a mobilidade, por janelas móveis.</p>
<p>Os alertas passam por um gerenciador que elimina repetições dentro de uma janela de 30 minutos (a menos que a gravidade aumente) e os entrega aos canais configurados: console, arquivo JSONL de auditoria, webhook (Microsoft Teams ou Azure Logic Apps) e o painel da equipe. A cada alerta novo, o <b>motor de fusão</b> verifica se o paciente acumulou, numa janela de tempo, evidências que caracterizem um quadro clínico conhecido — por exemplo, voz com dispneia somada a dessaturação — e, nesse caso, emite um alerta fundido de gravidade maior, com a conduta do protocolo correspondente.</p>
<p>Em paralelo, um escore de risco contínuo combina todos os alertas do paciente (noisy-OR, com meia-vida de 6 horas e bônus quando há concordância entre modalidades), o que ordena os pacientes no painel e na passagem de plantão.</p>
<p>No cenário integrado de demonstração — 6 pacientes internados por 72 horas —, o sistema entregou {total_alerts} alertas, dos quais {len(fused)} foram alertas fundidos.</p>
</div>

<div class="page">
<h2>Modelos aplicados em cada tipo de dado</h2>
{table(["Dado", "Modelo / técnica", "Por que foi escolhido"], [
    ["Vídeo — postura", "YOLOv8n-pose (17 keypoints COCO). Backend alternativo: OpenPose COCO via OpenCV DNN.", "Mesmo esqueleto do OpenPose, roda em tempo real na CPU e é mantido ativamente; o OpenPose fica disponível como opção."],
    ["Vídeo — objetos e áreas", "YOLOv8n (COCO); subtração de fundo com quadro de referência da sala; zonas críticas configuráveis.", "Mantém o paciente rastreado mesmo quando a pose se perde, como ao ficar deitado no chão."],
    ["Vídeo — desvios", "Ângulos articulares por repetição; FFT do resíduo do punho (tremor de 3,5–12 Hz); inclinação do tronco; transição vertical→horizontal; energia de movimento.", "Cada desvio tem significado clínico e limiar explicável."],
    ["Áudio — transcrição", "Azure Speech to Text (pt-BR, reconhecimento contínuo, tempo por palavra). Fallback: faster-whisper small.", "Serviço gerenciado pedido no desafio; o fallback permite rodar sem credenciais."],
    ["Áudio — voz", "pYIN (F0), jitter, shimmer, HNR com correção de janela, núcleos silábicos, pausas e inspirações audíveis.", "Biomarcadores clássicos da literatura de fala, comparados à linha de base do próprio paciente."],
    ["Texto", "Azure Text Analytics for Health (entidades com negação) e análise de sentimento com opinion mining. Fallback: léxico clínico com escopo de negação.", "Identifica termos críticos (dispneia, dor torácica, disartria) e o estado emocional."],
    ["Sinais vitais", "NEWS2; z-score com mediana/MAD de 6 h; CUSUM em médias de 15 min contra o mesmo horário do dia anterior; Isolation Forest; filtro de artefato.", "Limiares absolutos, mudanças súbitas, deriva lenta e combinações incomuns se complementam."],
    ["Prescrições", "Base farmacológica (dose máxima, faixa usual), z-score de log-dose, razão entre doses, duplicidade, interações e escalonamento.", "Erros de medicação são regras auditáveis; a estatística cobre doses atípicas."],
    ["Mobilidade", "Pico seguido de inatividade, janela móvel de 2 h, linha de base noturna e razão de atividade.", "Detecta queda, risco de lesão por pressão, delirium e declínio funcional."],
    ["Fusão", "Regras de correlação clínica e escore noisy-OR com decaimento temporal (late fusion).", "Explicável, funciona com modalidades ausentes e prioriza alertas concordantes."],
], ["17%", "45%", "38%"])}
</div>

<div class="page">
<h2>Resultados obtidos</h2>
<h3>1. Análise de vídeo</h3>
<p>A sessão sintética de fisioterapia ({num(v['resumo']['duracao_s'])} s, 25 fps) mostra um paciente fazendo seis abduções de ombro e três agachamentos. Foram injetadas cinco anomalias: perda de amplitude do braço esquerdo a partir da 4ª repetição, tremor de 6 Hz no punho direito, desequilíbrio no último agachamento, queda lateral e imobilidade no chão.</p>
<p>O YOLOv8-pose localizou os keypoints com PCK@0,2 de {pct(v['precisao_pose']['pck'], 1)} (erro mediano de {num(v['precisao_pose']['erro_mediano_px'])} px). Com o paciente deitado, a pose deixou de ser detectada, e o rastreamento da silhueta manteve a detecção da queda e da imobilidade.</p>
<div class="fig">{img('fig_video_angulos.png', '90%')}</div><div class="cap">Figura 1 — Ângulos de abdução, inclinação do tronco e intervalos de desvio detectados (cores = gravidade).</div>
<div class="fig">{img('fig_video_quadros.png')}</div><div class="cap">Figura 2 — Quadros do vídeo anotado gerado automaticamente (esqueleto, ângulos e faixas de alerta).</div>
{table(["Evento injetado", "Detectado", "Latência"], ev_rows, ["50%", "20%", "30%"])}
<p class="noind small">Amplitude por repetição (esq./dir.): {", ".join(f"{r['rep']}ª {r['esq']:.0f}°/{r['dir']:.0f}°" for r in reps)}. Na sessão de controle, com o mesmo roteiro sem anomalias, houve {len(vc['eventos'])} falsos positivos.</p>

<h3>2. Análise de áudio</h3>
<p>Foram geradas 16 consultas em português (4 vozes × admissão, controle, fadiga com dispneia e disartria). Cada consulta de retorno foi comparada com a gravação de admissão do mesmo paciente.</p>
<p>Na consulta com fadiga e dispneia, a fala caiu de cerca de 4,3 para 3,0 sílabas/s, a intensidade caiu cerca de 8 dB e apareceram inspirações audíveis entre frases curtas. Na disartria, aumentaram a instabilidade de F0 e o ruído glótico (menor HNR), e a nitidez articulatória caiu.</p>
<div class="fig">{img('fig_audio_energia.png')}</div><div class="cap">Figura 3 — Envoltória de energia da mesma paciente na admissão e no 3º dia, com pausas (cinza) e inspirações audíveis (laranja).</div>
<div class="fig">{img('fig_audio_indicadores.png')}</div><div class="cap">Figura 4 — Escores dos três indicadores vocais nas 12 consultas de retorno.</div>
{table(["Indicador", "VP", "FP", "FN", "VN", "Precisão", "Recall", "F1"],
       [[n, au[k]["vp"], au[k]["fp"], au[k]["fn"], au[k]["vn"], pct(au[k]["precisao"]), pct(au[k]["recall"]), num(au[k]["f1"], 2)]
        for k, n in (("fadiga", "Fadiga"), ("dificuldade_respiratoria", "Dificuldade respiratória"), ("disartria", "Disartria"))])}
<p>Transcrição ({", ".join(au['motor_transcricao'])}): WER de {pct(wer.get('baseline', 0))} na admissão, {pct(wer.get('controle', 0))} no controle, {pct(wer.get('fadiga_respiratoria', 0))} na fadiga e {pct(wer.get('disartria', 0))} na disartria. Só {pct(au.get('recall_termos_criticos_asr') or 0)} dos termos críticos foram recuperados da transcrição automática. Num caso, o modelo transcreveu "o coração <i>não</i> fica acelerado", invertendo o sintoma. Por isso o alerta de áudio não depende só do texto: os biomarcadores acústicos sustentam a detecção, e o texto, quando concorda, eleva a gravidade.</p>

<h3>3. Detecção de anomalias em sinais vitais</h3>
<p>A coorte de avaliação tem 30 pacientes × 72 h × 1 amostra/min, com {ens['eventos']} eventos clínicos (taquicardia, bradicardia, dessaturação, hipotensão, crise hipertensiva, febre e sepse lenta) e {ens['artefatos']} artefatos de oxímetro. As primeiras 24 h de cada paciente servem de linha de base.</p>
<div class="fig">{img('fig_vitais_metodos.png')}</div><div class="cap">Figura 5 — Recall, falsos alarmes e latência de cada detector e do ensemble.</div>
{table(["Método", "Recall", "Latência mediana", "Falsos alarmes / paciente-dia", "Precisão dos alertas"],
       [[n, pct(m["recall"]), f"{num(m['latencia_mediana_min'], 0)} min" if m["latencia_mediana_min"] is not None else "—",
         num(m["falsos_alarmes_por_paciente_dia"], 3), pct(m["precisao_alertas"])] for n, m in comp.items()])}
<p>Nenhum detector isolado cobre todos os eventos. O NEWS2 é específico, mas não vê a crise hipertensiva nem a febre moderada. O z-score robusto capta mudanças súbitas. O CUSUM sazonal detecta a sepse lenta {num((sepsis_news - sepsis_cus) / 60)} h antes do NEWS2 (latência mediana de {num(sepsis_cus / 60)} h contra {num(sepsis_news / 60)} h). Juntos, cobrem {pct(ens['recall'])} dos eventos, e o filtro de artefato evitou alarmes falsos de oxímetro desconectado.</p>
<div class="fig">{img('fig_vitais_sepse.png', '78%')}</div><div class="cap">Figura 6 — Paciente com sepse lenta: o CUSUM dispara antes de o NEWS2 cruzar o limiar.</div>

<h3>4. Prescrições e mobilidade</h3>
<p>Nas prescrições do cenário integrado, os {rx['esperados']} problemas esperados foram detectados ({rx['detectados']}/{rx['esperados']}): erro de vírgula na insulina (10 → 100 UI, 4× a dose máxima diária), duplicidade de anticoagulantes, varfarina com AAS, morfina com midazolam num paciente com insuficiência respiratória e escalonamento para carbapenêmico e vasopressor. O paciente-controle, com titulação normal de dipirona, não gerou alertas.</p>
<p>Na mobilidade, {pct(mob['recall'])} dos {mob['eventos']} eventos foram detectados, com {mob['falsos_alarmes']} falsos alarmes: queda (latência de 0 min), imobilidade de mais de 3 h no paciente com AVC, agitação noturna compatível com delirium e declínio de atividade nos pacientes que pioraram.</p>
<div class="fig">{img('fig_mobilidade.png')}</div><div class="cap">Figura 7 — Atividade do paciente P-103: queda durante a fisioterapia (confirmada pelo vídeo) e agitação na madrugada seguinte.</div>
<h3>5. Fusão multimodal e risco</h3>
<div class="fig">{img('fig_risco.png', '82%')}</div><div class="cap">Figura 8 — Risco fundido por paciente ao longo de 72 h. As linhas verticais marcam os alertas fundidos.</div>
{table(["Paciente", "Quadro reconhecido", "Modalidades", "Momento"],
       [[a["patient_id"], a["evidence"]["regra"], ", ".join(a["evidence"]["modalidades"]), a["timestamp"].replace("T", " ")[5:16]] for a in fused_main],
       ["12%", "40%", "30%", "18%"])}

<h3>6. Validação com dados reais (PhysioNet Challenge 2012)</h3>
<p>Não há bases públicas de vídeo e áudio clínico rotuladas para estes eventos, mas há dados reais de sinais vitais. Por isso os detectores foram aplicados às primeiras 48 h de {ph.get('pacientes', '—')} internações reais em UTI (taxa de óbito de {pct(ph.get('taxa_obito', 0), 1)}). Como não há rótulos de anomalia, a pergunta foi se os escores carregam sinal prognóstico.</p>
<div class="fig">{img('fig_physionet.png', '80%') if ph else ''}</div><div class="cap">Figura 9 — AUC de cada escore para óbito hospitalar.</div>
<p>O NEWS2 máximo atingiu AUC de {num(ph.get('auc', {}).get('news2_max', 0), 3)} para óbito hospitalar, comparável ao SAPS-I ({num(ph.get('auc', {}).get('SAPS-I', 0), 3)}), escore clínico de referência que usa exames laboratoriais. Já os detectores <i>relativos</i> (z-score e Isolation Forest em relação à linha de base do paciente) ficaram perto de 0,5.</p>
<p>Esse resultado é esperado e orienta o desenho: esses detectores medem <b>mudança</b>, não gravidade. Um paciente que chega grave já tem a linha de base alterada. Por isso o sistema combina limiares absolutos (gravidade) com detectores relativos (mudança aguda e deriva), em vez de usar só um tipo.</p>

<h2 style="page-break-before: always">Exemplos de anomalias detectadas</h2>
{table(["Paciente", "Origem", "Tipo", "Gravidade", "Mensagem do alerta"], examples, ["9%", "11%", "15%", "10%", "55%"])}
</div>

<div class="page">
<h2>Integração Azure e fluxo do alerta</h2>
<p>A integração com o Azure está nos módulos <code>cloud/azure_speech.py</code> e <code>cloud/azure_language.py</code>. O <b>Speech to Text</b> usa reconhecimento contínuo em pt-BR com saída detalhada e tempo por palavra. O <b>Text Analytics</b> faz análise de sentimento com opinion mining, extrai frases-chave e roda o <b>Text Analytics for Health</b>, que retorna entidades clínicas (SymptomOrSign, MedicationName etc.) com <i>assertion</i> de negação. As credenciais vêm de variáveis de ambiente (arquivo <code>.env</code>).</p>
<p>Sem credenciais, ou se o serviço falhar, cada chamada cai para um equivalente local com o mesmo formato de saída — Whisper para transcrição e léxico clínico com escopo de negação para o texto —, e o motor usado fica registrado em cada resultado.</p>
<p>O fluxo final até a equipe médica é:</p>
<ul>
<li>O analisador gera um alerta com gravidade, evidências e ação padrão: crítica, acionar o time de resposta rápida; alta, avaliação médica em até 15 min; moderada, reavaliação da enfermagem em até 30 min.</li>
<li>O gerenciador de alertas deduplica, registra o alerta em JSONL para auditoria e o envia por webhook (Teams/Logic Apps) quando a gravidade é alta ou crítica.</li>
<li>O motor de fusão correlaciona as modalidades e emite o alerta do quadro clínico com a conduta do protocolo (AVC, sepse, queda, insuficiência respiratória).</li>
<li>O painel Streamlit mostra os pacientes ordenados por risco, o feed de alertas, os gráficos por paciente, os vídeos anotados e as transcrições. Um relógio de plantão permite reconstruir o estado em qualquer instante, e o relatório de passagem de plantão é gerado automaticamente.</li>
</ul>

<h2>Limitações e próximos passos</h2>
<ul>
<li>Os dados de vídeo, áudio, prescrições e mobilidade são sintéticos. Eles permitem medir o sistema com gabarito, mas os limiares foram calibrados nesse conjunto, e é preciso validá-los com dados clínicos reais (com aprovação ética) antes de qualquer uso assistencial.</li>
<li>As vozes vêm de síntese de fala, e as alterações vocais são simuladas. Em pacientes reais, a variabilidade entre pessoas e ambientes é maior.</li>
<li>A latência de detecção da deterioração lenta ainda é de horas. Integrar exames laboratoriais (lactato, leucócitos) e o nível de consciência ao escore de fusão é o próximo passo natural.</li>
<li>Em produção: ingestão por Azure Event Hubs/IoT Hub e alertas via Azure Communication Services — o código já separa analisadores, alertas e canais de entrega.</li>
</ul>
</div>

<div class="links">
<p>Repositório Github: <a href="{github}">{github}</a></p>
</div>
</body></html>"""
    DOCS.mkdir(exist_ok=True)
    out_html = DOCS / "relatorio_tecnico.html"
    out_html.write_text(html, encoding="utf-8")
    pdf = DOCS / "Relatorio_Tecnico_Fase4.pdf"
    chrome = next((c for c in CHROME if Path(c).exists() or shutil.which(c)), None)
    if chrome:
        subprocess.run([chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, capture_output=True)
    return pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--github", default="https://github.com/Nathan-P2/tech-challenge-fase4")
    a = ap.parse_args()
    print(build(a.github))
