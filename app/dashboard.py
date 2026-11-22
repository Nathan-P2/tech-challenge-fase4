"""Painel da equipe médica (Streamlit).

    streamlit run app/dashboard.py

Lê as saídas de `hospital-monitor run` e permite analisar novos áudios/vídeos.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT, DATA = ROOT / "output", ROOT / "data"

from hospital_monitor.alerts import SEVERITY_LABEL, SEVERITY_RANK  # noqa: E402
from hospital_monitor.cli import _load_env  # noqa: E402

_load_env()
st.set_page_config(page_title="Monitoramento Multimodal", page_icon="🩺", layout="wide")
SEV_COLOR = {"critica": "#d03b3b", "alta": "#ec835a", "moderada": "#fab219", "baixa": "#898781", "info": "#c3c2b7"}
SEV_ICON = {"critica": "🔴", "alta": "🟠", "moderada": "🟡", "baixa": "⚪", "info": "·"}


@st.cache_data
def load():
    if not (OUT / "summary.json").exists():
        return None
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    alerts = pd.DataFrame([json.loads(x) for x in (OUT / "alertas.jsonl").read_text(encoding="utf-8").splitlines()])
    alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])
    risk = pd.read_csv(OUT / "risco_pacientes.csv", parse_dates=["timestamp"])
    vitals = pd.read_csv(DATA / "sinais_vitais.csv", parse_dates=["timestamp"])
    mob = pd.read_csv(DATA / "mobilidade.csv", parse_dates=["timestamp"])
    return summary, alerts, risk, vitals, mob


loaded = load()
st.title("🩺 Monitoramento multimodal de pacientes")
if loaded is None:
    st.warning("Nenhuma execução encontrada. Rode `hospital-monitor run` e recarregue a página.")
    st.stop()
summary, alerts, risk, vitals, mob = loaded

# relógio simulado: permite "voltar no tempo" e ver o painel como estava naquele instante
start = pd.Timestamp(summary["periodo"]["inicio"])
end = start + pd.Timedelta(hours=summary["periodo"]["horas"])
with st.sidebar:
    st.header("Relógio do plantão")
    now = st.slider("Momento", min_value=start.to_pydatetime(), max_value=end.to_pydatetime(),
                    value=(start + pd.Timedelta(hours=56)).to_pydatetime(), step=pd.Timedelta(minutes=30).to_pytimedelta(), format="DD/MM HH:mm")
    now = pd.Timestamp(now)
    min_sev = st.select_slider("Gravidade mínima no feed", options=["baixa", "moderada", "alta", "critica"],
                               value="moderada")
    st.caption("Integração Azure")
    for label, keys in (("Speech to Text", ("AZURE_SPEECH_KEY", "AZURE_SPEECH_REGION")),
                        ("Text Analytics", ("AZURE_LANGUAGE_KEY", "AZURE_LANGUAGE_ENDPOINT"))):
        ok = all(os.getenv(k) for k in keys)
        st.write(("✅ " if ok else "⚪ ") + label + ("" if ok else " (fallback local)"))

visible = alerts[alerts["timestamp"] <= now]
tab_panel, tab_patient, tab_video, tab_audio, tab_metrics = st.tabs(
    ["Painel da equipe", "Paciente", "Vídeo", "Áudio", "Métricas"])

# ------------------------------------------------------------------ painel
with tab_panel:
    r_now = risk[risk["timestamp"] <= now].groupby("patient_id").tail(1).set_index("patient_id")
    order = sorted(summary["pacientes"], key=lambda p: -float(r_now.loc[p, "risco"]) if p in r_now.index else 0)
    level_icon = {"crítico": "🔴", "alto": "🟠", "moderado": "🟡", "baixo": "🟢"}
    cols = st.columns(3)
    for k, pid in enumerate(order):
        info = summary["pacientes"][pid]
        rv = float(r_now.loc[pid, "risco"]) if pid in r_now.index else 0.0
        lvl = r_now.loc[pid, "nivel"] if pid in r_now.index else "baixo"
        # quadros clínicos fundidos nas últimas 24 h
        fused = visible[(visible["patient_id"] == pid) & (visible["source"] == "fusao")
                        & (visible["timestamp"] >= now - pd.Timedelta(hours=24))]
        with cols[k % 3].container(border=True):
            st.markdown(f"**{pid}** · {info['leito']} · {info['idade']} anos")
            st.caption(info["motivo"])
            st.metric("Risco fundido", f"{rv:.2f}", f"{level_icon.get(lvl, '')} {lvl}", delta_color="off")
            for _, f in fused.iterrows():
                st.markdown(f"{SEV_ICON[f['severity']]} **{f['evidence']['regra']}**  \n"
                            f"<small>{f['timestamp']:%d/%m %H:%M}</small>", unsafe_allow_html=True)
    st.subheader("Feed de alertas")
    feed = visible[visible["severity"].map(SEVERITY_RANK) >= SEVERITY_RANK[min_sev]].sort_values("timestamp",
                                                                                                 ascending=False)
    for _, a in feed.head(40).iterrows():
        with st.container(border=True):
            st.markdown(f"{SEV_ICON[a['severity']]} **{SEVERITY_LABEL[a['severity']]}** · `{a['timestamp']:%d/%m %H:%M}` · "
                        f"**{a['patient_id']}** · {a['source']}/{a['type']}")
            st.write(a["message"])
            if a["source"] == "fusao":
                st.info(f"Conduta: {a['evidence']['conduta']}")
            else:
                st.caption(f"Ação: {a['acao_recomendada']}")

# ------------------------------------------------------------------ paciente
with tab_patient:
    pid = st.selectbox("Paciente", list(summary["pacientes"]))
    info = summary["pacientes"][pid]
    st.markdown(f"**{pid}** · {info['idade']} anos · {info['leito']} — {info['motivo']}. {info['historia']}")
    v = vitals[(vitals["patient_id"] == pid) & (vitals["timestamp"] <= now)]
    pa = visible[(visible["patient_id"] == pid) & (visible["severity"] != "info")]
    rules = alt.Chart(pa[["timestamp", "severity", "source", "type", "message"]]).mark_rule(strokeWidth=1.5).encode(
        x="timestamp:T", color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEV_COLOR),
                                                                        range=list(SEV_COLOR.values())),
                                         legend=alt.Legend(title="Gravidade")),
        tooltip=["timestamp:T", "source", "type", "message"])
    charts = []
    for col, title in (("fc", "FC (bpm)"), ("pas", "PAS (mmHg)"), ("spo2", "SpO₂ (%)"), ("fr", "FR (irpm)"),
                       ("temp", "Temp (°C)")):
        line = alt.Chart(v).mark_line(strokeWidth=1.2, color="#2a78d6").encode(
            x=alt.X("timestamp:T", title=None), y=alt.Y(f"{col}:Q", title=title, scale=alt.Scale(zero=False)),
            tooltip=["timestamp:T", f"{col}:Q"])
        charts.append((line + rules).properties(height=110))
    st.altair_chart(alt.vconcat(*charts).resolve_scale(x="shared"), width="stretch")
    rr = risk[(risk["patient_id"] == pid) & (risk["timestamp"] <= now)]
    st.altair_chart(alt.Chart(rr).mark_area(color="#2a78d6", opacity=0.3, line=True).encode(
        x="timestamp:T", y=alt.Y("risco:Q", scale=alt.Scale(domain=[0, 1]), title="Risco fundido"),
        tooltip=["timestamp:T", "risco", "nivel", "modalidades_6h"]).properties(height=140), width="stretch")
    m = mob[(mob["patient_id"] == pid) & (mob["timestamp"] <= now)].copy()
    m["media_1h"] = m["atividade"].rolling(60, min_periods=1).mean()
    st.altair_chart(alt.Chart(m).mark_line(color="#1baf7a").encode(
        x="timestamp:T", y=alt.Y("media_1h:Q", title="Atividade (média 1 h)")).properties(height=110),
        width="stretch")
    st.dataframe(pa[["timestamp", "severity", "source", "type", "message"]].sort_values("timestamp"),
                 width="stretch", hide_index=True)

# ------------------------------------------------------------------ vídeo
with tab_video:
    vres = summary["video"]["sessao_fisioterapia"]
    c1, c2 = st.columns([3, 2])
    anot = OUT / "video" / "sessao_fisioterapia" / vres.get("video_anotado", "")
    with c1:
        if anot.is_file():
            st.video(str(anot))
    with c2:
        st.markdown((OUT / "video" / "sessao_fisioterapia" / "video_report.md").read_text(encoding="utf-8"))
    st.divider()
    st.subheader("Analisar outro vídeo")
    up = st.file_uploader("Vídeo clínico (.mp4/.mov)", type=["mp4", "mov", "avi"])
    if up and st.button("Analisar vídeo (YOLOv8-pose)"):
        from hospital_monitor.video.analyzer import analyze_video
        from hospital_monitor.video.pose_backends import YoloPoseBackend

        tmp = Path(tempfile.mkdtemp())
        path = tmp / up.name
        path.write_bytes(up.getvalue())
        with st.spinner("Estimando pose quadro a quadro..."):
            res = analyze_video(path, tmp / "out", YoloPoseBackend(str(ROOT / "models" / "yolov8n-pose.pt"), conf=0.25))
        st.markdown((tmp / "out" / "video_report.md").read_text(encoding="utf-8"))
        if (tmp / "out" / "video_anotado.mp4").exists():
            st.video(str(tmp / "out" / "video_anotado.mp4"))

# ------------------------------------------------------------------ áudio
with tab_audio:
    rep = json.loads((OUT / "audio" / "audio_report.json").read_text(encoding="utf-8"))
    names = [c["arquivo"] for c in rep["consultas"]]
    sel = st.selectbox("Consulta", names, index=names.index("P-101_fadiga_respiratoria.wav")
                       if "P-101_fadiga_respiratoria.wav" in names else 0)
    c = next(x for x in rep["consultas"] if x["arquivo"] == sel)
    st.audio(str(DATA / "audio" / sel))
    st.markdown(f"**Transcrição** ({c['transcricao']['motor']}, WER {c['wer']:.0%}): {c['transcricao']['texto']}")
    cols = st.columns(3)
    for col, (k, ind) in zip(cols, c["indicadores"].items()):
        col.metric(ind["rotulo"], f"{ind['escore']:.2f}", "detectado" if ind["detectado"] else "normal",
                   delta_color="inverse" if ind["detectado"] else "off")
        col.dataframe(pd.DataFrame(ind["componentes"])[["biomarcador", "valor", "referencia", "contribuicao"]],
                      hide_index=True, width="stretch")
    terms = [t for t in c["nlp"]["termos"]]
    st.markdown("**Termos clínicos** (" + c["nlp"]["motor"] + "): " +
                (", ".join(f"~~{t['normalizado']}~~" if t["negado"] else t["normalizado"] for t in terms) or "—"))
    st.markdown(f"**Sentimento:** {c['nlp']['sentimento']['sentimento']} {c['nlp']['sentimento']['scores']}")
    st.divider()
    st.subheader("Analisar nova gravação")
    up = st.file_uploader("Áudio da consulta (.wav/.mp3/.m4a)", type=["wav", "mp3", "m4a", "ogg"])
    base_opt = st.selectbox("Linha de base", ["referência populacional"] +
                            [n for n in names if "baseline" in n])
    if up and st.button("Analisar áudio"):
        from hospital_monitor.audio.analyzer import analyze_consultation
        from hospital_monitor.audio.features import extract_features, load_audio

        tmp = Path(tempfile.mkdtemp()) / up.name
        tmp.write_bytes(up.getvalue())
        baseline = None if base_opt.startswith("ref") else extract_features(load_audio(DATA / "audio" / base_opt))[0].to_dict()
        with st.spinner("Transcrevendo (Azure Speech → Whisper) e extraindo biomarcadores..."):
            res = analyze_consultation(tmp, "UPLOAD", datetime.now(), baseline)
        st.write(res["transcricao"]["texto"])
        st.json({k: {"escore": v["escore"], "detectado": v["detectado"]} for k, v in res["indicadores"].items()})
        for a in res["_alerts"]:
            st.warning(a.format())

# ------------------------------------------------------------------ métricas
with tab_metrics:
    s = summary
    c1, c2, c3, c4 = st.columns(4)
    ev = s["video"]["sessao_fisioterapia"]["avaliacao"]
    c1.metric("Vídeo — recall de eventos", f"{ev['recall']:.0%}", f"precisão {ev['precisao']:.0%}")
    au = s["audio"]["avaliacao"]
    c2.metric("Áudio — F1 médio", f"{sum(au[k]['f1'] for k in ('fadiga', 'dificuldade_respiratoria', 'disartria')) / 3:.2f}")
    ens = s["sinais_vitais"]["coorte"]["metodos"].get("Ensemble (todos)", {})
    c3.metric("Sinais vitais — recall", f"{ens.get('recall', 0):.0%}",
              f"{ens.get('falsos_alarmes_por_paciente_dia', 0):.2f} FA/paciente-dia", delta_color="off")
    c4.metric("Fusão — cenários reconhecidos", f"{s['fusao']['acertos']}/{len(s['fusao']['esperado'])}")
    st.subheader("Comparação de métodos — sinais vitais")
    st.dataframe(pd.DataFrame(s["sinais_vitais"]["coorte"]["metodos"]).T[
        ["recall", "latencia_mediana_min", "falsos_alarmes_por_paciente_dia", "precisao_alertas",
         "artefatos_suprimidos"]], width="stretch")
    figs = OUT / "figuras"
    for f in sorted(figs.glob("*.png")):
        st.image(str(f), caption=f.stem)
