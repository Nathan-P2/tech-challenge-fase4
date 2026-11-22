"""Figuras (PNG) e relatório de passagem de plantão."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .alerts import ACTION_BY_SEVERITY, SEVERITY_LABEL, SEVERITY_RANK  # noqa: E402

# paleta de referência (validada para daltonismo) e cores de estado reservadas para gravidade
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
STATUS = {"critica": "#d03b3b", "alta": "#ec835a", "moderada": "#fab219", "baixa": "#898781", "info": "#c3c2b7"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 1.6, "font.family": "DejaVu Sans",
})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_video(out: Path, video_dir: Path, summary: dict):
    df = pd.read_csv(video_dir / "video_frames.csv")
    events = summary["eventos"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    ax = axes[0]
    ax.plot(df["t"], df["abd_esq"], color=SERIES[0], label="Abdução ombro esq.")
    ax.plot(df["t"], df["abd_dir"], color=SERIES[1], label="Abdução ombro dir.")
    ax.set_ylabel("ângulo (°)")
    ax.set_title("Vídeo — ângulos articulares estimados pelo YOLOv8-pose e desvios detectados", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.02), ncol=2, fontsize=8)
    ax.set_ylim(0, 215)
    ax2 = axes[1]
    ax2.plot(df["t"], df["inclinacao_tronco"], color=SERIES[2], label="Inclinação do tronco")
    ax2.plot(df["t"], df["motion"].clip(upper=1.0) * 40, color=MUTED, lw=1, label="Energia de movimento (×40)")
    ax2.set_ylabel("graus")
    ax2.set_xlabel("tempo da sessão (s)")
    ax2.legend(loc="upper left", ncol=2)
    for e in events:
        for a in axes:
            a.axvspan(e["start"], e["end"], color=STATUS[e["severity"]], alpha=0.12, lw=0)
    # rótulos dos eventos em faixas acima das curvas, sem sobreposição
    rows_end: list[float] = []
    for e in events:
        if e["type"] in {"paciente_no_chao"}:
            continue
        label = e["type"].replace("_", " ")
        width = 0.32 * len(label)
        row = next((k for k, end in enumerate(rows_end) if e["start"] > end), len(rows_end))
        if row == len(rows_end):
            rows_end.append(0)
        rows_end[row] = e["start"] + width
        x = min(e["start"], 38 - width)
        axes[0].text(x + 0.2, 205 - 15 * row, label, fontsize=7.5, color=INK2, va="top")
    _save(fig, out / "fig_video_angulos.png")


def fig_video_frames(out: Path, video_path: Path, moments: list[tuple[float, str]]):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    fig, axes = plt.subplots(1, len(moments), figsize=(2.6 * len(moments), 2.4))
    for ax, (t, title) in zip(axes, moments):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, frame = cap.read()
        if ok:
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8, loc="left")
        ax.axis("off")
    cap.release()
    _save(fig, out / "fig_video_quadros.png")


def fig_audio(out: Path, audio_dir: Path, audio_report: dict):
    from .audio.features import extract_features, load_audio

    fig, axes = plt.subplots(2, 1, figsize=(8, 3.8), sharex=True)
    for ax, (name, label) in zip(axes, [("P-101_baseline.wav", "Admissão (linha de base)"),
                                        ("P-101_fadiga_respiratoria.wav", "3º dia — fadiga e dispneia")]):
        feats, s = extract_features(load_audio(audio_dir / name))
        ax.plot(s["t"], s["db"], color=SERIES[0], lw=0.8)
        for a, b in s["pausas"]:
            ax.axvspan(a, b, color=MUTED, alpha=0.15, lw=0)
        for a, b in s.get("respiracoes", []):
            ax.axvspan(a, b, color=STATUS["alta"], alpha=0.35, lw=0)
        ax.set_ylim(-95, 0)
        ax.set_ylabel("energia (dB)")
        ax.set_title(f"{label}: {feats.taxa_fala_sil_s:.1f} sílabas/s · {feats.pausas_por_min:.0f} pausas/min · "
                     f"{feats.respiracoes_audiveis} inspirações audíveis · {feats.intensidade_db:.0f} dB",
                     loc="left", fontsize=8.5)
    axes[1].set_xlabel("tempo (s) — cinza: pausas; laranja: inspirações audíveis detectadas")
    _save(fig, out / "fig_audio_energia.png")

    rows = [c for c in audio_report["consultas"] if c["condicao"] != "baseline"]
    labels = [f"{c['paciente']}\n{c['condicao'].replace('_respiratoria', '/resp.').replace('_', ' ')}" for c in rows]
    inds = [("fadiga", "Fadiga"), ("dificuldade_respiratoria", "Dif. respiratória"), ("disartria", "Disartria")]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9, 3.3))
    w = 0.26
    for k, (key, name) in enumerate(inds):
        ax.bar(x + (k - 1) * (w + 0.02), [c["indicadores"][key]["escore"] for c in rows], w, color=SERIES[k],
               label=name, edgecolor=SURFACE, linewidth=0.8)
    ax.axhline(0.5, color=INK2, lw=1, ls="--")
    ax.text(-0.45, 0.52, "limiar 0,5", ha="left", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("escore do indicador")
    ax.set_title("Áudio — escores vocais vs. linha de base do próprio paciente (12 consultas de retorno)", loc="left")
    ax.legend(loc="upper right", ncol=3)
    _save(fig, out / "fig_audio_indicadores.png")


def _alert_marks(ax, alerts, pid, sources=None, y=None):
    for a in alerts:
        if a.patient_id != pid or (sources and a.source not in sources) or a.severity == "info":
            continue
        t = pd.Timestamp(a.timestamp)
        ax.axvline(t, color=STATUS[a.severity], lw=1.1, alpha=0.9)


def fig_vitals(out: Path, sc: dict, alerts: list, pid: str = "P-104", name: str = "fig_vitais_sepse.png",
               title: str = ""):
    v = sc["sinais_vitais"]
    v = v[v["patient_id"] == pid].set_index("timestamp")
    sigs = [("fc", "FC (bpm)"), ("temp", "Temp (°C)"), ("fr", "FR (irpm)"), ("pas", "PAS (mmHg)"), ("spo2", "SpO₂ (%)")]
    fig, axes = plt.subplots(len(sigs), 1, figsize=(8, 6.2), sharex=True)
    firsts = {}
    for a in alerts:
        if a.patient_id == pid and a.source == "sinais_vitais" and a.severity != "info":
            key = a.evidence.get("metodo", a.type)
            firsts.setdefault(key, pd.Timestamp(a.timestamp))
    for k, (ax, (col, lab)) in enumerate(zip(axes, sigs)):
        ax.plot(v.index, v[col].rolling(5, min_periods=1).mean(), color=SERIES[0], lw=1)
        ax.set_ylabel(lab, fontsize=8)
        for j, (m, t) in enumerate(sorted(firsts.items(), key=lambda x: x[1])):
            ax.axvline(t, color=SERIES[1 + j % 5], lw=1.3, label=f"{m} — {t:%d/%m %H:%M}")
    axes[0].legend(loc="upper left", fontsize=7.5, ncol=2)
    axes[0].set_title(title or f"Sinais vitais — {pid}: primeiro alerta de cada método", loc="left")
    _save(fig, out / name)


def fig_methods(out: Path, comp: dict):
    names = list(comp)
    rec = [comp[n]["recall"] for n in names]
    fa = [comp[n]["falsos_alarmes_por_paciente_dia"] for n in names]
    lat = [comp[n]["latencia_mediana_min"] or 0 for n in names]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
    for ax, vals, title, fmt in [(axes[0], rec, "Recall de eventos", "{:.0%}"),
                                 (axes[1], fa, "Falsos alarmes / paciente-dia", "{:.2f}"),
                                 (axes[2], lat, "Latência mediana (min)", "{:.0f}")]:
        colors = [SERIES[0] if n.startswith("Ensemble") else "#86b6ef" for n in names]
        ax.barh(names, vals, color=colors, height=0.6, edgecolor=SURFACE)
        for i, v in enumerate(vals):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(vals) * 1.35 if max(vals) else 1)
    axes[0].invert_yaxis()
    _save(fig, out / "fig_vitais_metodos.png")


def fig_mobility(out: Path, sc: dict, alerts: list, pid: str = "P-103"):
    m = sc["mobilidade"]
    m = m[m["patient_id"] == pid].set_index("timestamp")
    fig, ax = plt.subplots(figsize=(8, 2.6))
    ax.plot(m.index, m["atividade"], color="#86b6ef", lw=0.5, label="atividade por minuto")
    ax.plot(m.index, m["atividade"].rolling(60, min_periods=1).mean(), color=SERIES[0], lw=1.6, label="média 60 min")
    for a in alerts:
        if a.patient_id == pid and a.source in {"mobilidade", "video"} and a.severity != "info":
            t = pd.Timestamp(a.timestamp)
            ax.axvline(t, color=STATUS[a.severity], lw=1.2)
            if a.type in {"queda", "agitacao_noturna"} and a.source == "mobilidade":
                ax.text(t, 95, " " + a.type.replace("_", " "), fontsize=7.5, color=INK2, va="top")
    ax.set_ylim(0, 100)
    ax.set_ylabel("índice de atividade")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2, fontsize=8)
    ax.set_title(f"Mobilidade — {pid}: queda na fisioterapia e agitação noturna", loc="left")
    _save(fig, out / "fig_mobilidade.png")


def fig_risk(out: Path, risk: pd.DataFrame, fused: list, patients: dict):
    pids = list(patients)
    fig, axes = plt.subplots(len(pids), 1, figsize=(8, 7), sharex=True, sharey=True)
    for ax, pid in zip(axes, pids):
        r = risk[risk["patient_id"] == pid]
        ax.fill_between(r["timestamp"], r["risco"], color=SERIES[0], alpha=0.15, lw=0)
        ax.plot(r["timestamp"], r["risco"], color=SERIES[0], lw=1.4)
        for thr in (0.3, 0.6, 0.85):
            ax.axhline(thr, color=GRID, lw=0.8, ls="--")
        for f in fused:
            if f.patient_id == pid:
                t = pd.Timestamp(f.timestamp)
                main = f.type != "risco_medicamentoso_paciente_instavel"
                ax.axvline(t, color=STATUS[f.severity], lw=1.4 if main else 1.0, ls="-" if main else ":")
                if main:
                    right = t > r["timestamp"].max() - pd.Timedelta(hours=20)
                    ax.text(t, 0.95, (f.evidence["regra"] + " ") if right else (" " + f.evidence["regra"]),
                            fontsize=7.5, color=INK, va="top", ha="right" if right else "left")
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(pid, rotation=0, ha="right", va="center", fontsize=8.5, color=INK)
    axes[0].set_title("Fusão multimodal — risco por paciente ao longo de 72 h", loc="left")
    _save(fig, out / "fig_risco.png")


def fig_physionet(out: Path, physio: dict):
    names = {"SAPS-I": "SAPS-I (escore clínico de referência)", "news2_max": "NEWS2 máximo (limiares)",
             "risco_combinado": "Risco combinado", "anomalias_z": "Anomalias z-score (relativas)",
             "if_horas_anomalas": "Isolation Forest (relativo)"}
    keys = [k for k in names if k in physio["auc"]]
    vals = [physio["auc"][k] for k in keys]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.barh([names[k] for k in keys], vals, color=[SERIES[0] if k == "news2_max" else "#86b6ef" for k in keys],
            height=0.6, edgecolor=SURFACE)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0.4, 0.75)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title(f"PhysioNet 2012 — AUC para óbito ({physio['pacientes']} pacientes reais de UTI)", loc="right")
    _save(fig, out / "fig_physionet.png")


def build_figures(out_dir: Path, data_dir: Path, sc: dict, vitals_hist, risk, alerts, summary):
    figs = out_dir / "figuras"
    figs.mkdir(exist_ok=True)
    vdir = out_dir / "video" / "sessao_fisioterapia"
    fig_video(figs, vdir, summary["video"]["sessao_fisioterapia"])
    anot = vdir / summary["video"]["sessao_fisioterapia"].get("video_anotado", "video_anotado.mp4")
    if anot.exists():
        fig_video_frames(figs, anot, [(5.2, "Rep. 1 — padrão normal"), (16.4, "Rep. 5 — assimetria + tremor"),
                                      (28.5, "Agachamento — desequilíbrio"), (30.6, "Queda"),
                                      (35.0, "Imobilidade no chão")])
    audio_report = json.loads((out_dir / "audio" / "audio_report.json").read_text(encoding="utf-8"))
    fig_audio(figs, data_dir / "audio", audio_report)
    fig_vitals(figs, sc, alerts, "P-104", "fig_vitais_sepse.png",
               "Sinais vitais — P-104 (sepse lenta): primeiro alerta de cada método")
    fig_vitals(figs, sc, alerts, "P-101", "fig_vitais_p101.png",
               "Sinais vitais — P-101 (insuficiência respiratória): primeiro alerta de cada método")
    fig_methods(figs, summary["sinais_vitais"]["coorte"]["metodos"])
    fig_mobility(figs, sc, alerts)
    fused = [a for a in alerts if a.source == "fusao"]
    fig_risk(figs, risk, fused, summary["pacientes"])
    if summary.get("physionet"):
        fig_physionet(figs, summary["physionet"])


def handoff_report(summary: dict, alerts: list) -> str:
    """Relatório de passagem de plantão: pacientes ordenados por risco, alertas fundidos primeiro."""
    pats = sorted(summary["pacientes"].items(), key=lambda kv: -kv[1]["risco_pico"])
    lines = ["# Passagem de plantão — monitoramento multimodal", "",
             f"Período: {summary['periodo']['inicio']} + {summary['periodo']['horas']} h · "
             f"alertas entregues: {summary['alertas']['total']} (suprimidos por deduplicação: "
             f"{summary['alertas']['suprimidos']})", ""]
    for pid, p in pats:
        pa = sorted([a for a in alerts if a.patient_id == pid and a.severity != "info"],
                    key=lambda a: (-SEVERITY_RANK[a.severity], a.timestamp))
        lines += [f"## {pid} · {p['idade']} anos · {p['leito']} — risco pico {p['risco_pico']:.2f} "
                  f"(final: {p['nivel_final']})", "", f"*{p['motivo']}.* {p['historia']}", ""]
        fused = [a for a in pa if a.source == "fusao"]
        for f in fused:
            lines += [f"**⚠ {SEVERITY_LABEL[f.severity]} — {f.evidence['regra']}** ({f.timestamp})",
                      f"Conduta: {f.evidence['conduta']}", ""]
            lines += [f"- [{e['origem']}] {e['mensagem']}" for e in f.evidence["evidencias"]]
            lines.append("")
        others = [a for a in pa if a.source != "fusao"][:6]
        if others:
            lines.append("Outros alertas relevantes:")
            lines += [f"- {SEVERITY_LABEL[a.severity]} · {a.timestamp[5:16]} · {a.source}/{a.type}: {a.message}"
                      for a in others]
        if not pa:
            lines.append("Sem alertas acionáveis no período. " + ACTION_BY_SEVERITY["info"])
        lines.append("")
    return "\n".join(lines)
