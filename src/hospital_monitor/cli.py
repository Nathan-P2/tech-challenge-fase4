"""Interface de linha de comando: `hospital-monitor <comando>`."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_env() -> None:
    env = ROOT / ".env"
    if env.exists():
        import os

        for line in env.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def cmd_run(args):
    from .pipeline import run

    run(Path(args.out), Path(args.data), args.pose_backend, args.transcriber, not args.rapido,
        physionet=args.physionet)


def cmd_audio(args):
    from .audio.analyzer import analyze_consultation
    from .audio.features import extract_features, load_audio

    baseline = extract_features(load_audio(args.baseline))[0].to_dict() if args.baseline else None
    res = analyze_consultation(args.arquivo, args.paciente, datetime.now(), baseline, args.transcriber)
    res.pop("_series")
    for a in res.pop("_alerts"):
        print(a.format())
    print(json.dumps({k: res[k] for k in ("transcricao", "indicadores", "biomarcadores")}, ensure_ascii=False,
                     indent=2, default=float)[:6000])


def cmd_video(args):
    import cv2

    from .video.analyzer import analyze_video, events_to_alerts
    from .video.pose_backends import make_backend

    kwargs = {"weights": str(ROOT / "models" / "yolov8n-pose.pt"), "conf": 0.15} if args.pose_backend == "yolo" else {}
    backend = make_backend(args.pose_backend, **kwargs)
    ref = cv2.imread(args.referencia) if args.referencia else None
    zones = json.loads(Path(args.zonas).read_text()) if args.zonas else []
    res = analyze_video(Path(args.arquivo), Path(args.out), backend, None, zones, ref)
    for a in events_to_alerts(res["_events"], args.paciente, datetime.now(), Path(args.arquivo).name):
        print(a.format())
    print(f"Relatório: {args.out}/video_report.md")


def cmd_stream(args):
    """Reproduz os alertas gerados como um feed em tempo real (simulação acelerada)."""
    from .alerts import Alert, ConsoleSink

    path = Path(args.out) / "alertas.jsonl"
    if not path.exists():
        raise SystemExit("Rode `hospital-monitor run` antes.")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    sink = ConsoleSink(args.min)
    prev = None
    for r in rows:
        r.pop("acao_recomendada", None)
        a = Alert(**r)
        ts = datetime.fromisoformat(a.timestamp)
        if prev is not None and args.speed > 0:
            time.sleep(min(2.0, (ts - prev).total_seconds() / args.speed))
        prev = ts
        sink.send(a)


def main(argv=None):
    _load_env()
    p = argparse.ArgumentParser(prog="hospital-monitor", description="Monitoramento multimodal de pacientes")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="pipeline completo (gera dados, analisa, funde, relata)")
    r.add_argument("--out", default=str(ROOT / "output"))
    r.add_argument("--data", default=str(ROOT / "data"))
    r.add_argument("--pose-backend", default="yolo", choices=["yolo", "openpose", "gt"])
    r.add_argument("--transcriber", default=None, choices=[None, "auto", "azure", "whisper", "referencia"])
    r.add_argument("--rapido", action="store_true", help="pula a comparação de métodos de sinais vitais")
    r.add_argument("--physionet", action="store_true", help="valida com dados reais do PhysioNet (download)")
    r.set_defaults(func=cmd_run)

    a = sub.add_parser("audio", help="analisa uma gravação de consulta")
    a.add_argument("arquivo")
    a.add_argument("--paciente", default="P-000")
    a.add_argument("--baseline", help="gravação de linha de base do mesmo paciente")
    a.add_argument("--transcriber", default=None)
    a.set_defaults(func=cmd_audio)

    v = sub.add_parser("video", help="analisa um vídeo clínico")
    v.add_argument("arquivo")
    v.add_argument("--paciente", default="P-000")
    v.add_argument("--pose-backend", default="yolo", choices=["yolo", "openpose"])
    v.add_argument("--referencia", help="quadro da sala vazia (câmera fixa)")
    v.add_argument("--zonas", help="JSON com zonas críticas")
    v.add_argument("--out", default=str(ROOT / "output" / "video_avulso"))
    v.set_defaults(func=cmd_video)

    s = sub.add_parser("stream", help="feed de alertas em tempo real (replay acelerado)")
    s.add_argument("--out", default=str(ROOT / "output"))
    s.add_argument("--speed", type=float, default=3600.0, help="segundos simulados por segundo real")
    s.add_argument("--min", default="moderada", help="gravidade mínima exibida")
    s.set_defaults(func=cmd_stream)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
