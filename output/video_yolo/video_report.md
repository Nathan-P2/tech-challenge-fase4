# Relatório automático — sessao_fisioterapia.mp4

- Backend de pose: **yolov8-pose**
- Duração: 37.96 s · quadros: 950 · com pose: 761
- Repetições de abdução: 6 · agachamentos: 14

## Amplitude por repetição

| Rep | t (s) | Esq (°) | Dir (°) | Desvio |
|---|---|---|---|---|
| 1 | 4.72 | 143.4 | 147.3 | — |
| 2 | 7.52 | 146.6 | 154.7 | — |
| 3 | 10.4 | 150.9 | 158.6 | — |
| 4 | 13.48 | 109.8 | 162.7 | assimetria_amplitude |
| 5 | 16.52 | 105.8 | 156.8 | assimetria_amplitude |
| 6 | 19.52 | 112.4 | 159.6 | assimetria_amplitude |

## Desvios detectados

- **assimetria_amplitude** (moderada) 12.0s–21.0s: Repetição 4: abdução esq 110° vs dir 163° (Δ 53°); membro esquerdo abaixo da amplitude inicial (24% de perda). | última: Repetição 6: abdução esq 112° vs dir 160° (Δ 47°); membro esquerdo abaixo da amplitude inicial (22% de perda).
- **tremor** (moderada) 14.7s–22.1s: Oscilação rítmica de 6.0 Hz no punho direito (amplitude 8.9% do tronco) — compatível com tremor.
- **desequilibrio** (moderada) 28.0s–29.0s: Inclinação lateral do tronco de até 23° por 1.1s — perda de controle postural.
- **queda** (critica) 30.4s–30.7s: Transição vertical→horizontal em 0.3s; silhueta passou a horizontal — provável queda.
- **paciente_no_chao** (alta) 30.8s–38.0s: Cabeça/tronco do paciente dentro da zona 'piso' por 7.2s.
- **imobilidade_pos_queda** (critica) 31.8s–38.0s: Sem movimento detectável por 6.2s após a queda — possível perda de consciência ou lesão.

## Avaliação contra o gabarito

Recall 100% · precisão 100%

- assimetria_amplitude: detectado (latência -0.0s)
- tremor: detectado (latência -0.3s)
- desequilibrio: detectado (latência +1.0s)
- queda: detectado (latência +0.4s)
- imobilidade: detectado (latência +0.8s)

Precisão da pose: PCK@0.2 = 82.4%, erro mediano 8.12 px, cobertura 80%.

## Objetos (YOLOv8)

- person: 25 quadros (conf. máx 0.92)
