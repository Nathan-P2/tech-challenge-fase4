# Relatório automático — sessao_fisioterapia.mp4

- Backend de pose: **ground-truth**
- Duração: 37.96 s · quadros: 950 · com pose: 767
- Repetições de abdução: 6 · agachamentos: 3

## Amplitude por repetição

| Rep | t (s) | Esq (°) | Dir (°) | Desvio |
|---|---|---|---|---|
| 1 | 4.48 | 166.3 | 170.4 | — |
| 2 | 7.48 | 172.9 | 171.6 | — |
| 3 | 10.52 | 171.7 | 170.5 | — |
| 4 | 13.48 | 104.0 | 169.2 | assimetria_amplitude |
| 5 | 16.48 | 105.5 | 170.8 | assimetria_amplitude |
| 6 | 19.48 | 103.2 | 169.5 | assimetria_amplitude |

## Desvios detectados

- **assimetria_amplitude** (moderada) 12.0s–21.0s: Repetição 4: abdução esq 104° vs dir 169° (Δ 65°); membro esquerdo abaixo da amplitude inicial (39% de perda). | última: Repetição 6: abdução esq 103° vs dir 170° (Δ 66°); membro esquerdo abaixo da amplitude inicial (39% de perda).
- **tremor** (moderada) 14.4s–22.6s: Oscilação rítmica de 6.0 Hz no punho direito (amplitude 5.8% do tronco) — compatível com tremor.
- **desequilibrio** (moderada) 27.9s–29.1s: Inclinação lateral do tronco de até 21° por 1.2s — perda de controle postural.
- **queda** (critica) 30.4s–30.7s: Transição vertical→horizontal em 0.3s; silhueta passou a horizontal — provável queda.
- **paciente_no_chao** (alta) 30.8s–38.0s: Cabeça/tronco do paciente dentro da zona 'piso' por 7.1s.
- **imobilidade_pos_queda** (critica) 31.8s–38.0s: Sem movimento detectável por 6.2s após a queda — possível perda de consciência ou lesão.

## Avaliação contra o gabarito

Recall 100% · precisão 100%

- assimetria_amplitude: detectado (latência -0.0s)
- tremor: detectado (latência -0.6s)
- desequilibrio: detectado (latência +0.9s)
- queda: detectado (latência +0.4s)
- imobilidade: detectado (latência +0.8s)

Precisão da pose: PCK@0.2 = 100.0%, erro mediano 1.75 px, cobertura 81%.
