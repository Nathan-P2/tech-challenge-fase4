# Passagem de plantão — monitoramento multimodal

Período: 2026-11-19T00:00:00 + 72 h · alertas entregues: 58 (suprimidos por deduplicação: 2)

## P-101 · 68 anos · UTI-03 — risco pico 1.00 (final: baixo)

*Pós-operatório de artroplastia de joelho.* Evolui com insuficiência respiratória no 3º dia (dessaturação progressiva, fala cansada e entrecortada, prescrição de opioide + benzodiazepínico).

**⚠ CRÍTICA — Suspeita de insuficiência respiratória** (2026-11-21T01:00:00)
Conduta: Avaliar via aérea, gasometria e oxigenoterapia; rever sedação/opioides.

- [sinais_vitais] Queda súbita de SpO₂: 91 (linha de base 97, z = -8.6).
- [sinais_vitais] NEWS2 = 4 por 5 min (SpO₂ 90, FC 91).
- [sinais_vitais] Combinação incomum de sinais vitais (Isolation Forest): SpO₂ 86 (z -15.7), FR 24 (z +5.4), FC 102 (z +4.7).
- [sinais_vitais] Elevação súbita de FR: 24 (linha de base 16, z = +5.4).
- [sinais_vitais] Tendência sustentada de piora (CUSUM) em FC↑, SpO₂↓, FR↑ vs. mesmo horário do dia anterior.
- [audio] Fadiga / cansaço vocal na voz (escore 0.91): queda de intensidade (dB) (-23.756 vs -15.981), lentificação da fala (3.034 vs 4.263), mais pausas por minuto (14.963 vs 9.714). Relato do paciente confirma: fadiga.

**⚠ ALTA — Risco medicamentoso em paciente instável** (2026-11-21T02:36:00)
Conduta: Farmácia clínica e médico assistente devem revisar a prescrição imediatamente.

- [sinais_vitais] Queda súbita de SpO₂: 91 (linha de base 97, z = -8.6).
- [sinais_vitais] NEWS2 = 4 por 5 min (SpO₂ 90, FC 91).
- [sinais_vitais] Combinação incomum de sinais vitais (Isolation Forest): SpO₂ 86 (z -15.7), FR 24 (z +5.4), FC 102 (z +4.7).
- [sinais_vitais] Elevação súbita de FR: 24 (linha de base 16, z = +5.4).
- [sinais_vitais] Elevação súbita de FC: 103 (linha de base 81, z = +4.9).
- [sinais_vitais] Tendência sustentada de piora (CUSUM) em FC↑, SpO₂↓, FR↑ vs. mesmo horário do dia anterior.
- [audio] Dificuldade respiratória na voz (escore 0.97): frases curtas (2.433 vs 4.36), inspirações audíveis/min (17.456 vs 0.0), tempo em pausa (0.292 vs 0.059). Relato do paciente confirma: dispneia, dor torácica.
- [prescricao] midazolam + morfina: depressão respiratória (opioide + benzodiazepínico).

Outros alertas relevantes:
- CRÍTICA · 11-21T01:00 · audio/fadiga: Fadiga / cansaço vocal na voz (escore 0.91): queda de intensidade (dB) (-23.756 vs -15.981), lentificação da fala (3.034 vs 4.263), mais pausas por minuto (14.963 vs 9.714). Relato do paciente confirma: fadiga.
- CRÍTICA · 11-21T01:00 · audio/dificuldade_respiratoria: Dificuldade respiratória na voz (escore 0.97): frases curtas (2.433 vs 4.36), inspirações audíveis/min (17.456 vs 0.0), tempo em pausa (0.292 vs 0.059). Relato do paciente confirma: dispneia, dor torácica.
- ALTA · 11-20T23:45 · sinais_vitais/dessaturacao: Queda súbita de SpO₂: 91 (linha de base 97, z = -8.6).
- ALTA · 11-20T23:51 · sinais_vitais/news2_elevado: NEWS2 = 4 por 5 min (SpO₂ 90, FC 91).
- ALTA · 11-21T00:01 · sinais_vitais/taquipneia: Elevação súbita de FR: 24 (linha de base 16, z = +5.4).
- ALTA · 11-21T00:08 · sinais_vitais/taquicardia: Elevação súbita de FC: 103 (linha de base 81, z = +4.9).

## P-103 · 81 anos · ENF-07 — risco pico 1.00 (final: baixo)

*Reabilitação pós-fratura de fêmur.* Queda durante a fisioterapia filmada, taquicardia reativa e agitação noturna (delirium).

**⚠ CRÍTICA — Queda com repercussão clínica** (2026-11-20T10:00:31)
Conduta: Avaliação médica à beira-leito, investigar trauma craniano/fratura, reavaliar risco de queda.

- [mobilidade] Pico de aceleração (100) seguido de 10 min sem movimento — possível queda do leito.
- [video] Transição vertical→horizontal em 0.3s; silhueta passou a horizontal — provável queda.
- [video] Sem movimento detectável por 6.2s após a queda — possível perda de consciência ou lesão.

**⚠ ALTA — Risco medicamentoso em paciente instável** (2026-11-20T11:06:00)
Conduta: Farmácia clínica e médico assistente devem revisar a prescrição imediatamente.

- [sinais_vitais] Elevação súbita de FC: 124 (linha de base 89, z = +11.8).
- [prescricao] ondansetrona + tramadol: redução do efeito analgésico / risco serotoninérgico.

Outros alertas relevantes:
- CRÍTICA · 11-20T10:00 · mobilidade/queda: Pico de aceleração (100) seguido de 10 min sem movimento — possível queda do leito.
- CRÍTICA · 11-20T10:00 · video/queda: Transição vertical→horizontal em 0.3s; silhueta passou a horizontal — provável queda.
- CRÍTICA · 11-20T10:00 · video/imobilidade_pos_queda: Sem movimento detectável por 6.2s após a queda — possível perda de consciência ou lesão.
- ALTA · 11-20T10:00 · video/paciente_no_chao: Cabeça/tronco do paciente dentro da zona 'piso' por 7.2s.
- ALTA · 11-20T10:02 · sinais_vitais/taquicardia: Elevação súbita de FC: 124 (linha de base 89, z = +11.8).
- MODERADA · 11-20T10:00 · video/assimetria_amplitude: Repetição 4: abdução esq 110° vs dir 163° (Δ 53°); membro esquerdo abaixo da amplitude inicial (24% de perda). | última: Repetição 6: abdução esq 112° vs dir 160° (Δ 47°); membro esquerdo abaixo da amplitude inicial (22% de perda).

## P-104 · 59 anos · ENF-15 — risco pico 0.94 (final: moderado)

*Pneumonia comunitária.* Deterioração lenta compatível com sepse: FC/Temp/FR sobem e PA cai ao longo de 16 h, mobilidade cai e o antibiótico é escalonado.

**⚠ CRÍTICA — Suspeita de sepse / deterioração clínica** (2026-11-21T06:00:00)
Conduta: Abrir protocolo sepse: lactato, hemoculturas, antibiótico na 1ª hora, reavaliação hemodinâmica.

- [sinais_vitais] Tendência sustentada de piora (CUSUM) em FC↑, Temp↑ vs. mesmo horário do dia anterior.
- [mobilidade] Atividade das últimas 12 h = 57% do mesmo período na admissão.

Outros alertas relevantes:
- ALTA · 11-20T14:29 · sinais_vitais/deterioracao_febril: Tendência sustentada de piora (CUSUM) em FC↑, Temp↑ vs. mesmo horário do dia anterior.
- ALTA · 11-20T17:48 · sinais_vitais/news2_elevado: NEWS2 = 4 por 5 min (FR 25, FC 93).
- ALTA · 11-20T21:15 · sinais_vitais/news2_elevado: NEWS2 = 5 por 5 min (FR 25, PAS 102, FC 96).
- ALTA · 11-21T10:00 · prescricao/escalonamento_terapeutico: noradrenalina: início de vasopressor — sugere instabilidade hemodinâmica.
- MODERADA · 11-20T20:04 · sinais_vitais/padrao_multivariado_anomalo: Combinação incomum de sinais vitais (Isolation Forest): PAS 100 (z -2.2), FR 26 (z +2.0), Temp 37.7 (z +1.3).
- MODERADA · 11-20T23:19 · sinais_vitais/padrao_multivariado_anomalo: Combinação incomum de sinais vitais (Isolation Forest): FC 100 (z +2.0), Temp 38.0 (z +2.0), PAS 96 (z -0.8).

## P-102 · 74 anos · ENF-12 — risco pico 0.88 (final: baixo)

*Fibrilação atrial anticoagulada.* Crise hipertensiva seguida de disartria e imobilidade — suspeita de AVC; duplicidade de anticoagulantes na véspera.

**⚠ CRÍTICA — Suspeita de AVC (protocolo AVC)** (2026-11-20T06:00:00)
Conduta: Acionar protocolo AVC: escala NIHSS, TC de crânio imediata, glicemia; revisar anticoagulação.

- [sinais_vitais] Elevação súbita de PAS: 162 (linha de base 111, z = +11.5).
- [audio] Disartria (fala arrastada/instável) na voz (escore 0.90): instabilidade de F0 (4.397 vs 2.988), voz mais ruidosa (HNR) (11.378 vs 17.858), perda de nitidez articulatória (0.043 vs 0.078).

**⚠ ALTA — Risco medicamentoso em paciente instável** (2026-11-20T05:05:00)
Conduta: Farmácia clínica e médico assistente devem revisar a prescrição imediatamente.

- [prescricao] enoxaparina prescrito com varfarina ativo(s) — duplicidade de anticoagulante.
- [prescricao] aas + varfarina: risco aumentado de sangramento.
- [sinais_vitais] Elevação súbita de PAS: 162 (linha de base 111, z = +11.5).

Outros alertas relevantes:
- ALTA · 11-19T20:00 · prescricao/duplicidade_terapeutica: enoxaparina prescrito com varfarina ativo(s) — duplicidade de anticoagulante.
- ALTA · 11-19T21:00 · prescricao/interacao_medicamentosa: aas + varfarina: risco aumentado de sangramento.
- ALTA · 11-20T05:05 · sinais_vitais/hipertensao: Elevação súbita de PAS: 162 (linha de base 111, z = +11.5).
- ALTA · 11-20T06:00 · audio/disartria: Disartria (fala arrastada/instável) na voz (escore 0.90): instabilidade de F0 (4.397 vs 2.988), voz mais ruidosa (HNR) (11.378 vs 17.858), perda de nitidez articulatória (0.043 vs 0.078).
- ALTA · 11-20T08:56 · mobilidade/imobilidade_prolongada: Sem reposicionamento há 3 h 34 min — risco de lesão por pressão/rebaixamento de consciência.
- MODERADA · 11-20T12:00 · mobilidade/declinio_mobilidade: Atividade das últimas 12 h = 44% do mesmo período na admissão.

## P-106 · 63 anos · ENF-09 — risco pico 0.75 (final: baixo)

*Diabetes descompensado.* Taquicardia paroxística breve e erro de prescrição de insulina (10 → 100 UI).

Outros alertas relevantes:
- CRÍTICA · 11-20T16:00 · prescricao/dose_acima_maxima: insulina_regular: 100 UI 4x/dia = 400 UI/dia (máximo 100).
- ALTA · 11-19T15:01 · sinais_vitais/taquicardia: Elevação súbita de FC: 134 (linha de base 80, z = +12.1).
- ALTA · 11-19T15:07 · sinais_vitais/news2_elevado: NEWS2 = 5 por 5 min (FR 22, FC 133).
- ALTA · 11-20T16:00 · prescricao/mudanca_abrupta_dose: insulina_regular: dose mudou de 10 para 100 UI (10.0×) — padrão de erro de vírgula (10×).
- MODERADA · 11-19T15:04 · sinais_vitais/padrao_multivariado_anomalo: Combinação incomum de sinais vitais (Isolation Forest): FC 132 (z +11.7), FR 21 (z +2.0), SpO₂ 98 (z -1.4).

## P-105 · 45 anos · ENF-02 — risco pico 0.00 (final: baixo)

*Pós-operatório de colecistectomia.* Paciente estável (controle). Dois artefatos de oxímetro que não devem gerar alarme.

Sem alertas acionáveis no período. Registrar no prontuário.
