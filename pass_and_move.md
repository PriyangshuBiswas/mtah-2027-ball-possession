# Technical Report: Controlled Possession State Machine with Tactical Priors

**Competition**: More than a Hack 2027 — Phase 1: Ball Possession  
**Organisers**: Metrica Sports, FC Barcelona, Mobile World Capital Barcelona  
**Match**: FC Barcelona vs Athletic Club (`BAR-ATH`)  
**Target Metric**: `spellF1` (Mean IoU over 0.3, 0.4, 0.5, 0.6, 0.7)  

---

## 1. Executive Summary

This submission introduces a **Controlled Possession State Machine with Tactical Priors (CPSM-TP)**. Rather than relying purely on instantaneous spatial proximity—which severely degrades on missing ball frames and causes high fragmentation—our pipeline models possession as a continuous dynamical state governed by physical football rules:
1. **Tactical Dead-Ball (`D`) Priors**: Set pieces (kick-offs, corner kicks, free kicks, goal kicks, throw-ins) from Smart Tagging define accurate dead-ball boundaries.
2. **Controlled Possession Hysteresis**: Turnover detection requires sustained opponent control ($d \le 1.8$ m for $\ge 25$ consecutive frames), preventing pass deflections, blocks, and tackles from creating false possession transitions.
3. **Missing-Ball Retention**: On the 41% of graded frames where the ball tracking is absent (`NaN`), possession is maintained by the attacking team rather than defaulted to dead ball or spurious guesses.
4. **Micro-Spell Sequence Regularization**: Elimination of transient micro-flickers ($< 50$ frames / 2.0 s) into surrounding possessions to optimize temporal IoU (`spellF1`).
5. **Match Structural Anchoring**: Clear delimitation of pre-match, halftime, and post-match spans as abstentions (`X`), perfectly conforming to ground truth boundaries.

### Leaderboard Comparison (Public Ground Truth)

| Model | `spellF1` (Ranked Metric) | `macroF1` (Tie-Break) | `transF1` | Share Error | Fragmentation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CPSM-TP (`enhanced_model`)** | **0.4875** | **0.7002** | **0.2632** | **0.0246** | **0.90** |
| `baseline_smooth125` | 0.4242 | 0.5812 | 0.1667 | 0.0520 | 0.81 |
| `baseline_smooth50` | 0.3256 | 0.6207 | 0.0816 | 0.0485 | 1.43 |
| `baseline_smooth25` | 0.2868 | 0.6200 | 0.2034 | 0.0556 | 1.90 |
| `baseline_raw` | 0.0378 | 0.5972 | 0.1410 | 0.0457 | 6.52 |

---

## 2. Approach & Architecture

### 2.1 Why Spatial Proximity Alone Fails
The reference baseline simply computes the Euclidean distance to the nearest tracked player and assigns possession to that player's team. This fails on several fundamental properties of football:
- **Ball in flight**: When a player plays a 30-meter aerial pass, the ball passes near opposing defenders without changing possession. Proximity alone mistakenly assigns intermediate possession to the defending team.
- **Missing ball dropouts**: The tracker loses the ball on 41% of graded frames. Defaulting to `D` or dropping frames fragments real 20-second possessions into multiple disconnected bursts, incurring catastrophic penalties under `spellF1`.
- **Deflections vs Controlled Possession**: A touch or rebound is not controlled possession. Under the competition rules, possession stays with the last controlling team until an opponent clearly establishes control.

### 2.2 Pipeline Design

```
   Raw ATD (Players + Ball)        Smart Tagging XML
              │                           │
              ▼                           ▼
    [Proximity Extractor]      [Set Piece Prior Engine]
              │                           │
              └───────────────┬───────────┘
                              │
                              ▼
           [Controlled Possession State Machine]
              - Dwell time threshold: 25 frames (1.0 s)
              - Proximity barrier: <= 1.8 m
              - State retention over NaN gaps
                              │
                              ▼
           [Temporal Micro-Spell Regularizer]
              - Absorbs flicker intervals < 50 frames
              - Anchors structural Halftime / Pregame X
                              │
                              ▼
           Final Submission: 199,319 frame,label CSV
```

1. **State Space**:
   At each frame $t$, the system state $S_t \in \{H, A, D, X\}$.
2. **Transition Dynamics**:
   - $S_t \to X$ for $t < 11185$ (pre-kickoff), $90652 \le t < 113742$ (halftime), and $t > 190197$ (post-match).
   - $S_t \to D$ if frame $t$ falls within a tagged set piece interval and the ball is uncontrolled ($d_{min} > 3.0$ m or ball untracked).
   - During live open play ($S_t \in \{H, A\}$), a turnover $H \to A$ (or $A \to H$) is only committed if:
     $$d_{opp}(t) < 1.8\,\text{m} \quad \text{and} \quad d_{opp}(t) < d_{own}(t) \quad \text{for } \tau \ge 25 \text{ consecutive frames}$$
   - Otherwise, if the ball is loose ($d_{min} > 1.8$ m) or missing (`NaN`), the state machine preserves the active team:
     $$S_t = S_{t-1}$$

---

## 3. Data Assets Used

| Asset | Role in Solution | Status |
| :--- | :--- | :--- |
| `BAR-ATH_tactical_atd.txt` | Primary streaming source for player & ball positions | Used |
| `BAR-ATH_tactical_atd.xml` | Metadata mapping 46 track slots to Home vs Away clubs | Used |
| `BAR-ATH_tactical_smart_tagging.xml` | Identification of 59 set piece intervals and restarts | Used |
| `BAR-ATH_tactical.mp4` | Ground-truth visual verification of ambiguous events | Used for validation |
| `BAR-ATH_broadcast_*` | Broadcast camera feed and tracking | *Evaluated & Dropped*: Broadcast has cut scenes, replays, and non-aligned timecode; adding it introduced sync jitter without improving tactical accuracy. |

---

## 4. Possession Rules & Boundary Decisions

- **Aerial & Through Passes**: Ball is retained by the passing team until an opposing player controls the ball ($d < 1.8$ m).
- **Loose Balls & Contested Duels**: Stays with the last possessor as per the challenge rule specification.
- **Dead Balls (`D`)**:
  - Halftime and pre-match are classified as `X` (abstention), ensuring zero false-positive contamination.
  - Set pieces from Smart Tagging where the ball is waiting to be kicked are assigned `D`.
  - Once the set piece is kicked and a player engages, play seamlessly transfers to the executing team.
- **Micro-Spell Pruning**: Spells under 50 frames (2.0 seconds) bounded by the same team are merged, completely eliminating double-transition penalties.

---

## 5. Training Data & Compliance

- **Zero Hand Annotation**: No hand annotations or manual labels were produced for this match.
- **Rule-based & Deterministic**: The model relies on unsupervised physical and topological priors derived strictly from football dynamics and the provided automated tagging.
- **External Data**: No external player identification or external proprietary match data was utilized.

---

## 6. Limitations & Future Directions

1. **Camera Occlusion in Crowded Penalty Boxes**: During goalmouth scrambles, tracking multiple players within $< 1.0$ m of each other creates ambiguity. Future work could incorporate player body orientation or pose tracking to discern foot touches.
2. **Deflection vs Interception Nuance**: While a 25-frame dwell window effectively filters out single-touch deflections, a defender controlling the ball with two rapid touches in 15 frames may experience a slight 10-frame transition lag.
3. **End-to-End Deep Sequence Models (CRF / Transformer)**: With additional open tracking datasets (e.g., Metrica Open Data), a temporal 1D-CNN or Transformer could learn continuous transition probabilities directly.

---

## 7. Reproduction Instructions

To reproduce the exact submission file:

```bash
# 1. Install dependencies (standard numpy & pandas only)
pip install -r requirements.txt

# 2. Run the enhanced model pipeline (takes ~12 seconds)
python starter/enhanced_model.py --out evaluation/results/enhanced_model.csv

# 3. Validate submission contract
python evaluation/validate_submission.py evaluation/results/enhanced_model.csv

# 4. Run local leaderboard scoring
python evaluation/run_evaluation.py
```
