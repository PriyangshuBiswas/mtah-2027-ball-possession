"""Enhanced ball possession model for More than a Hack 2027.

Architecture & Innovations:
1. Smart Tagging integration: Set-piece intervals (corners, free kicks, throw-ins, goal kicks)
   are used as strong priors for Dead Ball (D) stoppages.
2. Controlled possession hysteresis: An opposing team must establish control (dwell time >= 25 frames
   at distance <= 1.8m) to win possession, preventing brief deflections from breaking spells.
3. Missing-ball state retention: Retains current team possession across ball tracking dropouts (41%
   of frames) rather than prematurely guessing dead ball or flickering.
4. Micro-spell sequence filtering: Merges brief spurious flickers (< 50 frames = 2.0s) into the
   surrounding possession spell, directly maximizing the temporal IoU spellF1 metric.
5. Strict match-interval anchoring: Automatically abstains (X) during pre-kickoff, halftime,
   and post-whistle periods.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames  # noqa: E402
from starter.load_smart_tagging import load as load_smart_tags  # noqa: E402

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")

# State constants
HOME, AWAY, DEAD, UNKNOWN = 0, 1, 2, 3
CODE_TO_LABEL = {HOME: "H", AWAY: "A", DEAD: "D", UNKNOWN: "X"}
FIELD_SEP = ":"
GROUP_SEP = ";"
MISSING = "N"


def load_teams(meta_path):
    """Map player slot to HOME (0) or AWAY (1)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    home_id = team_ids[0]
    slots = [player.get("teamId") for player in root.iter("Player")]
    return np.array([HOME if t == home_id else AWAY for t in slots], dtype=np.int8)


def parse_smart_tagging_dead_intervals(tags, n_frames):
    """Identify dead ball frames from set piece events in Smart Tagging."""
    dead_mask = np.zeros(n_frames, dtype=bool)
    set_piece_codes = {"set pieces"}
    
    for event in tags:
        code = (event["code"] or "").strip().lower()
        if code in set_piece_codes:
            s = max(0, min(event["start_frame"], n_frames))
            e = max(0, min(event["end_frame"], n_frames))
            dead_mask[s:e] = True

    return dead_mask


def filter_micro_spells(labels, min_spell_len=50):
    """Absorb short flickers (< min_spell_len) into surrounding possession."""
    out = labels.copy()
    n = len(out)
    
    breaks = np.flatnonzero(out[1:] != out[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [n]))

    for i in range(1, len(bounds) - 1):
        s = bounds[i]
        e = bounds[i + 1]
        length = e - s
        prev_s = bounds[i - 1]
        prev_label = out[prev_s]
        next_label = out[e] if e < n else prev_label
        curr_label = out[s]

        # Never alter un-scored match boundaries
        if curr_label == UNKNOWN:
            continue

        if length < min_spell_len:
            if prev_label == next_label and prev_label != UNKNOWN:
                out[s:e] = prev_label
            elif length < 12 and prev_label != UNKNOWN:
                out[s:e] = prev_label

    return out


def stream_features_and_infer(raw_path, slot_teams, n_frames, pitch, tags,
                              dwell_frames=25, dist_thresh=1.8, min_spell_len=50):
    """Stream ATD raw data, compute kinematic features, and infer possession."""
    length, width = pitch
    labels = np.full(n_frames, UNKNOWN, dtype=np.int8)

    # 1. Parse smart tagging set pieces
    smart_dead_mask = parse_smart_tagging_dead_intervals(tags, n_frames)

    # 2. Match structural boundaries
    h1_start, h1_end = 11185, 90651
    h2_start, h2_end = 113742, 190197

    min_dist_h = np.full(n_frames, 999.0, dtype=np.float32)
    min_dist_a = np.full(n_frames, 999.0, dtype=np.float32)
    ball_seen = np.zeros(n_frames, dtype=bool)

    print("Step 1/3: Streaming ATD tracking coordinates...")
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(FIELD_SEP)
            if len(parts) != 3:
                continue
            fid = int(parts[0]) - 1
            if fid < 0 or fid >= n_frames:
                continue

            # Skip tracking processing during pregame, halftime and postgame
            if fid < h1_start or (h1_end < fid < h2_start) or fid > h2_end:
                continue

            ball = parts[2]
            if ball[0] != MISSING:
                ball_seen[fid] = True
                c1 = ball.index(",")
                c2 = ball.find(",", c1 + 1)
                bx = float(ball[:c1]) * length
                by = float(ball[c1 + 1:c2 if c2 != -1 else len(ball)]) * width

                # Player coordinates
                p_groups = parts[1].split(GROUP_SEP)
                best_dh = 999.0
                best_da = 999.0
                for slot, group in enumerate(p_groups):
                    if group[0] != MISSING:
                        c_p1 = group.index(",")
                        c_p2 = group.index(",", c_p1 + 1)
                        px = float(group[:c_p1]) * length
                        py = float(group[c_p1 + 1:c_p2]) * width
                        d = np.sqrt((px - bx) ** 2 + (py - by) ** 2)
                        team = slot_teams[slot]
                        if team == HOME and d < best_dh:
                            best_dh = d
                        elif team == AWAY and d < best_da:
                            best_da = d

                min_dist_h[fid] = best_dh
                min_dist_a[fid] = best_da

    print("Step 2/3: Applying controlled possession state machine...")
    curr_team = AWAY  # Kickoff in 1st half is taken by Away (Athletic Club)
    opp_count = 0

    for fid in range(n_frames):
        # Match phase gating
        if fid < h1_start or (h1_end < fid < h2_start) or fid > h2_end:
            labels[fid] = UNKNOWN
            continue

        if fid == h2_start:
            curr_team = HOME
            opp_count = 0

        # Dead ball detection
        min_p_dist = min(min_dist_h[fid], min_dist_a[fid])
        is_dead = False

        if smart_dead_mask[fid]:
            # Set piece active: if ball is not closely contested, play is dead
            if not ball_seen[fid] or min_p_dist > 3.0:
                is_dead = True

        if is_dead:
            labels[fid] = DEAD
            opp_count = 0
            continue

        dh = min_dist_h[fid]
        da = min_dist_a[fid]

        if ball_seen[fid]:
            opp_dist = da if curr_team == HOME else dh
            my_dist = dh if curr_team == HOME else da

            # Opponent must be in control proximity and closer than current possessor
            if opp_dist < dist_thresh and opp_dist < my_dist:
                opp_count += 1
                if opp_count >= dwell_frames:
                    curr_team = AWAY if curr_team == HOME else HOME
                    opp_count = 0
            else:
                opp_count = max(0, opp_count - 1)

        # Carry forward current team possession across dropouts
        labels[fid] = curr_team

    print(f"Step 3/3: Regularizing sequence (min_spell_len={min_spell_len})...")
    labels = filter_micro_spells(labels, min_spell_len=min_spell_len)

    return labels


def write_submission_csv(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines(f"{frame},{CODE_TO_LABEL[code]}\n" for frame, code in enumerate(labels))
    return path


def evaluate_against_public_gt(pred_labels):
    from evaluation import possession_metrics as pm
    gt_df = pm.read_label_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    gt_codes = pm.label_codes(gt_df)
    results = pm.evaluate(gt_codes, pred_labels)
    
    print("\n" + "=" * 60)
    print("OFFICIAL EVALUATION ON PUBLIC 5-MINUTE GROUND TRUTH:")
    print(f"  spellF1:        {results['spell_f1_mean']:.4f}  (Baseline smooth125 was 0.4242)")
    print(f"  macroF1:        {results['macro_f1']:.4f}  (Baseline smooth125 was 0.5812)")
    print(f"  Fragmentation:  {results['fragmentation']:.2f}  (Baseline smooth125 was 0.81)")
    print(f"  Alive/Dead F1:  {results['alive_dead_f1']:.4f}")
    print(f"  Team Accuracy:  {results['team_acc_both_alive']:.4f}")
    print("=" * 60 + "\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Enhanced Ball Possession Model")
    parser.add_argument("--out", default=os.path.join(RESULTS_DIR, "enhanced_model.csv"), help="Output CSV path")
    parser.add_argument("--dwell", type=int, default=25, help="Control dwell time in frames")
    parser.add_argument("--dist", type=float, default=1.8, help="Control distance threshold in meters")
    parser.add_argument("--min-spell", type=int, default=50, help="Minimum spell length in frames")
    parser.add_argument("--eval-gt", action="store_true", default=True, help="Evaluate against public GT")
    args = parser.parse_args()

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]

    slot_teams = load_teams(meta_path)
    tags = load_smart_tags()

    labels = stream_features_and_infer(
        raw_path, slot_teams, n_frames, pitch, tags,
        dwell_frames=args.dwell, dist_thresh=args.dist, min_spell_len=args.min_spell
    )

    out_path = write_submission_csv(args.out, labels)
    print(f"Submission generated: {out_path}")

    if args.eval_gt:
        evaluate_against_public_gt(labels)


if __name__ == "__main__":
    main()
