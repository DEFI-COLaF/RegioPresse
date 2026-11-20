#!/usr/bin/env python3
"""Mesure du temps de calcul des méthodes d'identification de langue sur un corpus TEI.

Pour que la comparaison soit propre, chaque méthode est chronométrée SEULE, sur exactement les
mêmes unités (div, bloc, phrase/vers), avec les mêmes textes que ceux de tag_lid.py /
evaluate_lid.py :
  swadesh (et chaque --extra-lexicon)  S.score_text sur les segments de l'unité
  zabanshenas                          un appel du modèle par unité (--zaban-chunk comme à l'étiquetage)
  glotlid / filtre français            un appel par phrase (c'est le niveau où le filtre travaille)

Trois choses sont séparées, parce qu'elles ne se comparent pas de la même façon :
  1. le CHARGEMENT (lexique, modèles) : payé une seule fois par exécution ;
  2. la LECTURE / DÉCOUPAGE du XML : commune à toutes les méthodes ;
  3. le TRAITEMENT par unité : ce qui augmente avec la taille du corpus.
Le résumé donne aussi une estimation du temps d'étiquetage d'un corpus pour les niveaux choisis
(--levels), et le rapport entre méthodes.

Exemple :
    python benchmark_lid.py lexique.tsv corpus_tei/ --zabanshenas --zaban-chunk 60 \\
        --fr-filter glotlid --glotlid-model cis-lmu/glotlid --repeat 3 --sample 300 -o bench
"""
import argparse
import csv
import os
import platform
import random
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swadesh_lid_xml as S  # noqa: E402

LEVELS = ("div", "bloc", "phrase")
BUCKETS = [(1, 4, "1-4"), (5, 9, "5-9"), (10, 19, "10-19"), (20, 49, "20-49"),
           (50, 99, "50-99"), (100, 10 ** 9, "100+")]


def pct(sorted_vals, q):
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def is_tei(f):
    try:
        for _ev, el in ET.iterparse(f, events=("start",)):
            return S.local(el.tag) == "TEI"
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return False
    return False


def env_info():
    info = [f"Python {platform.python_version()}", platform.platform(), f"{os.cpu_count()} CPU logiques"]
    try:
        import torch
        dev = "GPU (" + torch.cuda.get_device_name(0) + ")" if torch.cuda.is_available() else "CPU"
        info.append(f"torch {torch.__version__}, calcul du modèle sur {dev}, {torch.get_num_threads()} threads")
    except Exception:
        info.append("torch non détecté")
    return " | ".join(info)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lexicon")
    ap.add_argument("xml", help="fichier XML-TEI ou dossier (tous les sous-dossiers)")
    ap.add_argument("-o", "--output", default="benchmark", help="préfixe des fichiers de sortie")
    ap.add_argument("--extra-lexicon", action="append", default=[], metavar="NOM=CHEMIN")
    ap.add_argument("--zabanshenas", action="store_true")
    ap.add_argument("--zaban-max-length", type=int)
    ap.add_argument("--zaban-chunk", type=int, default=0)
    ap.add_argument("--fr-filter", choices=["glotlid", "zabanshenas"])
    ap.add_argument("--fr-threshold", type=float, default=0.8)
    ap.add_argument("--fr-codes", default="")
    ap.add_argument("--glotlid-model")
    ap.add_argument("--repeat", type=int, default=1,
                    help="nombre de passages complets ; on garde le meilleur temps de chaque unité "
                         "(réduit le bruit dû à la machine)")
    ap.add_argument("--sample", type=int, default=0,
                    help="ne chronométrer que N unités tirées au hasard par niveau (0 = toutes) ; "
                         "utile avec Zabanshenas sur un gros corpus")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--levels", default="div,bloc",
                    help="niveaux inclus dans l'estimation du temps d'étiquetage (défaut : div,bloc)")
    ap.add_argument("--warmup", type=int, default=3, help="appels d'échauffement avant chronométrage")
    ap.add_argument("--ignore-cols", default="n°")
    ap.add_argument("--french-col")
    ap.add_argument("--variant-sep", default="/;,")
    ap.add_argument("--block-tags", default=S.DEFAULT_BLOCK_TAGS)
    ap.add_argument("--exclude-tags", default="")
    args = ap.parse_args()

    levels_est = [x.strip() for x in args.levels.split(",") if x.strip()]
    print("Environnement :", env_info(), file=sys.stderr)

    # ---------------------------------------------------------------- 1. chargements
    loads = []  # (nom, secondes)
    ignore_cols = [c for c in args.ignore_cols.split(",") if c.strip()]
    lexes = {}

    def load_lex(name, path):
        t = time.perf_counter()
        lx, _ = S.load_lexicon(path, args.french_col, args.variant_sep, None, False, ignore_cols, None, False)
        vc, mx = S.build_vocab(lx)
        loads.append((f"liste de mots « {name} »", time.perf_counter() - t))
        lexes[name] = (lx, vc, mx)

    load_lex("swadesh", args.lexicon)
    for spec in args.extra_lexicon:
        name, sep, path = spec.partition("=")
        if not sep:
            path, name = spec, Path(spec).stem
        load_lex(name.strip(), path)

    zab_detect = None
    if args.zabanshenas or args.fr_filter == "zabanshenas":
        t = time.perf_counter()
        zab_detect = S.make_detector(args.zaban_max_length, args.zaban_chunk)
        loads.append(("modèle Zabanshenas", time.perf_counter() - t))
    fr_detect = None
    if args.fr_filter:
        t = time.perf_counter()
        fr_detect = S.make_fr_detector(args.fr_filter, args.fr_threshold, args.fr_codes,
                                       args.glotlid_model, zab_detect)
        loads.append((f"filtre français ({args.fr_filter})", time.perf_counter() - t))

    # ---------------------------------------------------------------- 2. lecture du XML
    p = Path(args.xml)
    files = [f for f in sorted(p.rglob("*.xml")) if f.is_file()] if p.is_dir() else [p]
    files = [f for f in files if is_tei(f)]
    if not files:
        sys.exit("Aucun fichier TEI trouvé.")
    bt = {t.strip() for t in args.block_tags.split(",") if t.strip()}
    ex = {t.strip() for t in args.exclude_tags.split(",") if t.strip()}
    t = time.perf_counter()
    all_divs = [S.parse_xml(str(f), bt, ex) for f in files]
    t_parse = time.perf_counter() - t

    units = {lv: [] for lv in LEVELS}

    def all_segments(d):
        segs = [x for b in d["blocks"] for x in b["_segments"]]
        for sub in d["_sub"]:
            segs += all_segments(sub)
        return segs

    for fi, divs in enumerate(all_divs):
        for d in divs:
            segs = all_segments(d)
            if segs:
                units["div"].append({"id": f"{fi}:{d['id']}", "segments": segs, "text": " ".join(segs)})
            for b in d["blocks"]:
                units["bloc"].append({"id": f"{fi}:{b['id']}", "segments": b["_segments"], "text": b["text"]})
                for u in b["units"]:
                    units["phrase"].append({"id": f"{fi}:{u['id']}", "segments": [u["text"]], "text": u["text"]})
    rnd = random.Random(args.seed)
    for lv in LEVELS:
        for u in units[lv]:
            u["formes"] = S.size_of(u["segments"], True)
            u["tokens"] = S.size_of(u["segments"], False)
        if args.sample and len(units[lv]) > args.sample:
            units[lv] = rnd.sample(units[lv], args.sample)
    sampled = bool(args.sample)

    # ---------------------------------------------------------------- 3. méthodes à chronométrer
    methods = []  # (nom, fonction(unité), niveaux concernés)
    for name, (lx, vc, mx) in lexes.items():
        methods.append((name, (lambda u, lx=lx, vc=vc, mx=mx: S.score_text(u["segments"], lx, vc, mx, 0, False)),
                        LEVELS))
    if args.zabanshenas:
        methods.append(("zabanshenas", lambda u: zab_detect(u["text"]), LEVELS))
    if fr_detect:
        methods.append((f"filtre-{args.fr_filter}", lambda u: fr_detect(u["text"]), ("phrase",)))

    # échauffement (premier appel d'un modèle = souvent bien plus lent : chargement paresseux, caches)
    first_call = {}
    sample_unit = (units["phrase"] or units["bloc"] or units["div"])[0]
    for name, fn, _lv in methods:
        t = time.perf_counter()
        fn(sample_unit)
        first_call[name] = time.perf_counter() - t
        for _ in range(max(0, args.warmup - 1)):
            fn(sample_unit)

    # ---------------------------------------------------------------- 4. chronométrage
    best = {}   # (méthode, niveau) -> liste des durées (s) par unité (meilleur passage)
    totals = {}  # (méthode, niveau) -> liste des totaux par passage
    for name, fn, lvs in methods:
        for lv in lvs:
            per_pass = []
            for _rep in range(max(1, args.repeat)):
                durs = []
                for u in units[lv]:
                    t0 = time.perf_counter()
                    fn(u)
                    durs.append(time.perf_counter() - t0)
                per_pass.append(durs)
            totals[(name, lv)] = [sum(d) for d in per_pass]
            best[(name, lv)] = [min(vals) for vals in zip(*per_pass)]
        print(f"  chronométré : {name}", file=sys.stderr)

    # ---------------------------------------------------------------- 5. rapports
    n_formes_lv = {lv: sum(u["formes"] for u in units[lv]) for lv in LEVELS}
    print("\n=== Chargement (une seule fois par exécution) ===")
    for nm, sec in loads:
        print(f"  {nm:40}{sec:10.2f} s")
    print(f"  {'lecture et découpage du XML (' + str(len(files)) + ' fichiers)':40}{t_parse:10.2f} s   (commun à toutes les méthodes)")
    print("\n=== Premier appel de chaque méthode (échauffement) ===")
    for nm, sec in first_call.items():
        print(f"  {nm:40}{sec * 1000:10.1f} ms")

    print(f"\n=== Traitement par unité{' (échantillon de ' + str(args.sample) + ' unités par niveau)' if sampled else ''} ===")
    print(f"{'méthode':26}{'niveau':8}{'unités':>8}{'total (s)':>11}{'moy. ms':>10}{'méd. ms':>10}"
          f"{'p95 ms':>10}{'formes/s':>11}")
    rows = []
    for name, fn, lvs in methods:
        for lv in lvs:
            d = best[(name, lv)]
            sd = sorted(d)
            tot = min(totals[(name, lv)])
            fps = n_formes_lv[lv] / tot if tot else None
            rows.append([name, lv, len(d), round(tot, 4), round(1000 * statistics.fmean(d), 4),
                         round(1000 * pct(sd, 0.5), 4), round(1000 * pct(sd, 0.95), 4),
                         round(fps, 1) if fps else ""])
            print(f"{name:26}{lv:8}{len(d):>8}{tot:11.3f}{1000 * statistics.fmean(d):10.3f}"
                  f"{1000 * pct(sd, 0.5):10.3f}{1000 * pct(sd, 0.95):10.3f}{fps or 0:11.0f}")
    if args.repeat > 1:
        print(f"  (meilleur de {args.repeat} passages ; variabilité des totaux entre passages : "
              + ", ".join(f"{n}/{lv} ±{(max(v) - min(v)) / min(v) * 100:.0f}%"
                          for (n, lv), v in totals.items() if min(v) > 0)[:300] + ")")

    # temps selon la taille
    print("\n=== Temps moyen par unité selon la taille (formes distinctes), tous niveaux confondus ===")
    print(f"{'méthode':26}" + "".join(f"{lab:>14}" for _, _, lab in BUCKETS))
    bucket_rows = []
    for name, fn, lvs in methods:
        pooled = []
        for lv in lvs:
            pooled += [(u["formes"], dur) for u, dur in zip(units[lv], best[(name, lv)])]
        cells = []
        for lo, hi, lab in BUCKETS:
            vals = [dur for n, dur in pooled if lo <= n <= hi]
            m = 1000 * statistics.fmean(vals) if vals else None
            bucket_rows.append([name, lab, len(vals), round(m, 4) if m is not None else ""])
            cells.append(f"{m:.3f}ms({len(vals)})" if m is not None else "-")
        print(f"{name:26}" + "".join(f"{c:>14}" for c in cells))
    print("  (moyenne en ms, nombre d'unités entre parenthèses)")

    # estimation de l'étiquetage et rapports
    full_units = {lv: len(units[lv]) for lv in LEVELS}
    print(f"\n=== Estimation du traitement pour étiqueter ces fichiers (niveaux : {', '.join(levels_est)}) ===")
    est = {}
    for name, fn, lvs in methods:
        use = [lv for lv in levels_est if lv in lvs]
        if args.fr_filter and name.startswith("filtre-"):
            use = ["phrase"]  # le filtre travaille toujours au niveau phrase
        est[name] = sum(min(totals[(name, lv)]) for lv in use)
        print(f"  {name:26}{est[name]:10.2f} s   ({'; '.join(use) or '-'})")
    base = est.get("swadesh")
    if base:
        for name, v in est.items():
            if name != "swadesh" and v:
                print(f"  rapport {name} / swadesh : x{v / base:.1f}")
    if sampled:
        print("  /!\\ Les temps portent sur un échantillon : multiplie par (unités du corpus / unités chronométrées) "
              "pour extrapoler.")
    print("\nÀ retenir pour lire ces chiffres : le chargement et le traitement ne s'additionnent pas de la même "
          "façon ; un modèle lent à charger mais rapide par unité devient avantageux sur un gros corpus. "
          "Les temps de Zabanshenas dépendent beaucoup du matériel (CPU/GPU) et du fait qu'on appelle le modèle "
          "une unité à la fois (pas de traitement par lots).")

    # ---------------------------------------------------------------- 6. fichiers
    out = args.output
    with open(out + "_resume.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["methode", "niveau", "unites", "total_s", "moyenne_ms", "mediane_ms", "p95_ms", "formes_par_s"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["methode", "taille_formes", "unites", "moyenne_ms"])
        w.writerows(bucket_rows)
        w.writerow([])
        w.writerow(["chargement", "secondes"])
        w.writerows([[n, round(sec, 4)] for n, sec in loads] + [["lecture_xml", round(t_parse, 4)]])
        w.writerow([])
        w.writerow(["methode", "premier_appel_ms", "estimation_etiquetage_s"])
        w.writerows([[n, round(first_call[n] * 1000, 2), round(est.get(n, 0), 3)] for n in first_call])
        w.writerow([])
        w.writerow(["environnement", env_info()])
    with open(out + "_unites.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["methode", "niveau", "id", "tokens", "formes", "ms"])
        for name, fn, lvs in methods:
            for lv in lvs:
                for u, dur in zip(units[lv], best[(name, lv)]):
                    w.writerow([name, lv, u["id"], u["tokens"], u["formes"], round(1000 * dur, 5)])
    print(f"\nFichiers : {out}_resume.tsv, {out}_unites.tsv", file=sys.stderr)


if __name__ == "__main__":
    main()
