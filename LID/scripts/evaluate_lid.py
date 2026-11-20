#!/usr/bin/env python3
"""Évaluation de l'identification de langue (Swadesh / Zabanshenas / filtre français)
sur des textes XML-TEI annotés par xml:lang.

Le premier argument XML peut être un dossier : tous les .xml qu'il contient sont évalués et les
résultats agrégés (détail par fichier dans <sortie>_fichiers.tsv).

La langue de référence (« gold ») d'une unité est la langue majoritaire (en tokens) de
son texte, d'après xml:lang : hérité de <TEI>/<text>/<body>, puis des <div>, puis de tout
élément plus fin qui le surcharge (<p>, <ab>, <note>, ou un élément en ligne) ; sans xml:lang
hérité, --default-lang donne la langue par défaut. Les trois niveaux sont évalués : div, bloc (p, lg, l, head,
note...), phrase (vers <l> ou phrase de prose), avec les mêmes découpages que
swadesh_lid_xml.py (qui doit se trouver dans le même dossier).

Systèmes comparés :
  swadesh      : liste de Swadesh seule (ne peut jamais répondre « français »)
  swadesh+fr   : filtre français (--fr-filter) puis Swadesh ; une unité jugée française
                 est étiquetée « français »
  zabanshenas  : modèle Zabanshenas (--zabanshenas) ; ses codes sont traduits en langues
                 de référence par --zaban-map (une langue sans classe ne peut pas être
                 trouvée : c'est un résultat, pas un bogue)

Passages dont la référence est le français : par défaut (--french-gold detection-only) ils ne
sont PAS scorés par les systèmes d'identification (Swadesh, swadesh+fr, Zabanshenas) ni
comptés dans leurs tableaux ; ils servent seulement à évaluer la détection du français.
--french-gold skip les écarte complètement ; --french-gold score les évalue comme les autres.

Autres listes de mots : --extra-lexicon NOM=chemin.tsv (répétable ; même format que le lexique
principal) ajoute, pour chaque liste, les systèmes NOM et NOM+fr, évalués exactement comme
« swadesh » et « swadesh+fr » (mêmes unités, mêmes découpages, même filtre français).

Mesures classiques par langue (précision, rappel, F1, moyennes macro / pondérée / micro, exactitude),
en unités et en tokens : dans la console, <sortie>_resume.tsv et <sortie>_classes.tsv.

--count-types : chaque forme n'est comptée qu'une fois par unité ; les tailles, seuils (--min-tokens) et
poids passent des tokens aux formes distinctes (colonnes *_formes). Le score Swadesh lui-même
compte déjà chaque item une seule fois, il ne change pas.

Fichier _confusion.tsv : pour chaque langue de référence, ce que le système a répondu. Les abstentions
sont détaillées (abstention:trop_court / aucune_info / ex_aequo). part_de_ref_* = part de cette réponse
parmi les unités (ou les formes/tokens) de la langue de référence (la diagonale est le rappel) ;
part_du_predit_* = part de la langue de référence parmi tout ce que le système a répondu pareil
(la diagonale est la précision).

Quand un code Zabanshenas couvre plusieurs langues de référence (ex. oci pour gascon, languedocien et
provençal dans --zaban-map), Zabanshenas est crédité si la référence est l'une d'elles. Pour une
comparaison équitable avec les listes de mots (qui doivent choisir entre ces langues), regroupe-les avec
--merge-labels "occitan=gasc-1240|lang-1309|prov-1235" : tout est alors évalué au niveau du groupe.

Un système « s'abstient » quand il ne répond pas (texte trop court, aucun item trouvé,
ex aequo). Mesures : couverture (part d'unités avec réponse), précision (exactes parmi les
réponses), exactitude (exactes parmi toutes les unités), exactitude pondérée par les tokens
(part du TEXTE correctement identifiée : la mesure la plus comparable entre niveaux).

Exemple :
    python evaluate_lid.py lexique.tsv virgile.xml \\
        --gold-map "bour-1247=bourguignon,stan-1290=français" \\
        --fr-filter glotlid --zabanshenas --zaban-map "bourguignon=frp|oci" \\
        --min-tokens 4 -o evaluation
"""
import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swadesh_lid_xml as S  # noqa: E402

import inspect  # noqa: E402
if "default_lang" not in inspect.signature(S.parse_xml).parameters:
    sys.exit(f"{S.__file__} est une ancienne version de swadesh_lid_xml.py (elle ne lit pas "
             "le xml:lang du <body> ni --default-lang). Remplace-la par la dernière version, "
             "dans le même dossier que evaluate_lid.py.")

FR_NAMES = {"français", "francais", "french", "fra", "fre", "fr"}
FR = "français"
BUCKETS = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 9, "6-9"), (10, 19, "10-19"),
           (20, 49, "20-49"), (50, 10 ** 9, "50+")]
LEVELS = ("div", "bloc", "phrase")


def is_fr(label):
    """Vrai pour « français », « fr », « fra », « fre », « french », « fr-FR »..."""
    if label is None:
        return False
    n = S.norm(label)
    return n in FR_NAMES or n.split("-")[0] in FR_NAMES


def header_names(path):
    """{ident: nom} d'après <langUsage><language ident=...><name>."""
    names = {}
    for e in ET.parse(path).getroot().iter():
        if S.local(e.tag) == "language" and e.get("ident"):
            nm = next((c.text for c in e if S.local(c.tag) == "name" and c.text), None)
            names[e.get("ident")] = nm
    return names


def parse_gold_map(spec):
    """'xml:lang=langue de référence,...' ; une clé répétée écrase la précédente (avertissement)."""
    out = {}
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in out and out[k] != v:
                print(f"/!\\ --gold-map : la valeur de xml:lang « {k} » est donnée deux fois "
                      f"(« {out[k]} » puis « {v} ») ; seule la dernière (« {v} ») est retenue. "
                      f"Rappel : le format est xml:lang=langue_de_référence.", file=sys.stderr)
            out[k] = v
    return out


def resolve_gold_src(raw, gold_map, header, lex_cols):
    """(langue de référence, origine de l'étiquette) pour une valeur de xml:lang.

    Ordre : 1) --gold-map ; 2) nom déclaré dans <teiHeader>/<langUsage><language ident=...>,
    rapproché d'une colonne du lexique si possible ; 3) la valeur brute de xml:lang.
    """
    if raw is None:
        return None, "aucun xml:lang"
    if raw in gold_map:
        return (FR if is_fr(gold_map[raw]) else gold_map[raw]), "--gold-map"
    name = header.get(raw)
    if name:
        n = S.norm(name)
        if n in FR_NAMES:
            return FR, f"<langUsage> « {name} »"
        for col in lex_cols:
            c = S.norm_col(col)
            if len(c) >= 4 and (c == n or n.startswith(c)):
                return col, f"<langUsage> « {name} » rapproché de la colonne « {col} »"
        return name, f"<langUsage> « {name} » (aucune colonne du lexique ne correspond)"
    return (FR if is_fr(raw) else raw), "valeur brute de xml:lang (ni --gold-map ni <langUsage>)"


def resolve_gold(raw, gold_map, header, lex_cols):
    return resolve_gold_src(raw, gold_map, header, lex_cols)[0]


def pred_swadesh(res):
    """(étiquette ou None, motif d'abstention)."""
    if res.get("trop_court"):
        return None, "trop_court"
    if res.get("exclu_fr"):
        return FR, ""
    if res["best"] is None:
        return None, "aucune_info"
    if "ties" in res:
        return None, "ex_aequo"
    return res["best"], ""


def pred_zaban(res, code2labels):
    """(liste des langues de référence compatibles avec le code prédit, motif d'abstention).
    Un même code peut correspondre à plusieurs langues (ex. oci -> gascon, languedocien, provençal)."""
    z = res.get("zabanshenas")
    if not z or z.get("ecarte"):
        return None, "trop_court"
    code = z.get("code")
    if not code:
        return None, "aucune_info"
    return code2labels.get(code) or ["autre:" + str(code)], ""


def parse_merge_labels(spec):
    """'occitan=gasc-1240|lang-1309|prov-1235,autre=a|b' -> ({membre: groupe}, {groupe: [membres]})."""
    m2g, groups = {}, {}
    for part in (spec or "").split(","):
        if "=" not in part:
            continue
        g, members = part.split("=", 1)
        g = g.strip()
        for m in members.split("|"):
            m = m.strip()
            if not m:
                continue
            if m in m2g and m2g[m] != g:
                sys.exit(f"--merge-labels : « {m} » est dans deux groupes ({m2g[m]} et {g}).")
            m2g[m] = g
            groups.setdefault(g, []).append(m)
    return m2g, groups


def metrics(items):
    """items : liste de dicts {gold, pred, ntok}."""
    n = len(items)
    ans = [i for i in items if i["pred"] is not None]
    ok = [i for i in ans if i["pred"] == i["gold"]]
    tok_all = sum(i["ntok"] for i in items)
    tok_ok = sum(i["ntok"] for i in ok)
    return {
        "n": n, "repondus": len(ans),
        "couverture": len(ans) / n if n else None,
        "precision": len(ok) / len(ans) if ans else None,
        "exactitude": len(ok) / n if n else None,
        "exactitude_tokens": tok_ok / tok_all if tok_all else None,
    }


def binary(items):
    """items : {gold, pred_fr} -> précision/rappel du français et taux d'exclusion à tort."""
    tp = sum(1 for i in items if is_fr(i["gold"]) and i["pred_fr"])
    fn = sum(1 for i in items if is_fr(i["gold"]) and not i["pred_fr"])
    fp = sum(1 for i in items if not is_fr(i["gold"]) and i["pred_fr"])
    tn = sum(1 for i in items if not is_fr(i["gold"]) and not i["pred_fr"])
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    return {"n_fr": tp + fn, "n_non_fr": fp + tn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision_fr": p, "rappel_fr": r, "f1_fr": f1,
            "exclusion_a_tort": fp / (fp + tn) if fp + tn else None}


def per_class(items, weighted=False):
    """Mesures classiques par langue de référence.

    TP = gold c et prédit c ; FP = gold autre et prédit c ; FN = gold c et pas prédit c (erreur OU
    abstention). Précision = TP/(TP+FP) ; rappel = TP/(TP+FN) ; F1 = 2TP/(2TP+FP+FN).
    weighted=True pondère chaque unité par son nombre de tokens (part du TEXTE, comparable entre
    niveaux). Renvoie (lignes par langue, agrégats macro / pondéré / micro, exactitude).
    """
    w = (lambda i: i["ntok"]) if weighted else (lambda i: 1)
    classes = sorted({i["gold"] for i in items} | {i["pred"] for i in items if i["pred"] is not None})
    rows = {}
    for c in classes:
        tp = sum(w(i) for i in items if i["gold"] == c and i["pred"] == c)
        fp = sum(w(i) for i in items if i["gold"] != c and i["pred"] == c)
        fn = sum(w(i) for i in items if i["gold"] == c and i["pred"] != c)
        sup = sum(w(i) for i in items if i["gold"] == c)
        rows[c] = {"support": sup, "tp": tp, "fp": fp, "fn": fn,
                   "precision": tp / (tp + fp) if tp + fp else None,
                   "rappel": tp / (tp + fn) if tp + fn else None,
                   "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
    gold_classes = [c for c in classes if rows[c]["support"] > 0]
    agg = {}
    if gold_classes:
        def mean(key, weights=None):
            vals = [(rows[c][key] or 0.0) for c in gold_classes]  # précision non définie -> 0
            if weights is None:
                return sum(vals) / len(vals)
            tot = sum(weights)
            return sum(v * x for v, x in zip(vals, weights)) / tot if tot else None
        sups = [rows[c]["support"] for c in gold_classes]
        agg["macro"] = {k: mean(k) for k in ("precision", "rappel", "f1")}
        agg["pondere"] = {k: mean(k, sups) for k in ("precision", "rappel", "f1")}
    tp_all = sum(r["tp"] for r in rows.values())
    answered = sum(w(i) for i in items if i["pred"] is not None)
    total = sum(w(i) for i in items)
    mp = tp_all / answered if answered else None
    mr = tp_all / total if total else None
    agg["micro"] = {"precision": mp, "rappel": mr,
                    "f1": 2 * mp * mr / (mp + mr) if mp and mr else (0.0 if mp is not None and mr is not None else None)}
    return rows, agg, (tp_all / total if total else None)


def fmt(x, pct=True):
    if x is None:
        return "  n/a"
    return f"{100 * x:5.1f}%" if pct else f"{x}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lexicon")
    ap.add_argument("xml", help="fichier XML-TEI annoté, ou dossier de .xml")
    ap.add_argument("-o", "--output", default="evaluation", help="préfixe des fichiers de sortie")
    ap.add_argument("--gold-map", default="",
                    help="xml:lang -> langue de référence (nom de colonne du lexique, ou « français »), "
                         "ex. 'bour-1247=bourguignon,stan-1290=français'. Par défaut : noms de "
                         "<langUsage> rapprochés des colonnes du lexique")
    ap.add_argument("--default-lang",
                    help="xml:lang à utiliser quand aucun xml:lang n'est hérité (ex. cors-1241) ; "
                         "sans cela, ces unités sont écartées de l'évaluation et comptées")
    ap.add_argument("--extra-lexicon", action="append", default=[], metavar="NOM=CHEMIN",
                    help="autre liste de mots (même format que le lexique principal) à évaluer en "
                         "plus ; répétable. Les noms de colonnes (langues) doivent être les mêmes "
                         "que dans le lexique principal. Ex. --extra-lexicon swadesh207=../data/autre.tsv")
    ap.add_argument("--merge-labels", default="",
                    help="regroupe des langues avant le calcul des mesures, ex. "
                         "'occitan=gasc-1240|lang-1309|prov-1235'. La référence et les prédictions de "
                         "tous les systèmes sont ramenées au groupe : comparaison au niveau de la "
                         "« macro-langue », utile quand Zabanshenas n'a qu'une classe (oci) pour plusieurs "
                         "variétés que la liste de mots distingue")
    ap.add_argument("--zabanshenas", action="store_true", help="évaluer aussi Zabanshenas")
    ap.add_argument("--zaban-map", default="",
                    help="langue de référence -> codes Zabanshenas, ex. 'bourguignon=frp|oci' "
                         "(« français » vaut fra|fr par défaut)")
    ap.add_argument("--zaban-max-length", type=int)
    ap.add_argument("--zaban-chunk", type=int, default=0,
                    help="applique Zabanshenas par fenêtres de N mots (vote pondéré) au lieu d'un seul "
                         "appel sur tout le texte (évite la troncature sur les div et blocs longs) ; "
                         "0 = désactivé (défaut)")
    ap.add_argument("--fr-filter", choices=["glotlid", "zabanshenas"],
                    help="évaluer aussi le pipeline filtre français + Swadesh")
    ap.add_argument("--fr-threshold", type=float, default=0.8)
    ap.add_argument("--fr-codes", default="")
    ap.add_argument("--glotlid-model")
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--count-types", action="store_true",
                    help="chaque forme n'est comptée qu'une fois par unité : --min-tokens, les poids "
                         "(« exactitude pondérée », mesures en tokens) et les tranches de longueur portent "
                         "sur le nombre de FORMES DISTINCTES de l'unité, pas sur son nombre de tokens")
    ap.add_argument("--french-gold", choices=["detection-only", "skip", "score"],
                    default="detection-only",
                    help="traitement des unités dont la référence est le français : "
                         "detection-only (défaut) = hors des tableaux d'identification, utilisées "
                         "seulement pour la détection du français ; skip = ignorées (Zabanshenas n'est "
                         "pas appelé sur elles, pas d'évaluation du français) ; score = scorées "
                         "comme les autres")
    ap.add_argument("--min-purity", type=float, default=0.0,
                    help="ignore les unités dont la langue majoritaire couvre moins de cette part "
                         "des tokens (0 = tout garder)")
    ap.add_argument("--ignore-cols", default="n°")
    ap.add_argument("--french-col")
    ap.add_argument("--variant-sep", default="/;,")
    ap.add_argument("--french-forms")
    ap.add_argument("--drop-identical-fr", action="store_true")
    ap.add_argument("--block-tags", default=S.DEFAULT_BLOCK_TAGS)
    ap.add_argument("--exclude-tags", default="")
    args = ap.parse_args()
    U = "formes" if args.count_types else "tokens"  # unité de taille et de pondération

    ignore_cols = [c for c in args.ignore_cols.split(",") if c.strip()]
    french_forms = S.load_french_forms(args.french_forms) if args.french_forms else None
    lex, _ = S.load_lexicon(args.lexicon, args.french_col, args.variant_sep, french_forms,
                            False, ignore_cols, None, args.drop_identical_fr)
    vocab, maxn = S.build_vocab(lex)
    lex_cols = list(lex)

    # --- Zabanshenas : correspondance code -> langue de référence
    zmap = S.parse_zaban_map(args.zaban_map)
    zmap.setdefault(FR, {"fra", "fr"})
    code2labels = {}
    for lab, codes in zmap.items():
        for c in codes:
            code2labels.setdefault(c, []).append(lab)
    member2group, groups = parse_merge_labels(args.merge_labels)
    if groups:
        print("Groupes de langues (--merge-labels) : " + "; ".join(f"{g} = {', '.join(m)}"
              for g, m in groups.items()) + ". Référence ET prédictions de tous les systèmes sont "
              "ramenées à ces groupes avant le calcul des mesures.", file=sys.stderr)
    for c, labs in code2labels.items():
        if len(labs) > 1:
            same = len({member2group.get(l, l) for l in labs}) == 1
            print(f"Zabanshenas : le code « {c} » correspond à {len(labs)} langues ({', '.join(labs)}). "
                  + ("Elles sont dans le même groupe (--merge-labels)." if same else
                     "Zabanshenas est crédité quand la référence est l'une d'elles (évaluation indulgente) ; "
                     "pour comparer équitablement avec les listes de mots, qui doivent, elles, choisir "
                     "entre ces langues, regroupe-les avec --merge-labels."), file=sys.stderr)
    zab_detect = S.make_detector(args.zaban_max_length, args.zaban_chunk) if (args.zabanshenas or args.fr_filter == "zabanshenas") else None
    fr_detect = (S.make_fr_detector(args.fr_filter, args.fr_threshold, args.fr_codes,
                                    args.glotlid_model, zab_detect) if args.fr_filter else None)

    extras = []  # (nom, lexique, vocabulaire, longueur max)
    for spec in args.extra_lexicon:
        name, sep, path = spec.partition("=")
        if not sep:
            path, name = spec, Path(spec).stem
        name = name.strip()
        if name in ("swadesh", "zabanshenas") or any(name == e[0] for e in extras):
            sys.exit(f"Nom de liste déjà utilisé ou réservé : {name!r} (utilise NOM=CHEMIN)")
        lx, _ = S.load_lexicon(path, args.french_col, args.variant_sep, french_forms,
                               False, ignore_cols, None, args.drop_identical_fr)
        vc, mx = S.build_vocab(lx)
        extras.append((name, lx, vc, mx))
    if fr_detect and extras:  # le filtre est appelé une fois par texte, pas une fois par liste
        _raw, _cache = fr_detect, {}

        def fr_detect(text):
            if text not in _cache:
                _cache[text] = _raw(text)
            return _cache[text]

    p = Path(args.xml)
    files = [f for f in sorted(p.rglob("*.xml")) if f.is_file()] if p.is_dir() else [p]
    relname = (lambda f: str(f.relative_to(p))) if p.is_dir() else (lambda f: f.name)

    def is_tei(f):
        try:
            for _ev, el in ET.iterparse(f, events=("start",)):
                return S.local(el.tag) == "TEI"
        except (ET.ParseError, OSError, UnicodeDecodeError):
            return False
        return False

    if p.is_dir():  # sous-dossiers de tout niveau ; XML non TEI ou illisibles ignorés
        n_all = len(files)
        files = [f for f in files if is_tei(f)]
        print(f"{n_all} fichier(s) XML trouvé(s), {len(files)} TEI retenu(s).", file=sys.stderr)
    gold_map = parse_gold_map(args.gold_map)

    units = []
    gold_tokens = Counter()
    nolang = Counter()
    raw_tokens = Counter()
    for f in files:
        header = header_names(f)
        ns = SimpleNamespace(texts=str(f), block_tags=args.block_tags,
                             exclude_tags=args.exclude_tags, min_tokens=args.min_tokens,
                             default_lang=args.default_lang, count_types=args.count_types)
        cmp_detect = zab_detect if args.zabanshenas else None
        if cmp_detect and args.french_gold == "skip":
            skip_texts = set()
            for dd in S.parse_xml(str(f), {t.strip() for t in args.block_tags.split(",") if t.strip()},
                                  {t.strip() for t in args.exclude_tags.split(",") if t.strip()},
                                  args.default_lang):
                for bb in dd["blocks"]:
                    if is_fr(resolve_gold(bb["lang"], gold_map, header, lex_cols)):
                        skip_texts |= {bb["text"], " ".join(bb["_segments"])}
                    for uu in bb["units"]:
                        if is_fr(resolve_gold(uu["lang"], gold_map, header, lex_cols)):
                            skip_texts.add(uu["text"])

            def cmp_detect(text, _d=zab_detect, _skip=skip_texts):
                if text in _skip:
                    return {"code": None, "language": None, "score": None, "ecarte": True}
                return _d(text)
        _, rows_a = S.run_xml(ns, lex, vocab, maxn, cmp_detect, None)
        rows_b = S.run_xml(ns, lex, vocab, maxn, None, fr_detect)[1] if fr_detect else None
        if rows_b is not None and len(rows_b) != len(rows_a):
            sys.exit("Incohérence interne entre les deux passes d'analyse.")
        rows_x = {}
        for name, lx, vc, mx in extras:
            ra = S.run_xml(ns, lx, vc, mx, None, None)[1]
            rb = S.run_xml(ns, lx, vc, mx, None, fr_detect)[1] if fr_detect else None
            if len(ra) != len(rows_a) or (rb is not None and len(rb) != len(rows_a)):
                sys.exit("Incohérence interne entre les passes d'analyse.")
            rows_x[name] = (ra, rb)
        for i, (level, d, b, u, res, _dr, _br) in enumerate(rows_a):
            src = d if level == "div" else b if level == "bloc" else u
            raw = d.get("lang_majoritaire") if level == "div" else src.get("lang")
            pur = d.get("purity") if level == "div" else src.get("purity")
            gold, gold_src = resolve_gold_src(raw, gold_map, header, lex_cols)
            if gold is None:
                if pur is not None:  # (div vide : pas de texte, rien à signaler)
                    nolang[relname(f)] += 1
                continue
            if pur is not None and pur < args.min_purity:
                continue
            if is_fr(gold) and args.french_gold == "skip":
                continue
            uid = (u or b or d)["id"]
            res_b = rows_b[i][4] if rows_b else None
            sw, sw_why = pred_swadesh(res)
            swfr, swfr_why = pred_swadesh(res_b) if res_b else (None, "")
            zb_list, zb_why = pred_zaban(res, code2labels) if args.zabanshenas else (None, "")
            zb = None if zb_list is None else (gold if gold in zb_list else zb_list[0])
            z = res.get("zabanshenas") or {}
            hors = is_fr(gold) and args.french_gold == "detection-only"
            if hors:  # hors périmètre de l'identification : on ne garde aucune prédiction
                sw = swfr = zb = None
                sw_why = swfr_why = zb_why = "hors_perimetre"
            text = (u["text"] if u else b["text"] if b else " ".join(
                x["text"] for x in d["blocks"][:1]))[:80]
            unit = {
                "fichier": relname(f), "niveau": level, "id": f"{relname(f)}:{uid}", "gold": gold,
                "purity": pur, "ntok": res["n_tokens"], "texte": text,
                "swadesh": sw, "swadesh_why": sw_why, "swadesh_fr": swfr, "swadesh_fr_why": swfr_why,
                "zaban": zb, "zaban_why": zb_why, "zaban_code": z.get("code"),
                "fr_zaban": bool(z.get("code") and z["code"] in zmap[FR]) if args.zabanshenas else None,
                "fr_filtre": bool(res_b.get("exclu_fr")) if res_b else None,
            }
            for name, _lx, _vc, _mx in extras:
                ra, rb = rows_x[name]
                p1, w1 = pred_swadesh(ra[i][4])
                p2, w2 = pred_swadesh(rb[i][4]) if rb else (None, "")
                if hors:
                    p1 = p2 = None
                    w1 = w2 = "hors_perimetre"
                unit[f"x:{name}"], unit[f"x:{name}_why"] = p1, w1
                unit[f"x:{name}+fr"], unit[f"x:{name}+fr_why"] = p2, w2
            if member2group:
                for k in list(unit):
                    if (k == "gold" or k in ("swadesh", "swadesh_fr", "zaban") or
                            (k.startswith("x:") and not k.endswith("_why"))) and isinstance(unit[k], str):
                        unit[k] = member2group.get(unit[k], unit[k])
            units.append(unit)
            if level == "phrase":
                gold_tokens[gold] += res["n_tokens"]
                raw_tokens[(raw, gold, gold_src)] += res["n_tokens"]

    if nolang:
        print(f"/!\\ {sum(nolang.values())} unité(s) sans langue de référence (aucun xml:lang hérité), "
              f"écartées : " + ", ".join(f"{k} ({v})" for k, v in nolang.items())
              + ". Utilise --default-lang pour leur donner une langue.", file=sys.stderr)
    if not units:
        sys.exit("Aucune unité annotée (xml:lang) trouvée.")
    ident_units = [u for u in units if args.french_gold == "score" or not is_fr(u["gold"])]

    # --- Avertissements sur la référence
    print("Langues de référence (tokens, niveau phrase) :", file=sys.stderr)
    for g, n in gold_tokens.most_common():
        note = ""
        if is_fr(g):
            note = {"detection-only": "  (français : hors des tableaux d'identification ; sert à évaluer la détection du français)",
                    "score": "  (classe « français » : hors lexique ; seuls filtre/Zabanshenas peuvent la trouver)",
                    "skip": "  (français : ignoré)"}[args.french_gold]
        elif g not in lex:
            note = "  /!\\ pas de colonne correspondante dans le lexique : Swadesh ne peut pas la trouver"
        elif args.zabanshenas and g not in zmap:
            note = "  /!\\ pas de code Zabanshenas associé (--zaban-map) : Zabanshenas ne peut pas la trouver"
        print(f"  {g}: {n}{note}", file=sys.stderr)
        for name, lx, _vc, _mx in extras:
            if not is_fr(g) and g not in lx:
                print(f"    /!\\ pas de colonne « {g} » dans la liste {name} : ce système ne peut pas "
                      "la trouver", file=sys.stderr)
        if n < 100 and not (is_fr(g) and args.french_gold == "skip"):
            print(f"    attention : très peu de texte pour « {g} » ; résultats peu fiables", file=sys.stderr)

    print("\nD'où vient chaque langue de référence (valeur de xml:lang -> étiquette ; tokens, niveau phrase) :",
          file=sys.stderr)
    for (raw, gold, src), n in sorted(raw_tokens.items(), key=lambda kv: -kv[1]):
        print(f"  xml:lang={raw!r} -> « {gold} »  [{src}]  {n}", file=sys.stderr)

    systems = [("swadesh", "swadesh")]
    if fr_detect:
        systems.append(("swadesh+fr", "swadesh_fr"))
    for name, _lx, _vc, _mx in extras:
        systems.append((name, f"x:{name}"))
        if fr_detect:
            systems.append((f"{name}+fr", f"x:{name}+fr"))
    if args.zabanshenas:
        systems.append(("zabanshenas", "zaban"))

    def items_for(key, sel):
        return [{"gold": u["gold"], "pred": u[key], "ntok": u["ntok"]} for u in sel]

    summary = []   # lignes du fichier résumé
    print("\n=== Identification de la langue : exactitude par système et par niveau ===")
    print(f"{'système':16}{'niveau':8}{'unités':>7}{'couvert.':>10}{'précision':>11}"
          f"{'exactitude':>12}{'exact.' + U:>14}")
    for name, key in systems:
        for lvl in LEVELS:
            sel = [u for u in ident_units if u["niveau"] == lvl]
            m = metrics(items_for(key, sel))
            summary.append([name, lvl, m["n"], m["repondus"], m["couverture"], m["precision"],
                            m["exactitude"], m["exactitude_tokens"]])
            print(f"{name:16}{lvl:8}{m['n']:>7}{fmt(m['couverture']):>10}{fmt(m['precision']):>11}"
                  f"{fmt(m['exactitude']):>12}{fmt(m['exactitude_tokens']):>14}")
    print("  (couverture = unités avec réponse ; précision = exactes parmi les réponses ;\n"
          "   exactitude = exactes parmi toutes les unités ; exact." + U + " = part du texte bien identifiée)")

    # --- Rappel par langue de référence
    print("\n=== Rappel par langue de référence (exactes / unités) ===")
    golds = sorted({u["gold"] for u in ident_units})
    print(f"{'système':16}{'niveau':8}" + "".join(f"{g[:14]:>16}" for g in golds))
    recall_rows = []
    for name, key in systems:
        for lvl in LEVELS:
            cells = []
            for g in golds:
                sel = [u for u in ident_units if u["niveau"] == lvl and u["gold"] == g]
                ok = sum(1 for u in sel if u[key] == g)
                cells.append(f"{ok}/{len(sel)}")
                recall_rows.append([name, lvl, g, ok, len(sel)])
            print(f"{name:16}{lvl:8}" + "".join(f"{c:>16}" for c in cells))

    # --- Mesures classiques par langue : précision, rappel, F1
    class_rows = []   # [systeme, niveau, ref, support, tp, fp, fn, P, R, F1, support_tok, P_tok, R_tok, F1_tok]
    print("\n=== Mesures par langue : précision / rappel / F1 (en unités ; entre crochets : en " + U + ") ===")
    print(f"{'système':16}{'niveau':8}{'langue':26}{'support':>8}{'préc.':>8}{'rappel':>8}{'F1':>7}"
          f"{'  [préc.':>10}{'rappel':>8}{'F1]':>7}")
    for name, key in systems:
        for lvl in LEVELS:
            sel = items_for(key, [u for u in ident_units if u["niveau"] == lvl])
            if not sel:
                continue
            ru, au, acc_u = per_class(sel)
            rt, at, acc_t = per_class(sel, weighted=True)
            for c in ru:
                a, b_ = ru[c], rt[c]
                class_rows.append([name, lvl, c, a["support"], a["tp"], a["fp"], a["fn"], a["precision"],
                                   a["rappel"], a["f1"], b_["support"], b_["precision"], b_["rappel"], b_["f1"]])
                print(f"{name:16}{lvl:8}{c[:25]:26}{a['support']:>8}{fmt(a['precision']):>8}{fmt(a['rappel']):>8}"
                      f"{fmt(a['f1']):>7}{fmt(b_['precision']):>10}{fmt(b_['rappel']):>8}{fmt(b_['f1']):>7}")
            for k, lab in (("macro", "(moyenne macro)"), ("pondere", "(moyenne pondérée)"), ("micro", "(micro)")):
                if k in au:
                    class_rows.append([name, lvl, lab, "", "", "", "", au[k]["precision"], au[k]["rappel"],
                                       au[k]["f1"], "", at[k]["precision"], at[k]["rappel"], at[k]["f1"]])
                    print(f"{name:16}{lvl:8}{lab:26}{'':>8}{fmt(au[k]['precision']):>8}{fmt(au[k]['rappel']):>8}"
                          f"{fmt(au[k]['f1']):>7}{fmt(at[k]['precision']):>10}{fmt(at[k]['rappel']):>8}"
                          f"{fmt(at[k]['f1']):>7}")
            class_rows.append([name, lvl, "(exactitude)", "", "", "", "", "", acc_u, "", "", "", acc_t, ""])
            print(f"{name:16}{lvl:8}{'(exactitude)':26}{'':>8}{'':>8}{fmt(acc_u):>8}{'':>7}{'':>10}{fmt(acc_t):>8}")
    print("  (abstention = FN pour la langue de référence, jamais FP ; précision non définie = langue jamais\n"
          "   prédite, comptée 0 dans la moyenne macro ; macro/pondérée : langues de référence uniquement ;\n"
          "   micro précision = exactes / réponses, micro rappel = exactitude)")

    # --- Selon la longueur (tous niveaux confondus) : sépare l'effet du niveau de celui de la longueur
    print(f"\n=== Exactitude selon la longueur de l'unité ({U}, tous niveaux confondus) ===")
    print(f"{'système':16}" + "".join(f"{lab:>16}" for _, _, lab in BUCKETS))
    bucket_rows = []
    for name, key in systems:
        cells = []
        for lo, hi, lab in BUCKETS:
            sel = [u for u in ident_units if lo <= u["ntok"] <= hi]
            m = metrics(items_for(key, sel))
            bucket_rows.append([name, lab, m["n"], m["couverture"], m["precision"], m["exactitude"]])
            cells.append(f"{fmt(m['exactitude']).strip()} ({m['n']})")
        print(f"{name:16}" + "".join(f"{c:>16}" for c in cells))
    print("  (exactitude, avec le nombre d'unités entre parenthèses)")

    # --- Français / non-français
    bin_rows = []
    detectors = []
    if args.zabanshenas:
        detectors.append(("zabanshenas", "fr_zaban"))
    if fr_detect:
        detectors.append((f"filtre-{args.fr_filter}", "fr_filtre"))
    if detectors and args.french_gold == "skip":
        print("\n(Détection du français non évaluée : --french-gold skip.)")
    elif detectors and any(is_fr(u["gold"]) for u in units):
        print("\n=== Détection du français (français contre non-français) ===")
        print(f"{'détecteur':20}{'niveau':8}{'n fr':>6}{'n non-fr':>10}{'préc. fr':>10}"
              f"{'rappel fr':>11}{'F1 fr':>8}{'non-fr exclu à tort':>22}{'VP/FP/FN/VN':>16}")
        for name, key in detectors:
            for lvl in LEVELS:
                sel = [{"gold": u["gold"], "pred_fr": u[key]} for u in units if u["niveau"] == lvl]
                m = binary(sel)
                bin_rows.append([name, lvl] + [m[k] for k in (
                    "n_fr", "n_non_fr", "tp", "fp", "fn", "tn", "precision_fr", "rappel_fr",
                    "f1_fr", "exclusion_a_tort")])
                print(f"{name:20}{lvl:8}{m['n_fr']:>6}{m['n_non_fr']:>10}{fmt(m['precision_fr']):>10}"
                      f"{fmt(m['rappel_fr']):>11}{fmt(m['f1_fr']):>8}{fmt(m['exclusion_a_tort']):>22}"
                      f"{str(m['tp']) + '/' + str(m['fp']) + '/' + str(m['fn']) + '/' + str(m['tn']):>16}")
        print("  (« non-fr exclu à tort » = part de l'unité régionale classée française : c'est l'erreur\n"
              "   la plus coûteuse pour le pipeline, car ce texte est perdu pour Swadesh)")
    elif detectors:
        print("\n(Aucune unité de référence française : détection du français non évaluable.)")

    # --- Détail par fichier
    file_rows = []
    fnames = sorted({u["fichier"] for u in units})
    for fn in fnames:
        for name, key in systems:
            for lvl in LEVELS:
                sel = [u for u in ident_units if u["fichier"] == fn and u["niveau"] == lvl]
                m = metrics(items_for(key, sel))
                file_rows.append([fn, name, lvl, m["n"], m["couverture"], m["precision"],
                                  m["exactitude"], m["exactitude_tokens"]])
    if len(fnames) > 1:
        print(f"\n=== Exactitude pondérée par les {U}, par fichier (div / bloc / phrase) ===")
        for fn in fnames:
            for name, key in systems:
                vals = [next((r[7] for r in file_rows if r[0] == fn and r[1] == name and r[2] == lvl), None)
                        for lvl in LEVELS]
                print(f"{fn[:34]:35}{name:16}" + "".join(f"{fmt(v):>9}" for v in vals))

    # --- Fichiers
    out = args.output
    with open(out + "_unites.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["fichier", "niveau", "id", "ref", "purete", "n_" + U, "swadesh", "motif_swadesh",
                    "swadesh_fr", "motif_swadesh_fr", "zaban", "zaban_code", "detecte_fr_zaban", "detecte_fr_filtre"]
                    + [h for name, *_ in extras
                       for h in (name, f"motif_{name}", f"{name}_fr", f"motif_{name}_fr")]
                    + ["texte"])
        for u in units:
            w.writerow([u["fichier"], u["niveau"], u["id"], u["gold"],
                        "" if u["purity"] is None else round(u["purity"], 3), u["ntok"],
                        u["swadesh"] or "", u["swadesh_why"], u["swadesh_fr"] or "", u["swadesh_fr_why"],
                        u["zaban"] or "", u["zaban_code"] or "",
                        "" if u["fr_zaban"] is None else int(u["fr_zaban"]),
                        "" if u["fr_filtre"] is None else int(u["fr_filtre"]),
                        *[v for name, *_ in extras
                          for v in (u[f"x:{name}"] or "", u[f"x:{name}_why"],
                                    u[f"x:{name}+fr"] or "", u[f"x:{name}+fr_why"])],
                        u["texte"].replace("\t", " ")])
    with open(out + "_resume.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["systeme", "niveau", "unites", "repondus", "couverture", "precision",
                    "exactitude", "exactitude_" + U])
        w.writerows(summary)
        w.writerow([])
        w.writerow(["systeme", "niveau", "ref", "exactes", "unites"])
        w.writerows(recall_rows)
        w.writerow([])
        w.writerow(['systeme', 'niveau', 'ref', 'support', 'tp', 'fp', 'fn', 'precision', 'rappel', 'f1', 'support_' + U, 'precision_' + U, 'rappel_' + U, 'f1_' + U])
        w.writerows(class_rows)
        w.writerow([])
        w.writerow(["systeme", "longueur_" + U, "unites", "couverture", "precision", "exactitude"])
        w.writerows(bucket_rows)
        if bin_rows:
            w.writerow([])
            w.writerow(["detecteur", "niveau", "n_fr", "n_non_fr", "tp", "fp", "fn", "tn",
                        "precision_fr", "rappel_fr", "f1_fr", "exclusion_a_tort"])
            w.writerows(bin_rows)
    with open(out + "_fichiers.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["fichier", "systeme", "niveau", "unites", "couverture", "precision",
                    "exactitude", "exactitude_" + U])
        w.writerows(file_rows)
    with open(out + "_classes.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(['systeme', 'niveau', 'ref', 'support', 'tp', 'fp', 'fn', 'precision', 'rappel', 'f1', 'support_' + U, 'precision_' + U, 'rappel_' + U, 'f1_' + U])
        w.writerows(class_rows)
    conf = defaultdict(lambda: [0, 0])
    for name, key in systems:
        for u in ident_units:
            if u[key]:
                pred = u[key]
            else:  # abstention : on garde le motif (trop_court / aucune_info / ex_aequo)
                pred = "(abstention:" + (u.get(key + "_why") or "inconnu") + ")"
            c = conf[(name, u["niveau"], u["gold"], pred)]
            c[0] += 1
            c[1] += u["ntok"]
    # totaux par langue de référence et par prédiction (pour les parts)
    tot_ref, tot_pred = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for (a_, b_, c_, d_), (n, t) in conf.items():
        tot_ref[(a_, b_, c_)][0] += n
        tot_ref[(a_, b_, c_)][1] += t
        tot_pred[(a_, b_, d_)][0] += n
        tot_pred[(a_, b_, d_)][1] += t
    with open(out + "_confusion.tsv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["systeme", "niveau", "ref", "predit", "unites", U,
                    "part_de_ref_unites", "part_de_ref_" + U, "part_du_predit_unites", "part_du_predit_" + U])
        for (a_, b_, c_, d_), (n, t) in sorted(conf.items()):
            r, q = tot_ref[(a_, b_, c_)], tot_pred[(a_, b_, d_)]
            w.writerow([a_, b_, c_, d_, n, t,
                        round(n / r[0], 4) if r[0] else "", round(t / r[1], 4) if r[1] else "",
                        round(n / q[0], 4) if q[0] else "", round(t / q[1], 4) if q[1] else ""])
    print(f"\nFichiers : {out}_unites.tsv, {out}_resume.tsv, {out}_classes.tsv, {out}_fichiers.tsv, {out}_confusion.tsv",
          file=sys.stderr)


if __name__ == "__main__":
    main()
