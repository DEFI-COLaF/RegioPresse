#!/usr/bin/env python3
"""Étiquetage de fichiers XML-TEI par xml:lang, avec plusieurs méthodes d'identification.

Pour chaque fichier et chaque méthode, le script écrit une COPIE annotée du fichier
(<sortie>/<méthode>/<fichier>.xml) et un tableau de tous les scores (<sortie>/annotations.tsv).

Principe (de haut en bas) :
  1. document : une langue pour tout le texte, posée sur <body> ;
  2. <div> : si la langue d'une div est différente de celle de son parent ET que la méthode est
     assez sûre, la div reçoit son propre xml:lang ; sinon elle hérite ;
  3. blocs (<p>, <ab>, <head>, <note>, <lg>, <l>...) : même règle, par rapport à la langue
     effective de la div ;
  4. (option --tag-levels div,bloc,phrase) les vers <l> d'un <lg>.
Quand une méthode s'abstient (texte trop court, preuves insuffisantes), rien n'est écrit : le
passage hérite de la langue du niveau supérieur.

Méthodes (chacune produit son propre jeu de fichiers) :
  swadesh        liste de Swadesh seule
  swadesh+fr     filtre français (--fr-filter) puis Swadesh ; peut étiqueter du français
  NOM, NOM+fr    autres listes de mots (--extra-lexicon NOM=chemin.tsv)
  zabanshenas    modèle Zabanshenas (--zabanshenas, avec --zaban-map)
  francais       détecteur français / non français (avec --fr-filter) : dit, pour chaque passage
                 (article, bloc...), s'il est en français ou non. Score = confiance dans cette
                 décision ; la part de mots jugés français est dans hits / items_valides.

Scores (tous entre 0 et 1, voir le README affiché par --help-scores) :
  Zabanshenas : probabilité de la langue prédite (avec --zaban-chunk : part du texte, pondérée
                par la probabilité, attribuée à la langue gagnante)
  français    : probabilité du français donnée par le filtre (GlotLID ou Zabanshenas)
  Swadesh     : « part des indices » = score de la langue gagnante / somme des scores de toutes
                les langues ; en plus : nombre d'items trouvés (hits) et marge sur la 2e langue.
  Écrit dans le TEI : xml:lang et cert (high / medium / low d'après le score) ; les valeurs
  numériques sont dans annotations.tsv (clé : chemin XPath de l'élément).

Un seul fichier pour plusieurs méthodes (--merge) : xml:lang ne peut porter qu'une valeur, donc
chaque méthode écrit deux attributs dans un espace de noms à part (--attr-ns, préfixe lid) :
    lid:<méthode>        la langue (valeur de xml:lang)      ex. lid:swadesh-fr="cors-1241"
    lid:<méthode>-score  le degré de certitude (0 à 1)       ex. lid:swadesh-fr-score="0.8125"
Variante TEI (--merge --merge-style ana) : un seul attribut @ana par élément, qui pointe vers des
catégories déclarées dans <teiHeader>/<encodingDesc>/<classDecl>/<taxonomy xml:id="lid-taxonomy"> :
    ana="#swadesh-fr_cors-1241_0.7 #zabanshenas_cors-1241_0.8"
chaque pointeur = méthode _ langue _ borne basse de la tranche de score (--score-bins 10 : tranches
de 0,1 ; « 0.7 » = score de 0,7 à moins de 0,8 ; « na » = pas de score). Les catégories utilisées sont
déclarées automatiquement. Les valeurs exactes sont dans annotations.tsv.

--primary <méthode> écrit en plus xml:lang + cert pour cette méthode (fichier utilisable tel quel par
les outils TEI). Les attributs ne s'héritent pas par règle XML : par convention, un élément sans
lid:<méthode> reprend la valeur de l'ancêtre le plus proche qui en a un.

--force-div-type article : pour les <div type="article">, une langue est TOUJOURS écrite (la meilleure
candidate de la méthode, même sous les seuils ou identique à celle du parent ; à défaut de candidate,
la langue héritée, avec cert="unknown"). La colonne force d'annotations.tsv les repère.

Exemple :
    python tag_lid.py lexique.tsv corpus_tei/ -o corpus_annote \\
        --lang-codes "cors.=cors-1241,français=stan-1290" \\
        --fr-filter glotlid --glotlid-model cis-lmu/glotlid \\
        --zabanshenas --zaban-map "cors.=cos" --min-tokens 4
"""
import argparse
import csv
import datetime
import json
import math  # noqa: F401
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import swadesh_lid_xml as S  # noqa: E402

import inspect  # noqa: E402
if "root" not in inspect.signature(S.parse_xml).parameters:
    sys.exit(f"{S.__file__} est une ancienne version de swadesh_lid_xml.py : remplace-la par "
             "la dernière version, dans le même dossier que tag_lid.py.")

FR = "français"
NONFR = "non-français"
XML_LANG = S.XML_LANG
HELP_SCORES = """
CHOIX DES SCORES
----------------
Les trois méthodes ne produisent pas la même chose, donc le score est défini par méthode ; tous
sont ramenés entre 0 et 1, mais ce ne sont PAS des probabilités calibrées.

* Zabanshenas : probabilité que le modèle attribue à la classe gagnante (softmax). Les modèles de ce
  type sont souvent trop confiants : une valeur de 0,9 ne veut pas dire « 90 % de chances d'avoir
  raison ». Avec --zaban-chunk N, le texte est coupé en fenêtres de N mots ; le score est la part du
  texte (pondérée par la probabilité) qui vote pour la langue gagnante.
* Français (filtre GlotLID ou Zabanshenas) : probabilité du français ; l'unité est jugée française
  si elle dépasse --fr-threshold.
* Swadesh : le pourcentage d'items trouvés n'est pas comparable d'une langue à l'autre (les listes
  n'ont pas la même longueur) ni d'un texte à l'autre (il dépend de la longueur). On utilise donc la
  « part des indices » : score de la langue gagnante divisé par la somme des scores de toutes les
  langues. 1 = seule cette langue a des items dans le texte ; 1/N = indices répartis également.
  Pour ne pas se fier à un seul mot, une décision n'est acceptée que si hits >= --min-hits (items
  trouvés dans la langue gagnante) et marge >= --min-margin (points de pourcentage d'avance sur
  la 2e langue).
* cert (dans le TEI) : high si score >= --cert-high, medium si >= --cert-medium, sinon low.
* Les seuils par défaut sont des points de départ : choisis-les avec l'évaluation (evaluate_lid.py),
  par exemple en cherchant le seuil à partir duquel la précision devient acceptable.
"""


def local(tag):
    return S.local(tag)


def parse_tree(path):
    """Arbre ElementTree qui conserve commentaires et instructions de traitement, et les préfixes."""
    for _ev, (pre, uri) in ET.iterparse(path, events=("start-ns",)):
        try:
            ET.register_namespace(pre, uri)
        except ValueError:
            pass
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
    return ET.parse(path, parser=parser)


def tei_status(path):
    """'tei' si la racine est <TEI>, 'autre' pour un autre XML, 'illisible' si mal formé."""
    try:
        for _ev, el in ET.iterparse(path, events=("start",)):
            return "tei" if local(el.tag) == "TEI" else "autre"
    except (ET.ParseError, OSError, UnicodeDecodeError):
        return "illisible"
    return "illisible"


def read_prolog(path):
    """Texte qui précède la balise racine (déclaration XML, instructions xml-model, commentaires)."""
    import xml.parsers.expat
    data = Path(path).read_bytes()
    pos = []
    parser = xml.parsers.expat.ParserCreate()
    parser.StartElementHandler = lambda name, attrs: pos.append(parser.CurrentByteIndex) if not pos else None
    parser.Parse(data, True)
    return data[:pos[0]].decode("utf-8") if pos else ""


def save_tree(root, target, src_path):
    """Écrit l'arbre en conservant le prologue du fichier source et le style « <lb/> »."""
    prolog = read_prolog(src_path)
    if not prolog.lstrip().startswith("<?xml"):
        prolog = "<?xml version='1.0' encoding='UTF-8'?>\n" + prolog
    body = ET.tostring(root, encoding="unicode").replace(" />", "/>")
    Path(target).write_text(prolog + body + "\n", encoding="utf-8")


def xpath_of(elem, parents):
    parts = []
    while elem is not None:
        par = parents.get(elem)
        name = local(elem.tag)
        if par is None:
            parts.append(name)
        else:
            sibs = [c for c in par if isinstance(c.tag, str) and local(c.tag) == name]
            parts.append(f"{name}[{sibs.index(elem) + 1}]")
        elem = par
    return "/" + "/".join(reversed(parts))


def cert_of(score, args):
    if score is None:
        return "unknown"
    return "high" if score >= args.cert_high else "medium" if score >= args.cert_medium else "low"


def empty_dec(n_tokens, why):
    return {"label": None, "score": None, "hits": None, "valid": None, "margin": None,
            "second": None, "accepted": False, "why": why, "n_tokens": n_tokens}


def lex_decision(res, args):
    """Décision d'une méthode « liste de mots » à partir du résultat de score_text."""
    n = res["n_tokens"]
    if res.get("trop_court"):
        return empty_dec(n, "trop_court")
    if res["best"] is None:
        return empty_dec(n, "aucune_info")
    if "ties" in res:
        return empty_dec(n, "ex_aequo")
    top = res["best"]
    scores = {l: v for l, v in res["scores"].items() if v}
    pct = res["scores"][top]
    total = sum(scores.values())
    others = [(v, l) for l, v in scores.items() if l != top]
    second_v, second_l = max(others) if others else (0.0, None)
    hits, valid = res["counts"][top]
    share = pct / total if total else 0.0
    margin = pct - second_v
    ok = hits >= args.min_hits and margin >= args.min_margin and share >= args.min_share
    return {"label": top, "score": round(share, 4), "hits": hits, "valid": valid,
            "margin": round(margin, 2), "second": second_l, "accepted": ok,
            "why": "" if ok else "preuves_insuffisantes", "n_tokens": n}


def zab_decision(z, n_tokens, code2label, args, too_short):
    if too_short:
        return empty_dec(n_tokens, "trop_court")
    code = (z or {}).get("code")
    if not code:
        return empty_dec(n_tokens, "aucune_info")
    sc = z.get("score")
    label = code2label.get(code, code)
    ok = sc is not None and sc >= args.min_prob
    d = empty_dec(n_tokens, "" if ok else "probabilite_faible")
    d.update(label=label, score=None if sc is None else round(float(sc), 4), accepted=ok)
    return d


def fr_decision(p_fr, n_tokens):
    d = empty_dec(n_tokens, "")
    d.update(label=FR, score=None if p_fr is None else round(p_fr, 4), accepted=True)
    return d


def bin_label(score, n):
    """Borne basse (texte) de la tranche de score ; « na » sans score."""
    if score is None:
        return "na", None
    i = min(int(score * n + 1e-9), n - 1)
    d = next((k for k in range(1, 4) if (10 ** k) % n == 0), 3)
    return f"{i / n:.{d}f}", f"{(i + 1) / n:.{d}f}"


def declare_taxonomy(root, cats, fr_code="fr", nonfr_code="non-fr"):
    """Déclare les catégories d'@ana dans teiHeader/encodingDesc/classDecl/taxonomy."""
    header = next((e for e in root.iter() if local(e.tag) == "teiHeader"), None)
    if header is None:
        print("/!\\ pas de <teiHeader> : les catégories d'@ana ne peuvent pas être déclarées.",
              file=sys.stderr)
        return
    ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    enc = next((c for c in header if local(c.tag) == "encodingDesc"), None)
    if enc is None:
        enc = ET.Element(ns + "encodingDesc")
        idx = next((i + 1 for i, c in enumerate(header) if local(c.tag) == "fileDesc"), len(header))
        header.insert(idx, enc)
    cd = next((c for c in enc if local(c.tag) == "classDecl"), None)
    if cd is None:
        cd = ET.SubElement(enc, ns + "classDecl")
    if any(t.get(S.XML_ID) == "lid-taxonomy" for t in cd):
        return
    taxo = ET.SubElement(cd, ns + "taxonomy", {S.XML_ID: "lid-taxonomy"})
    for cid, (mname, code, low, high) in sorted(cats.items()):
        cat = ET.SubElement(taxo, ns + "category", {S.XML_ID: cid})
        desc = ET.SubElement(cat, ns + "catDesc")
        if mname == "francais":
            quoi = "français" if code == fr_code else "non français"
            desc.text = (f"Passage {quoi} d'après le détecteur de français ; "
                         + ("confiance non disponible" if low == "na"
                            else f"confiance de {low} à moins de {high}"))
            continue
        desc.text = (f"Langue « {code} » d'après la méthode {mname} ; score non disponible"
                     if low == "na" else
                     f"Langue « {code} » d'après la méthode {mname} ; score de {low} à moins de {high}"
                     if high != "1.0" and high != "1.00" else
                     f"Langue « {code} » d'après la méthode {mname} ; score de {low} à 1")


def add_change(root, text):
    """Ajoute une note de révision dans <teiHeader>/<revisionDesc> (dernier enfant du header)."""
    header = next((e for e in root.iter() if local(e.tag) == "teiHeader"), None)
    if header is None:
        return
    ns = root.tag[:root.tag.index("}") + 1] if root.tag.startswith("{") else ""
    rev = next((c for c in header if local(c.tag) == "revisionDesc"), None)
    if rev is None:
        rev = ET.SubElement(header, ns + "revisionDesc")
    ch = ET.SubElement(rev, ns + "change", {"when": datetime.date.today().isoformat()})
    ch.text = text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lexicon", nargs="?", help="liste de mots principale (TSV)")
    ap.add_argument("xml", nargs="?", help="fichier XML-TEI ou dossier de .xml")
    ap.add_argument("-o", "--output", default="annote", help="dossier de sortie")
    ap.add_argument("--help-scores", action="store_true", help="explique le choix des scores et quitte")
    ap.add_argument("--methods", help="méthodes à lancer, séparées par des virgules (défaut : toutes "
                                      "celles que les options rendent disponibles)")
    ap.add_argument("--lang-codes", default="",
                    help="étiquette de la méthode -> valeur de xml:lang, ex. "
                         "'cors.=cors-1241,français=stan-1290' (« français » vaut « fr » par défaut)")
    ap.add_argument("--default-lang", help="xml:lang du document si la méthode n'en trouve aucun")
    ap.add_argument("--force-div-type", default="",
                    help="types de <div> (séparés par des virgules, ex. article) pour lesquels une "
                         "langue est toujours écrite, même sous les seuils")
    ap.add_argument("--merge", action="store_true",
                    help="un seul fichier de sortie pour toutes les méthodes (attributs lid:<méthode> "
                         "et lid:<méthode>-score) au lieu d'un dossier par méthode")
    ap.add_argument("--merge-style", choices=["attrs", "ana"], default="attrs",
                    help="avec --merge : attrs = attributs lid:<méthode> et lid:<méthode>-score ; "
                         "ana = @ana pointant vers des catégories déclarées dans l'en-tête TEI")
    ap.add_argument("--score-bins", type=int, default=10,
                    help="avec --merge-style ana : nombre de tranches de score entre 0 et 1 "
                         "(10 = tranches de 0,1 ; 20 = 0,05 ; 100 = 0,01)")
    ap.add_argument("--primary",
                    help="avec --merge : méthode dont la langue est aussi écrite dans xml:lang (+ cert)")
    ap.add_argument("--attr-ns", default="http://example.org/ns/lid",
                    help="espace de noms des attributs de --merge (défaut : %(default)s)")
    ap.add_argument("--attr-prefix", default="lid", help="préfixe de cet espace de noms (défaut : lid)")
    ap.add_argument("--tag-levels", default="div,bloc",
                    help="niveaux où une langue peut être écrite : div, bloc, phrase (= vers <l> d'un "
                         "<lg>). Défaut : div,bloc (les vers/phrases courts sont peu fiables)")
    ap.add_argument("--min-tokens", type=int, default=4, help="en dessous, la méthode s'abstient")
    ap.add_argument("--count-types", action="store_true",
                    help="chaque forme n'est comptée qu'une fois par passage : --min-tokens et la part de "
                         "français (--fr-share) portent sur les FORMES DISTINCTES, pas sur les tokens")
    ap.add_argument("--min-hits", type=int, default=3, help="Swadesh : items trouvés minimum")
    ap.add_argument("--min-share", type=float, default=0.5, help="Swadesh : part des indices minimale")
    ap.add_argument("--min-margin", type=float, default=0.0,
                    help="Swadesh : avance minimale (en points de %%) sur la 2e langue")
    ap.add_argument("--min-prob", type=float, default=0.6, help="Zabanshenas : probabilité minimale")
    ap.add_argument("--cert-high", type=float, default=0.85)
    ap.add_argument("--cert-medium", type=float, default=0.65)
    ap.add_argument("--extra-lexicon", action="append", default=[], metavar="NOM=CHEMIN")
    ap.add_argument("--zabanshenas", action="store_true")
    ap.add_argument("--zaban-map", default="", help="étiquette -> codes Zabanshenas, ex. 'cors.=cos'")
    ap.add_argument("--zaban-max-length", type=int)
    ap.add_argument("--zaban-chunk", type=int, default=0)
    ap.add_argument("--fr-filter", choices=["glotlid", "zabanshenas"])
    ap.add_argument("--fr-threshold", type=float, default=0.8)
    ap.add_argument("--fr-share", type=float, default=0.5,
                    help="part minimale de mots jugés français pour qu'un passage (div, bloc, document) "
                         "soit déclaré français (méthode francais, et méthodes +fr) ; défaut 0,5")
    ap.add_argument("--fr-codes", default="")
    ap.add_argument("--glotlid-model")
    ap.add_argument("--glob", default="*.xml",
                    help="motif des fichiers cherchés dans tous les sous-dossiers (défaut : %(default)s)")
    ap.add_argument("--no-tei-check", action="store_true",
                    help="traiter aussi les XML dont la racine n'est pas <TEI> (par défaut, ils sont ignorés)")
    ap.add_argument("--report-sentences", action="store_true",
                    help="ajoute à annotations.tsv les phrases de prose (non étiquetables dans le TEI)")
    ap.add_argument("--ignore-cols", default="n°")
    ap.add_argument("--french-col")
    ap.add_argument("--variant-sep", default="/;,")
    ap.add_argument("--french-forms")
    ap.add_argument("--drop-identical-fr", action="store_true")
    ap.add_argument("--block-tags", default=S.DEFAULT_BLOCK_TAGS)
    ap.add_argument("--exclude-tags", default="")
    args = ap.parse_args()

    if args.help_scores:
        print(HELP_SCORES)
        return
    if not args.lexicon or not args.xml:
        ap.error("lexicon et xml sont requis")

    levels = {x.strip() for x in args.tag_levels.split(",") if x.strip()}
    bt = {t.strip() for t in args.block_tags.split(",") if t.strip()}
    ex = {t.strip() for t in args.exclude_tags.split(",") if t.strip()}

    # --- listes de mots
    ignore_cols = [c for c in args.ignore_cols.split(",") if c.strip()]
    french_forms = S.load_french_forms(args.french_forms) if args.french_forms else None

    def load(path):
        lx, _ = S.load_lexicon(path, args.french_col, args.variant_sep, french_forms, False,
                               ignore_cols, None, args.drop_identical_fr)
        vc, mx = S.build_vocab(lx)
        return lx, vc, mx

    lexes = {"swadesh": load(args.lexicon)}
    for spec in args.extra_lexicon:
        name, sep, path = spec.partition("=")
        if not sep:
            path, name = spec, Path(spec).stem
        name = name.strip()
        if name in lexes or name == "zabanshenas":
            sys.exit(f"Nom de liste déjà utilisé ou réservé : {name!r}")
        lexes[name] = load(path)

    # --- modèles
    zmap = S.parse_zaban_map(args.zaban_map)
    zmap.setdefault(FR, {"fra", "fr"})
    code2label = {c: lab for lab, codes in zmap.items() for c in codes}
    _by_code = {}
    for lab, cs in zmap.items():
        for c in cs:
            _by_code.setdefault(c, []).append(lab)
    for c, labs in _by_code.items():
        if len(labs) > 1:
            print(f"/!\\ --zaban-map : le code « {c} » est associé à plusieurs étiquettes ({', '.join(labs)}) ; "
                  f"« {code2label[c]} » (la dernière) sera utilisée. Pour étiqueter, associe un code "
                  f"Zabanshenas à UNE étiquette (ex. occitan=oci, puis --lang-codes occitan=…).",
                  file=sys.stderr)
    need_zab = args.zabanshenas or args.fr_filter == "zabanshenas"
    zab_detect = S.make_detector(args.zaban_max_length, args.zaban_chunk) if need_zab else None
    fr_detect = (S.make_fr_detector(args.fr_filter, args.fr_threshold, args.fr_codes,
                                    args.glotlid_model, zab_detect) if args.fr_filter else None)
    if fr_detect:
        _raw, _cache = fr_detect, {}

        def fr_detect(text):
            if text not in _cache:
                _cache[text] = _raw(text)
            return _cache[text]

    # --- méthodes disponibles : (nom, type, liste, avec filtre)
    methods = []
    for name in lexes:
        methods.append((name, "lex", name, False))
        if fr_detect:
            methods.append((f"{name}+fr", "lex", name, True))
    if args.zabanshenas:
        methods.append(("zabanshenas", "zab", "swadesh", False))
    if fr_detect:
        methods.append(("francais", "fr", "swadesh", True))
    if args.methods:
        want = {m.strip() for m in args.methods.split(",") if m.strip()}
        unknown = want - {m[0] for m in methods}
        if unknown:
            sys.exit(f"Méthode(s) indisponible(s) : {sorted(unknown)}. Disponibles : {[m[0] for m in methods]}")
        methods = [m for m in methods if m[0] in want]
    if not methods:
        sys.exit("Aucune méthode à lancer.")

    codes = {}
    for part in args.lang_codes.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            k, v = k.strip(), v.strip()
            if k in codes and codes[k] != v:
                print(f"/!\\ --lang-codes : l'étiquette « {k} » est donnée deux fois (« {codes[k]} » puis "
                      f"« {v} ») ; seule la dernière est retenue. Format : étiquette=xml:lang.",
                      file=sys.stderr)
            codes[k] = v
    codes.setdefault(FR, "fr")
    codes.setdefault(NONFR, "non-fr")
    warned = set()

    def code_of(label):
        if label in codes:
            return codes[label]
        if label not in warned:
            warned.add(label)
            print(f"/!\\ pas de code xml:lang pour l'étiquette « {label} » : utilise --lang-codes "
                  f"(« {label}=xxxx »). L'étiquette est écrite telle quelle.", file=sys.stderr)
        return label

    p = Path(args.xml)
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    if p.is_dir():
        out_abs = outdir.resolve()
        cand = []
        for f in sorted(p.rglob(args.glob)):
            if not f.is_file():
                continue
            try:
                f.resolve().relative_to(out_abs)  # ne pas retraiter le dossier de sortie
                continue
            except ValueError:
                cand.append(f)
    else:
        cand = [p]
    files, ignored = [], {"autre": [], "illisible": []}
    for f in cand:
        st = tei_status(f)
        if st == "tei" or (st == "autre" and args.no_tei_check):
            files.append(f)
        else:
            ignored[st].append(f)
    n_dirs = len({f.parent for f in files})
    print(f"{len(cand)} fichier(s) {args.glob} trouvé(s) : {len(files)} TEI retenu(s) "
          f"dans {n_dirs} dossier(s) ; {len(ignored['autre'])} XML non TEI ignoré(s) ; "
          f"{len(ignored['illisible'])} illisible(s) ignoré(s).", file=sys.stderr)
    for f in ignored["illisible"]:
        print(f"  illisible : {f}", file=sys.stderr)
    if not files:
        sys.exit("Aucun fichier TEI à traiter.")
    rows_out = []
    summary = []
    timings = []  # [fichier, méthode, secondes, phrases, divs]

    force_types = {t.strip() for t in args.force_div_type.split(",") if t.strip()}
    if args.merge and args.merge_style == "ana" and args.score_bins < 1:
        sys.exit("--score-bins doit être >= 1")
    if args.merge and args.merge_style == "attrs":
        try:
            ET.register_namespace(args.attr_prefix, args.attr_ns)
        except ValueError as e:
            sys.exit(f"--attr-prefix invalide : {e}")
        if args.primary and args.primary not in {m[0] for m in methods}:
            sys.exit(f"--primary : méthode inconnue {args.primary!r} (disponibles : {[m[0] for m in methods]})")

    def slug(name):
        sl = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
        return sl if re.match(r"[A-Za-z_]", sl) else "m-" + sl

    def open_file(f):
        tree = parse_tree(f)
        root = tree.getroot()
        for e in root.iter():  # on repart d'un fichier sans xml:lang
            if isinstance(e.tag, str):
                e.attrib.pop(XML_LANG, None)
        parents = {c: pr for pr in root.iter() for c in pr}
        divs = S.parse_xml(str(f), bt, ex, None, root=root)
        return tree, root, parents, divs

    def note_text(names):
        return (f"Étiquetage automatique des langues par la méthode « {names} » (tag_lid.py ; seuils : "
                f"min_tokens={args.min_tokens}, min_hits={args.min_hits}, min_share={args.min_share}, "
                f"min_prob={args.min_prob}, fr_threshold={args.fr_threshold if fr_detect else 'sans objet'}"
                f"{'; div forcées : ' + ','.join(sorted(force_types)) if force_types else ''}).")

    def process_file(f):
        rel = f.relative_to(p) if p.is_dir() else Path(f.name)
        shared = open_file(f) if args.merge else None
        cats = {}
        for mname, kind, lname, with_fr in methods:
            tree, root, parents, divs = shared if args.merge else open_file(f)
            t_method0 = time.perf_counter()
            lex, vocab, maxn = lexes[lname]
            ns = SimpleNamespace(texts=str(f), block_tags=args.block_tags, exclude_tags=args.exclude_tags,
                                 min_tokens=args.min_tokens, default_lang=None, count_types=args.count_types)
            rows = S.run_xml(ns, lex, vocab, maxn, zab_detect if kind == "zab" else None,
                             fr_detect if with_fr else None, divs=divs)[1]
            res_div, res_blk, res_unit = {}, {}, {}
            for level, d, b, u, res, _a, _b in rows:
                if level == "div":
                    res_div[d["id"]] = res
                elif level == "bloc":
                    res_blk[b["id"]] = res
                else:
                    res_unit[u["id"]] = res
            p_fr = {uid: r["fr_filtre"]["p_fr"] for uid, r in res_unit.items()
                    if r.get("fr_filtre", {}).get("p_fr") is not None}
            french_ids = {uid for uid, r in res_unit.items() if r.get("exclu_fr")} if with_fr else set()
            sl = slug(mname)

            def writable(label):
                """Hors --merge, « non-français » n'a pas de xml:lang : on n'écrit rien."""
                return not (kind == "fr" and label == NONFR and not args.merge)

            def put(elem, code, score, cert):
                """Écrit la langue : xml:lang + cert, ou attributs lid:<méthode>(-score) avec --merge."""
                if args.merge:
                    if args.merge_style == "ana":
                        low, high = bin_label(score, args.score_bins)
                        cid = slug(f"{mname}_{code}_{low}")
                        cats[cid] = (mname, code, low, high)
                        elem.set("ana", (elem.get("ana", "") + " #" + cid).strip())
                    else:
                        elem.set(f"{{{args.attr_ns}}}{sl}", code)
                        if score is not None:
                            elem.set(f"{{{args.attr_ns}}}{sl}-score", str(score))
                    if mname == args.primary:
                        elem.set(XML_LANG, code)
                        elem.set("cert", cert)
                else:
                    elem.set(XML_LANG, code)
                    elem.set("cert", cert)

            def mean_p(units):
                v = [p_fr[u["id"]] for u in units if u["id"] in p_fr]
                return sum(v) / len(v) if v else None

            def fr_share_of(units):
                """(part de formes/mots jugés français, français, jugés) parmi les phrases jugées.
                Avec --count-types : formes distinctes (une forme présente dans plusieurs phrases ne
                compte qu'une fois ; elle est « française » si elle figure dans une phrase française)."""
                if args.count_types:
                    fr_forms, all_forms = set(), set()
                    for u in units:
                        fi = res_unit.get(u["id"], {}).get("fr_filtre")
                        if fi and fi.get("juge"):
                            forms = set(S.tokenize(u["text"]))
                            all_forms |= forms
                            if fi["francais"]:
                                fr_forms |= forms
                    return (len(fr_forms) / len(all_forms) if all_forms else None), len(fr_forms), len(all_forms)
                frt = tot = 0
                for u in units:
                    fi = res_unit.get(u["id"], {}).get("fr_filtre")
                    if fi and fi.get("juge"):
                        n_u = len(S.tokenize(u["text"]))
                        tot += n_u
                        frt += n_u if fi["francais"] else 0
                return (frt / tot if tot else None), frt, tot

            def fr_binary_decision(units):
                sh, frt, tot = fr_share_of(units)
                n_all = S.size_of([u["text"] for u in units], args.count_types)
                if sh is None:
                    return empty_dec(n_all, "non_juge")
                is_fr_ = sh >= args.fr_share
                d = empty_dec(n_all, "")
                d.update(label=FR if is_fr_ else NONFR, score=round(sh if is_fr_ else 1 - sh, 4),
                         hits=frt, valid=tot, accepted=True)
                return d

            def french_majority(units):
                sh, _f, tot = fr_share_of(units)
                return bool(with_fr and tot and sh >= args.fr_share)

            def decide(res, units):
                """Décision d'un div/bloc/vers à partir de son résultat run_xml."""
                if kind == "fr":
                    return fr_binary_decision(units)
                if kind == "lex":
                    if french_majority(units):
                        return fr_decision(mean_p([u for u in units if u["id"] in french_ids]),
                                           res["n_tokens"])
                    if res.get("exclu_fr"):
                        return fr_decision(mean_p(units), res["n_tokens"])
                    return lex_decision(res, args)
                z = res.get("zabanshenas")
                return zab_decision(z, res["n_tokens"], code2label, args,
                                    bool(res.get("trop_court") or z is None or z.get("ecarte")))

            # --- niveau document
            all_units = [u for d in divs for b in d["blocks"] for u in b["units"]]
            kept = []
            for d in divs:
                for b in d["blocks"]:
                    if with_fr:
                        kept.extend(u["text"] for u in b["units"] if u["id"] not in french_ids)
                    else:
                        kept.extend(b["_segments"])
            tot_tok = S.size_of([u["text"] for u in all_units] if with_fr else kept, args.count_types)
            if kind == "fr":
                doc_dec = fr_binary_decision(all_units)
            elif kind == "lex":
                if french_majority(all_units):
                    doc_dec = fr_decision(mean_p([u for u in all_units if u["id"] in french_ids]), tot_tok)
                else:
                    doc_dec = lex_decision(S.score_text(kept, lex, vocab, maxn, args.min_tokens, args.count_types), args)
            else:
                txt = " ".join(kept)
                n_doc = S.size_of(txt, args.count_types)
                z = zab_detect(txt) if n_doc >= args.min_tokens else None
                doc_dec = zab_decision(z, n_doc, code2label, args, z is None)
            body = next((e for e in root.iter() if local(e.tag) == "body"), None)
            if body is None:
                body = next((e for e in root.iter() if local(e.tag) == "text"), root)
            doc_label = doc_dec["label"] if doc_dec["accepted"] else None
            if doc_label is None and args.default_lang:
                doc_code = args.default_lang  # déjà un code xml:lang
            else:
                doc_code = code_of(doc_label) if doc_label else None
            if doc_code and writable(doc_label):
                put(body, doc_code, doc_dec["score"] if doc_dec["accepted"] else None,
                    cert_of(doc_dec["score"], args) if doc_dec["accepted"] else "unknown")
            rows_out.append(dict(fichier=str(rel), methode=mname, niveau="doc", chemin=xpath_of(body, parents),
                                 balise=local(body.tag), unite="doc", dec=doc_dec, code=doc_code or "",
                                 parent="", ecrit=int(bool(doc_code)), force=0, texte=" ".join(kept)[:80]))

            counts = {"div": 0, "bloc": 0, "phrase": 0}

            def emit(niveau, elem, uid, dec, eff, texte, forced=False):
                """Écrit éventuellement la langue ; renvoie la langue effective."""
                write, new_eff, code = False, eff, None
                if elem is not None and forced:
                    lab = dec["label"] or eff
                    if lab and writable(lab):
                        write, new_eff, code = True, lab, code_of(lab)
                        put(elem, code, dec["score"] if dec["label"] else None,
                            cert_of(dec["score"], args) if dec["label"] else "unknown")
                elif (niveau in levels and elem is not None and dec["accepted"] and dec["label"] != eff
                      and writable(dec["label"])):
                    write, new_eff, code = True, dec["label"], code_of(dec["label"])
                    put(elem, code, dec["score"], cert_of(dec["score"], args))
                if write:
                    counts[niveau] += 1
                rows_out.append(dict(fichier=str(rel), methode=mname, niveau=niveau,
                                     chemin=xpath_of(elem, parents) if elem is not None else "",
                                     balise=local(elem.tag) if elem is not None else "", unite=uid,
                                     dec=dec, code=code or (code_of(dec["label"]) if dec["label"] else ""),
                                     parent=code_of(eff) if eff else "", ecrit=int(write),
                                     force=int(write and forced), texte=texte[:80]))
                return new_eff

            def walk(d, eff):
                dd = decide(res_div[d["id"]], [u for b in d["blocks"] for u in b["units"]])
                txt = d["blocks"][0]["text"] if d["blocks"] else ""
                eff_d = (emit("div", d["_elem"], d["id"], dd, eff, txt, forced=d["type"] in force_types)
                         if d["_elem"] is not None else eff)
                for b in d["blocks"]:
                    db = decide(res_blk[b["id"]], b["units"])
                    eff_b = emit("bloc", b["_elem"], b["id"], db, eff_d, b["text"])
                    for u in b["units"]:
                        r = res_unit.get(u["id"])
                        if u["_elem"] is not None and r is not None and u["_elem"] is not b["_elem"]:
                            emit("phrase", u["_elem"], u["id"], decide(r, [u]), eff_b, u["text"])
                        elif args.report_sentences and r is not None:
                            du = decide(r, [u])
                            rows_out.append(dict(fichier=str(rel), methode=mname, niveau="phrase",
                                                 chemin="", balise="(prose)", unite=u["id"], dec=du,
                                                 code=code_of(du["label"]) if du["label"] else "",
                                                 parent=code_of(eff_b) if eff_b else "", ecrit=0, force=0,
                                                 texte=u["text"][:80]))
                for sub in d["_sub"]:
                    walk(sub, eff_d)

            for d in divs:
                if d["parent"] is None:
                    walk(d, doc_label)
            timings.append([str(rel), mname, round(time.perf_counter() - t_method0, 4),
                            sum(1 for _ in all_units), len(divs)])

            if not args.merge:
                add_change(root, note_text(mname))
                target = outdir / mname.replace("+", "-plus-") / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                save_tree(root, target, f)
            print(f"{rel} [{mname}] : langue du document = {doc_code} ; balises écrites : {counts}",
                  file=sys.stderr)
        if args.merge:
            if args.merge_style == "ana":
                declare_taxonomy(shared[1], cats, codes[FR], codes[NONFR])
            add_change(shared[1], note_text(" + ".join(m[0] for m in methods)))
            target = outdir / "fusion" / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            save_tree(shared[1], target, f)

    failures = []
    for i_f, f in enumerate(files, 1):
        n0 = len(rows_out)
        try:
            print(f"[{i_f}/{len(files)}] {f}", file=sys.stderr)
            process_file(f)
        except Exception as e:  # un fichier défectueux ne doit pas arrêter tout le corpus
            del rows_out[n0:]
            failures.append((str(f), repr(e)[:300]))
            print(f"  /!\\ ÉCHEC, fichier ignoré : {e!r}", file=sys.stderr)
    if failures:
        with open(outdir / "erreurs.tsv", "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["fichier", "erreur"])
            w.writerows(failures)
        print(f"{len(failures)} fichier(s) en échec : voir {outdir}/erreurs.tsv", file=sys.stderr)

    with open(outdir / "annotations.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["fichier", "methode", "niveau", "chemin", "balise", "unite", "n_formes" if args.count_types else "n_tokens", "etiquette",
                    "xml_lang", "score", "confiance", "hits", "items_valides", "marge", "deuxieme",
                    "accepte", "motif", "langue_parente", "ecrit", "force", "texte"])
        for r in rows_out:
            d = r["dec"]
            w.writerow([r["fichier"], r["methode"], r["niveau"], r["chemin"], r["balise"], r["unite"],
                        d["n_tokens"], d["label"] or "", r["code"], "" if d["score"] is None else d["score"],
                        cert_of(d["score"], args) if d["score"] is not None else "",
                        "" if d["hits"] is None else d["hits"], "" if d["valid"] is None else d["valid"],
                        "" if d["margin"] is None else d["margin"], d["second"] or "", int(d["accepted"]),
                        d["why"], r["parent"], r["ecrit"], r["force"], r["texte"].replace("\t", " ")])
    with open(outdir / "temps.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["fichier", "methode", "secondes", "phrases", "divs"])
        w.writerows(timings)
    tot = {}
    for _f, m, sec, *_ in timings:
        tot[m] = tot.get(m, 0.0) + sec
    print("Temps de traitement par méthode (analyse + décisions, hors lecture du fichier et écriture ; "
          "chaque méthode recalcule aussi la liste de mots : voir benchmark_lid.py pour des temps "
          "isolés) : " + ", ".join(f"{m} {v:.1f} s" for m, v in tot.items()), file=sys.stderr)
    with open(outdir / "parametres.json", "w", encoding="utf-8") as fh:
        json.dump({"methodes": [m[0] for m in methods], "arguments": vars(args)}, fh,
                  ensure_ascii=False, indent=2, default=str)
    print(f"\nSorties dans {outdir}/ : {'fusion/' if args.merge else 'un dossier par méthode'}, annotations.tsv, parametres.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
